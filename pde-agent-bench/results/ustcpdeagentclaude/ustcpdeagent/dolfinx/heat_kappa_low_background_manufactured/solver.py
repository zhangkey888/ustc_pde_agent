import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc


def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD

    # Parameters
    t0 = 0.0
    t_end = 0.1
    dt_val = 0.005  # smaller than suggested for accuracy
    n_steps = int(round((t_end - t0) / dt_val))

    # Grid output spec
    grid = case_spec["output"]["grid"]
    nx_out = grid["nx"]
    ny_out = grid["ny"]
    bbox = grid["bbox"]

    # Mesh
    N = 96
    domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
    degree = 2
    V = fem.functionspace(domain, ("Lagrange", degree))

    x = ufl.SpatialCoordinate(domain)
    t_const = fem.Constant(domain, PETSc.ScalarType(t0))
    dt_const = fem.Constant(domain, PETSc.ScalarType(dt_val))

    # Coefficient kappa
    kappa = 0.2 + ufl.exp(-120.0 * ((x[0] - 0.55) ** 2 + (x[1] - 0.45) ** 2))

    # Manufactured solution: u = exp(-t)*sin(pi*x)*sin(pi*y)
    pi = ufl.pi
    u_exact = ufl.exp(-t_const) * ufl.sin(pi * x[0]) * ufl.sin(pi * x[1])

    # Compute f = du/dt - div(kappa grad u)
    # du/dt = -exp(-t) sin(pi x) sin(pi y)
    du_dt = -ufl.exp(-t_const) * ufl.sin(pi * x[0]) * ufl.sin(pi * x[1])
    grad_u = ufl.grad(u_exact)
    div_term = ufl.div(kappa * grad_u)
    f_expr = du_dt - div_term

    # Trial/Test
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    u_n = fem.Function(V)  # previous time step
    # Initial condition: sin(pi x) sin(pi y)
    u_n.interpolate(lambda X: np.sin(np.pi * X[0]) * np.sin(np.pi * X[1]))

    # Backward Euler: (u - u_n)/dt + ... = f
    # (u, v) + dt*(kappa grad u, grad v) = (u_n, v) + dt*(f, v)
    a = u * v * ufl.dx + dt_const * ufl.inner(kappa * ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = u_n * v * ufl.dx + dt_const * f_expr * v * ufl.dx

    # Boundary condition: u = u_exact on boundary
    u_bc = fem.Function(V)
    bc_expr = fem.Expression(u_exact, V.element.interpolation_points)
    u_bc.interpolate(bc_expr)

    fdim = domain.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda X: np.ones(X.shape[1], dtype=bool)
    )
    bdofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc = fem.dirichletbc(u_bc, bdofs)

    a_form = fem.form(a)
    L_form = fem.form(L)

    A = petsc.assemble_matrix(a_form, bcs=[bc])
    A.assemble()
    b = petsc.create_vector(L_form.function_spaces)

    ksp = PETSc.KSP().create(comm)
    ksp.setOperators(A)
    ksp.setType("cg")
    pc = ksp.getPC()
    pc.setType("hypre")
    pc.setHYPREType("boomeramg")
    ksp.setTolerances(rtol=1e-10, atol=1e-12, max_it=500)

    u_sol = fem.Function(V)

    total_iters = 0
    t_current = t0
    for step in range(n_steps):
        t_current += dt_val
        t_const.value = t_current
        # Update BC
        u_bc.interpolate(bc_expr)

        # Assemble RHS
        with b.localForm() as loc:
            loc.set(0.0)
        petsc.assemble_vector(b, L_form)
        petsc.apply_lifting(b, [a_form], bcs=[[bc]])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        petsc.set_bc(b, [bc])

        ksp.solve(b, u_sol.x.petsc_vec)
        u_sol.x.scatter_forward()
        total_iters += ksp.getIterationNumber()

        u_n.x.array[:] = u_sol.x.array[:]

    # Sample on output grid
    xs = np.linspace(bbox[0], bbox[1], nx_out)
    ys = np.linspace(bbox[2], bbox[3], ny_out)
    XX, YY = np.meshgrid(xs, ys)
    pts = np.c_[XX.ravel(), YY.ravel(), np.zeros(nx_out * ny_out)]

    tree = geometry.bb_tree(domain, domain.topology.dim)
    cand = geometry.compute_collisions_points(tree, pts)
    coll = geometry.compute_colliding_cells(domain, cand, pts)

    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    for i in range(pts.shape[0]):
        links = coll.links(i)
        if len(links) > 0:
            points_on_proc.append(pts[i])
            cells_on_proc.append(links[0])
            eval_map.append(i)

    u_values = np.zeros(pts.shape[0])
    if len(points_on_proc) > 0:
        vals = u_sol.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u_values[eval_map] = vals.flatten()

    u_grid = u_values.reshape(ny_out, nx_out)

    # Initial condition grid
    u_initial = np.sin(np.pi * XX) * np.sin(np.pi * YY)

    return {
        "u": u_grid,
        "u_initial": u_initial,
        "solver_info": {
            "mesh_resolution": N,
            "element_degree": degree,
            "ksp_type": "cg",
            "pc_type": "hypre",
            "rtol": 1e-10,
            "iterations": total_iters,
            "dt": dt_val,
            "n_steps": n_steps,
            "time_scheme": "backward_euler",
        },
    }


if __name__ == "__main__":
    import time
    spec = {
        "output": {"grid": {"nx": 64, "ny": 64, "bbox": [0.0, 1.0, 0.0, 1.0]}}
    }
    t0 = time.time()
    result = solve(spec)
    elapsed = time.time() - t0
    u = result["u"]
    # Compute analytical at t=0.1
    xs = np.linspace(0, 1, 64)
    ys = np.linspace(0, 1, 64)
    XX, YY = np.meshgrid(xs, ys)
    u_ex = np.exp(-0.1) * np.sin(np.pi * XX) * np.sin(np.pi * YY)
    err = np.sqrt(np.mean((u - u_ex) ** 2))
    maxerr = np.max(np.abs(u - u_ex))
    print(f"Time: {elapsed:.2f}s, RMSE: {err:.3e}, MaxErr: {maxerr:.3e}")
    print(f"Info: {result['solver_info']}")
