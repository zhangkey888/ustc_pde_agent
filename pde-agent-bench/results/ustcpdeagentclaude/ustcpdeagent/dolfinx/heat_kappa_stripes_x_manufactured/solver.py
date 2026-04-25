import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc


def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD
    ScalarType = PETSc.ScalarType

    # Output grid
    grid = case_spec["output"]["grid"]
    nx_out = grid["nx"]
    ny_out = grid["ny"]
    bbox = grid["bbox"]

    # Time parameters
    t0 = 0.0
    t_end = 0.1
    dt_val = 0.005  # smaller than suggested for accuracy
    n_steps = int(round((t_end - t0) / dt_val))

    # Mesh
    N = 96
    degree = 2
    domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", degree))

    x = ufl.SpatialCoordinate(domain)
    t_const = fem.Constant(domain, ScalarType(t0))
    dt_c = fem.Constant(domain, ScalarType(dt_val))

    # kappa
    kappa = 1 + 0.5 * ufl.sin(6 * ufl.pi * x[0])

    # Exact solution: u = exp(-t)*sin(2*pi*x)*sin(pi*y)
    u_exact = ufl.exp(-t_const) * ufl.sin(2 * ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])

    # Compute f = du/dt - div(kappa * grad(u))
    # du/dt = -u_exact
    # grad(u) = exp(-t)*(2pi cos(2pi x) sin(pi y), pi sin(2pi x) cos(pi y))
    # kappa*grad(u), then divergence
    dudt = -u_exact
    flux = kappa * ufl.grad(u_exact)
    f_expr = dudt - ufl.div(flux)

    # Initial condition
    u_n = fem.Function(V)
    u_exact_t0 = ufl.sin(2 * ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])
    u_n.interpolate(fem.Expression(u_exact_t0, V.element.interpolation_points))

    # BC
    u_bc = fem.Function(V)
    u_bc.interpolate(fem.Expression(u_exact, V.element.interpolation_points))

    fdim = domain.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
    )
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc = fem.dirichletbc(u_bc, boundary_dofs)

    # Variational form: backward Euler
    # (u - u_n)/dt - div(kappa grad u) = f
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    a = (u * v / dt_c) * ufl.dx + ufl.inner(kappa * ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = (u_n * v / dt_c) * ufl.dx + f_expr * v * ufl.dx

    a_form = fem.form(a)
    L_form = fem.form(L)

    A = petsc.assemble_matrix(a_form, bcs=[bc])
    A.assemble()

    b = petsc.create_vector(L_form.function_spaces)

    solver = PETSc.KSP().create(comm)
    solver.setOperators(A)
    solver.setType(PETSc.KSP.Type.CG)
    pc = solver.getPC()
    pc.setType(PETSc.PC.Type.HYPRE)
    pc.setHYPREType("boomeramg")
    solver.setTolerances(rtol=1e-10, atol=1e-12, max_it=500)

    u_sol = fem.Function(V)
    total_iters = 0

    # Expressions for updating BC and f need time-dependent kappa/f which depend on t_const
    # We'll reassemble L each step. A is time-independent since kappa doesn't depend on t.

    t_cur = t0
    for step in range(n_steps):
        t_cur += dt_val
        t_const.value = t_cur

        # Update BC
        u_bc.interpolate(fem.Expression(u_exact, V.element.interpolation_points))

        # Assemble RHS
        with b.localForm() as loc:
            loc.set(0)
        petsc.assemble_vector(b, L_form)
        petsc.apply_lifting(b, [a_form], bcs=[[bc]])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        petsc.set_bc(b, [bc])

        solver.solve(b, u_sol.x.petsc_vec)
        u_sol.x.scatter_forward()
        total_iters += solver.getIterationNumber()

        u_n.x.array[:] = u_sol.x.array

    # Sample on output grid
    xs = np.linspace(bbox[0], bbox[1], nx_out)
    ys = np.linspace(bbox[2], bbox[3], ny_out)
    XX, YY = np.meshgrid(xs, ys)
    pts = np.c_[XX.ravel(), YY.ravel(), np.zeros(nx_out * ny_out)]

    tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(tree, pts)
    colliding = geometry.compute_colliding_cells(domain, cell_candidates, pts)

    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    for i in range(pts.shape[0]):
        links = colliding.links(i)
        if len(links) > 0:
            points_on_proc.append(pts[i])
            cells_on_proc.append(links[0])
            eval_map.append(i)

    u_vals = np.full(pts.shape[0], np.nan)
    if len(points_on_proc) > 0:
        vals = u_sol.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u_vals[eval_map] = vals.flatten()

    u_grid = u_vals.reshape(ny_out, nx_out)

    # Initial condition grid
    u_init_grid = np.exp(-t0) * np.sin(2 * np.pi * XX) * np.sin(np.pi * YY)

    # Verify accuracy against manufactured solution
    u_true_grid = np.exp(-t_end) * np.sin(2 * np.pi * XX) * np.sin(np.pi * YY)
    err = np.sqrt(np.mean((u_grid - u_true_grid) ** 2))
    print(f"L2 error vs manufactured: {err:.3e}, iters={total_iters}")

    return {
        "u": u_grid,
        "u_initial": u_init_grid,
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
    case_spec = {
        "output": {"grid": {"nx": 64, "ny": 64, "bbox": [0, 1, 0, 1]}},
        "pde": {"time": True},
    }
    t0 = time.time()
    out = solve(case_spec)
    print(f"Wall time: {time.time() - t0:.2f}s")
    print(f"u shape: {out['u'].shape}")
