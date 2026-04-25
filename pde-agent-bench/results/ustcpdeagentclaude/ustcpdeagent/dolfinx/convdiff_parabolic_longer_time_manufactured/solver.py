import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc


def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD

    # Parameters
    eps = 0.05
    beta_val = np.array([2.0, 1.0])
    t0 = 0.0
    t_end = 0.2
    dt_val = 0.005
    n_steps = int(round((t_end - t0) / dt_val))

    # Mesh
    N = 96
    degree = 2
    domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)

    V = fem.functionspace(domain, ("Lagrange", degree))
    x = ufl.SpatialCoordinate(domain)

    t_const = fem.Constant(domain, PETSc.ScalarType(t0))
    dt_c = fem.Constant(domain, PETSc.ScalarType(dt_val))
    eps_c = fem.Constant(domain, PETSc.ScalarType(eps))
    beta = fem.Constant(domain, PETSc.ScalarType((beta_val[0], beta_val[1])))

    # Manufactured solution: u = exp(-2t)*sin(pi*x)*sin(pi*y)
    def u_exact_ufl(t_expr):
        return ufl.exp(-2.0 * t_expr) * ufl.sin(ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])

    u_ex = u_exact_ufl(t_const)
    # time derivative
    dudt = -2.0 * u_ex
    grad_u = ufl.grad(u_ex)
    lap_u = -2.0 * ufl.pi**2 * u_ex  # lap(sin(pi x)sin(pi y)) = -2 pi^2 * ...
    f_expr = dudt - eps_c * lap_u + beta[0] * grad_u[0] + beta[1] * grad_u[1]

    # Initial condition
    u_n = fem.Function(V)
    u_n.interpolate(fem.Expression(
        ufl.exp(-2.0 * 0.0) * ufl.sin(ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1]),
        V.element.interpolation_points
    ))

    u_sol = fem.Function(V)
    u_sol.x.array[:] = u_n.x.array[:]

    # Variational form (backward Euler), evaluating f at t^{n+1}
    u_tr = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    a = (u_tr * v * ufl.dx
         + dt_c * eps_c * ufl.inner(ufl.grad(u_tr), ufl.grad(v)) * ufl.dx
         + dt_c * ufl.inner(beta, ufl.grad(u_tr)) * v * ufl.dx)
    L = u_n * v * ufl.dx + dt_c * f_expr * v * ufl.dx

    # Boundary conditions (Dirichlet from exact solution)
    u_bc = fem.Function(V)
    fdim = domain.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
    )
    bdofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc = fem.dirichletbc(u_bc, bdofs)

    bc_expr = fem.Expression(u_exact_ufl(t_const), V.element.interpolation_points)

    a_form = fem.form(a)
    L_form = fem.form(L)

    A = petsc.assemble_matrix(a_form, bcs=[bc])
    A.assemble()
    b = petsc.create_vector(L_form.function_spaces)

    solver = PETSc.KSP().create(comm)
    solver.setOperators(A)
    solver.setType("gmres")
    solver.getPC().setType("ilu")
    solver.setTolerances(rtol=1e-10)

    total_iters = 0
    t = t0
    u_initial_arr = None

    for step in range(n_steps):
        t_new = t + dt_val
        t_const.value = t_new
        # update BC values
        u_bc.interpolate(bc_expr)

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

        u_n.x.array[:] = u_sol.x.array[:]
        t = t_new

    # Sample on output grid
    out = case_spec["output"]["grid"]
    nx = out["nx"]; ny = out["ny"]; bbox = out["bbox"]
    xs = np.linspace(bbox[0], bbox[1], nx)
    ys = np.linspace(bbox[2], bbox[3], ny)
    XX, YY = np.meshgrid(xs, ys)
    pts = np.c_[XX.ravel(), YY.ravel(), np.zeros(nx*ny)]

    tree = geometry.bb_tree(domain, domain.topology.dim)
    cand = geometry.compute_collisions_points(tree, pts)
    coll = geometry.compute_colliding_cells(domain, cand, pts)
    cells = []
    pts_on = []
    idx_map = []
    for i in range(pts.shape[0]):
        links = coll.links(i)
        if len(links) > 0:
            pts_on.append(pts[i])
            cells.append(links[0])
            idx_map.append(i)
    u_vals = np.full(pts.shape[0], np.nan)
    if pts_on:
        vals = u_sol.eval(np.array(pts_on), np.array(cells, dtype=np.int32))
        u_vals[idx_map] = vals.flatten()
    u_grid = u_vals.reshape(ny, nx)

    # Initial condition on grid
    u_init_func = fem.Function(V)
    u_init_func.interpolate(fem.Expression(
        ufl.sin(ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1]),
        V.element.interpolation_points
    ))
    u_init_vals = np.full(pts.shape[0], np.nan)
    if pts_on:
        vals0 = u_init_func.eval(np.array(pts_on), np.array(cells, dtype=np.int32))
        u_init_vals[idx_map] = vals0.flatten()
    u_init_grid = u_init_vals.reshape(ny, nx)

    return {
        "u": u_grid,
        "u_initial": u_init_grid,
        "solver_info": {
            "mesh_resolution": N,
            "element_degree": degree,
            "ksp_type": "gmres",
            "pc_type": "ilu",
            "rtol": 1e-10,
            "iterations": total_iters,
            "dt": dt_val,
            "n_steps": n_steps,
            "time_scheme": "backward_euler",
        }
    }


if __name__ == "__main__":
    import time
    case_spec = {
        "output": {"grid": {"nx": 64, "ny": 64, "bbox": [0, 1, 0, 1]}}
    }
    t0 = time.time()
    result = solve(case_spec)
    elapsed = time.time() - t0
    u = result["u"]
    # Verify against exact
    nx, ny = 64, 64
    xs = np.linspace(0, 1, nx); ys = np.linspace(0, 1, ny)
    XX, YY = np.meshgrid(xs, ys)
    u_ex = np.exp(-2*0.2) * np.sin(np.pi*XX) * np.sin(np.pi*YY)
    err = np.sqrt(np.mean((u - u_ex)**2))
    print(f"Time: {elapsed:.3f}s, L2 err: {err:.3e}")
    print(result["solver_info"])
