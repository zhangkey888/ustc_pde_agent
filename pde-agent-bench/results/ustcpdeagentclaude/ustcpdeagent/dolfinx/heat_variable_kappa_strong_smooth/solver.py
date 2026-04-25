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
    dt_val = 0.005  # refine time step
    n_steps = int(round((t_end - t0) / dt_val))

    # Mesh
    nx_mesh = 128
    degree = 2
    domain = mesh.create_unit_square(comm, nx_mesh, nx_mesh, cell_type=mesh.CellType.triangle)

    V = fem.functionspace(domain, ("Lagrange", degree))

    x = ufl.SpatialCoordinate(domain)
    t_const = fem.Constant(domain, PETSc.ScalarType(t0))

    # Exact solution u = exp(-t)*sin(3*pi*x)*sin(2*pi*y)
    u_exact_expr = ufl.exp(-t_const) * ufl.sin(3 * ufl.pi * x[0]) * ufl.sin(2 * ufl.pi * x[1])

    # kappa
    kappa = 1 + 0.8 * ufl.sin(2 * ufl.pi * x[0]) * ufl.sin(2 * ufl.pi * x[1])

    # Source term: f = du/dt - div(kappa * grad(u))
    # Compute manually with ufl
    du_dt = -u_exact_expr
    grad_u = ufl.grad(u_exact_expr)
    flux = kappa * grad_u
    div_flux = ufl.div(flux)
    f_expr = du_dt - div_flux

    # Initial condition
    u_n = fem.Function(V)
    u_init_expr = ufl.sin(3 * ufl.pi * x[0]) * ufl.sin(2 * ufl.pi * x[1])  # t=0
    u_n.interpolate(fem.Expression(u_init_expr, V.element.interpolation_points))

    u_initial_array = None  # will compute later

    # Boundary condition function
    u_bc = fem.Function(V)
    u_bc.interpolate(fem.Expression(u_exact_expr, V.element.interpolation_points))

    fdim = domain.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
    )
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc = fem.dirichletbc(u_bc, boundary_dofs)

    # Variational form - backward Euler
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    dt_c = fem.Constant(domain, PETSc.ScalarType(dt_val))

    a = u * v * ufl.dx + dt_c * ufl.inner(kappa * ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = (u_n + dt_c * f_expr) * v * ufl.dx

    a_form = fem.form(a)
    L_form = fem.form(L)

    A = petsc.assemble_matrix(a_form, bcs=[bc])
    A.assemble()

    b = petsc.create_vector(L_form.function_spaces)

    u_sol = fem.Function(V)

    solver = PETSc.KSP().create(comm)
    solver.setOperators(A)
    solver.setType(PETSc.KSP.Type.CG)
    solver.getPC().setType(PETSc.PC.Type.HYPRE)
    solver.setTolerances(rtol=1e-10, atol=1e-12, max_it=1000)

    total_iters = 0

    # Store initial condition on output grid
    # Output grid spec
    grid_spec = case_spec["output"]["grid"]
    nx_out = grid_spec["nx"]
    ny_out = grid_spec["ny"]
    bbox = grid_spec["bbox"]

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

    points_on_proc = np.array(points_on_proc)
    cells_on_proc = np.array(cells_on_proc, dtype=np.int32)

    def sample(func):
        vals = np.full((pts.shape[0],), np.nan)
        if len(eval_map) > 0:
            v = func.eval(points_on_proc, cells_on_proc).flatten()
            for idx, i in enumerate(eval_map):
                vals[i] = v[idx]
        return vals.reshape(ny_out, nx_out)

    u_initial_array = sample(u_n)

    # Time stepping
    t = t0
    for step in range(n_steps):
        t += dt_val
        t_const.value = t

        # Update BC (u_exact_expr uses t_const)
        u_bc.interpolate(fem.Expression(u_exact_expr, V.element.interpolation_points))

        # Reassemble RHS (depends on u_n and f(t))
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

    u_grid = sample(u_sol)

    return {
        "u": u_grid,
        "u_initial": u_initial_array,
        "solver_info": {
            "mesh_resolution": nx_mesh,
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
        "output": {
            "grid": {
                "nx": 64,
                "ny": 64,
                "bbox": [0.0, 1.0, 0.0, 1.0],
            }
        }
    }
    t0 = time.time()
    res = solve(case_spec)
    elapsed = time.time() - t0
    u_grid = res["u"]
    # Compute error
    xs = np.linspace(0, 1, 64)
    ys = np.linspace(0, 1, 64)
    XX, YY = np.meshgrid(xs, ys)
    u_ex = np.exp(-0.1) * np.sin(3 * np.pi * XX) * np.sin(2 * np.pi * YY)
    err = np.sqrt(np.mean((u_grid - u_ex) ** 2))
    max_err = np.max(np.abs(u_grid - u_ex))
    print(f"Elapsed: {elapsed:.2f}s")
    print(f"RMSE: {err:.3e}, Max err: {max_err:.3e}")
    print(f"Solver info: {res['solver_info']}")
