import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc

def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD

    grid_info = case_spec["output"]["grid"]
    nx_out = grid_info["nx"]
    ny_out = grid_info["ny"]
    bbox = grid_info["bbox"]
    xmin, xmax, ymin, ymax = bbox

    mesh_res = 96
    elem_degree = 3

    domain = mesh.create_unit_square(comm, mesh_res, mesh_res, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", elem_degree))

    x = ufl.SpatialCoordinate(domain)
    kappa = 1.0 + 0.4 * ufl.cos(4 * ufl.pi * x[0]) * ufl.sin(2 * ufl.pi * x[1])
    u_exact_expr = ufl.sin(ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])

    grad_u_exact = ufl.grad(u_exact_expr)
    kappa_grad_u = kappa * grad_u_exact
    f_expr = -ufl.div(kappa_grad_u)

    u_trial = ufl.TrialFunction(V)
    v_test = ufl.TestFunction(V)

    f_func = fem.Function(V)
    f_interp = fem.Expression(f_expr, V.element.interpolation_points)
    f_func.interpolate(f_interp)

    a = ufl.inner(kappa * ufl.grad(u_trial), ufl.grad(v_test)) * ufl.dx
    L = ufl.inner(f_func, v_test) * ufl.dx

    fdim = domain.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool))
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)

    g_func = fem.Function(V)
    g_interp = fem.Expression(u_exact_expr, V.element.interpolation_points)
    g_func.interpolate(g_interp)

    bc = fem.dirichletbc(g_func, boundary_dofs)

    ksp_type = "preonly"
    pc_type = "lu"
    rtol = 1e-10

    problem = petsc.LinearProblem(
        a, L, bcs=[bc],
        petsc_options={
            "ksp_type": ksp_type,
            "pc_type": pc_type,
        },
        petsc_options_prefix="poisson_"
    )

    u_sol = problem.solve()
    u_sol.x.scatter_forward()

    iterations = 1

    xs = np.linspace(xmin, xmax, nx_out)
    ys = np.linspace(ymin, ymax, ny_out)
    XX, YY = np.meshgrid(xs, ys)
    pts = np.column_stack([XX.ravel(), YY.ravel(), np.zeros(nx_out * ny_out)])

    bb_tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, pts)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, pts)

    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    for i in range(pts.shape[0]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(pts[i])
            cells_on_proc.append(links[0])
            eval_map.append(i)

    u_values = np.full((nx_out * ny_out,), np.nan)
    if len(points_on_proc) > 0:
        vals = u_sol.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u_values[eval_map] = vals.flatten()

    if comm.size > 1:
        u_values_global = np.zeros_like(u_values)
        comm.Allreduce(u_values, u_values_global, op=MPI.SUM)
        mask = np.isnan(u_values_global) & ~np.isnan(u_values)
        u_values_global[mask] = u_values[mask]
        u_values = u_values_global

    u_grid = u_values.reshape(ny_out, nx_out)

    u_exact_func = fem.Function(V)
    u_exact_interp = fem.Expression(u_exact_expr, V.element.interpolation_points)
    u_exact_func.interpolate(u_exact_interp)

    error_L2 = np.sqrt(domain.comm.allreduce(
        fem.assemble_scalar(fem.form(ufl.inner(u_sol - u_exact_func, u_sol - u_exact_func) * ufl.dx)),
        op=MPI.SUM
    ))

    if comm.rank == 0:
        print(f"L2 error: {error_L2:.6e}")

    solver_info = {
        "mesh_resolution": mesh_res,
        "element_degree": elem_degree,
        "ksp_type": ksp_type,
        "pc_type": pc_type,
        "rtol": rtol,
        "iterations": int(iterations),
    }

    return {"u": u_grid, "solver_info": solver_info}

if __name__ == "__main__":
    case_spec = {
        "output": {
            "grid": {
                "nx": 50,
                "ny": 50,
                "bbox": [0.0, 1.0, 0.0, 1.0]
            }
        }
    }
    import time
    t0 = time.time()
    result = solve(case_spec)
    t1 = time.time()
    print(f"Output shape: {result['u'].shape}")
    print(f"Solver info: {result['solver_info']}")
    print(f"Wall time: {t1-t0:.3f}s")
