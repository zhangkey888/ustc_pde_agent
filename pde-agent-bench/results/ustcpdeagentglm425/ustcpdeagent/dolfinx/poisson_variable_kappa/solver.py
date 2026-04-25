import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc

def solve(case_spec: dict) -> dict:
    nx_out = case_spec["output"]["grid"]["nx"]
    ny_out = case_spec["output"]["grid"]["ny"]
    bbox = case_spec["output"]["grid"]["bbox"]
    xmin, xmax, ymin, ymax = bbox

    mesh_res = 224
    elem_degree = 2

    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_square(comm, mesh_res, mesh_res, cell_type=mesh.CellType.triangle)

    V = fem.functionspace(domain, ("Lagrange", elem_degree))

    x = ufl.SpatialCoordinate(domain)
    u_exact_ufl = ufl.sin(2*ufl.pi*x[0]) * ufl.sin(2*ufl.pi*x[1])
    kappa_ufl = 1.0 + 0.5 * ufl.sin(2*ufl.pi*x[0]) * ufl.sin(2*ufl.pi*x[1])
    grad_u_exact = ufl.grad(u_exact_ufl)
    kappa_grad_u = kappa_ufl * grad_u_exact
    f_ufl = -ufl.div(kappa_grad_u)

    kappa_func = fem.Function(V)
    kappa_func.interpolate(fem.Expression(kappa_ufl, V.element.interpolation_points))
    kappa_func.x.scatter_forward()

    f_func = fem.Function(V)
    f_func.interpolate(fem.Expression(f_ufl, V.element.interpolation_points))
    f_func.x.scatter_forward()

    g_func = fem.Function(V)
    g_func.interpolate(fem.Expression(u_exact_ufl, V.element.interpolation_points))
    g_func.x.scatter_forward()

    tdim = domain.topology.dim
    fdim = tdim - 1
    boundary_facets = mesh.locate_entities_boundary(domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool))
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc = fem.dirichletbc(g_func, boundary_dofs)

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    a = kappa_func * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = f_func * v * ufl.dx

    ksp_type = "cg"
    pc_type = "hypre"
    rtol = 1e-10

    problem = petsc.LinearProblem(
        a, L, bcs=[bc],
        petsc_options={
            "ksp_type": ksp_type,
            "pc_type": pc_type,
            "ksp_rtol": str(rtol),
            "ksp_atol": "1e-12",
            "ksp_max_it": "1000",
        },
        petsc_options_prefix="poisson_",
    )
    u_sol = problem.solve()
    u_sol.x.scatter_forward()

    ksp = problem.solver
    iterations = ksp.getIterationNumber()

    xs = np.linspace(xmin, xmax, nx_out)
    ys = np.linspace(ymin, ymax, ny_out)
    XX, YY = np.meshgrid(xs, ys)
    pts = np.column_stack([XX.ravel(), YY.ravel(), np.zeros(nx_out * ny_out)])

    bb_tree = geometry.bb_tree(domain, tdim)
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

    u_values = np.zeros((nx_out * ny_out,))
    if len(points_on_proc) > 0:
        vals = u_sol.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u_values[eval_map] = vals.flatten()

    u_grid = u_values.reshape(ny_out, nx_out)

    # Compute L2 error for verification
    u_exact_func = fem.Function(V)
    u_exact_func.interpolate(fem.Expression(u_exact_ufl, V.element.interpolation_points))
    u_exact_func.x.scatter_forward()

    error_ufl = ufl.inner(u_sol - u_exact_func, u_sol - u_exact_func) * ufl.dx
    error_form = fem.form(error_ufl)
    l2_error = np.sqrt(domain.comm.allreduce(fem.assemble_scalar(error_form), op=MPI.SUM))
    if comm.rank == 0:
        print("L2 error: {:.6e}, iterations: {}".format(l2_error, iterations))

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
    import time as _time
    t0 = _time.time()
    case_spec = {
        "output": {
            "grid": {"nx": 64, "ny": 64, "bbox": [0.0, 1.0, 0.0, 1.0]}
        },
        "pde": {"time": None}
    }
    result = solve(case_spec)
    t1 = _time.time()
    print("Output shape: {}".format(result["u"].shape))
    print("Solver info: {}".format(result["solver_info"]))
    print("Wall time: {:.3f}s".format(t1 - t0))
