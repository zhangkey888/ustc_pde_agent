import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc

def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD
    k_val = case_spec["pde"]["params"]["k"]
    
    grid_info = case_spec["output"]["grid"]
    nx_out = grid_info["nx"]
    ny_out = grid_info["ny"]
    bbox = grid_info["bbox"]
    xmin, xmax, ymin, ymax = bbox

    mesh_res = 64
    elem_degree = 2

    domain = mesh.create_unit_square(comm, mesh_res, mesh_res, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", elem_degree))

    x = ufl.SpatialCoordinate(domain)
    pi = np.pi
    t = 6.0 * (x[0] - 0.5)
    sech_t = 1.0 / ufl.cosh(t)
    tanh_t = ufl.tanh(t)
    sin_piy = ufl.sin(pi * x[1])
    
    u_exact = tanh_t * sin_piy
    d2u_dx2 = -72.0 * sech_t**2 * tanh_t * sin_piy
    d2u_dy2 = -pi**2 * tanh_t * sin_piy
    laplacian_u = d2u_dx2 + d2u_dy2
    f_expr = -laplacian_u - k_val**2 * u_exact
    
    f_func = fem.Function(V)
    f_func.interpolate(fem.Expression(f_expr, V.element.interpolation_points))

    g_func = fem.Function(V)
    g_func.interpolate(fem.Expression(u_exact, V.element.interpolation_points))

    tdim = domain.topology.dim
    fdim = tdim - 1
    boundary_facets = mesh.locate_entities_boundary(domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool))
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc = fem.dirichletbc(g_func, boundary_dofs)

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    a = ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx - k_val**2 * ufl.inner(u, v) * ufl.dx
    L = ufl.inner(f_func, v) * ufl.dx

    rtol = 1e-10
    problem = petsc.LinearProblem(
        a, L, bcs=[bc],
        petsc_options_prefix="helmholtz_",
        petsc_options={
            "ksp_type": "preonly",
            "pc_type": "lu",
            "pc_factor_mat_solver_type": "mumps",
        }
    )
    u_sol = problem.solve()
    u_sol.x.scatter_forward()

    xs = np.linspace(xmin, xmax, nx_out)
    ys = np.linspace(ymin, ymax, ny_out)
    XX, YY = np.meshgrid(xs, ys)
    points = np.zeros((nx_out * ny_out, 3))
    points[:, 0] = XX.ravel()
    points[:, 1] = YY.ravel()
    points[:, 2] = 0.0

    bb_tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, points)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points)

    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    for i in range(points.shape[0]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(points[i])
            cells_on_proc.append(links[0])
            eval_map.append(i)

    u_values = np.full((points.shape[0],), np.nan, dtype=np.float64)
    if len(points_on_proc) > 0:
        vals = u_sol.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u_values[eval_map] = vals.flatten()

    u_grid = u_values.reshape(ny_out, nx_out)

    if comm.Get_size() > 1:
        u_grid_global = np.zeros_like(u_grid)
        comm.Allreduce(u_grid, u_grid_global, op=MPI.SUM)
        u_grid = u_grid_global

    solver_info = {
        "mesh_resolution": mesh_res,
        "element_degree": elem_degree,
        "ksp_type": "preonly",
        "pc_type": "lu",
        "rtol": rtol,
        "iterations": 1,
    }

    return {"u": u_grid, "solver_info": solver_info}
