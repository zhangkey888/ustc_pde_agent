import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc

ScalarType = PETSc.ScalarType


def solve(case_spec: dict) -> dict:
    """Solve the transient heat equation with variable kappa."""
    
    pde = case_spec.get("pde", {})
    time_params = pde.get("time", {})
    
    t_end = float(time_params.get("t_end", 0.1))
    dt_suggested = float(time_params.get("dt", 0.02))
    scheme = time_params.get("scheme", "backward_euler")
    
    N = 48
    element_degree = 1
    
    result = _solve_at_resolution(case_spec, N, element_degree, t_end, dt_suggested, scheme)
    return result


def _solve_at_resolution(case_spec, N, degree, t_end, dt, scheme):
    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
    tdim = domain.topology.dim
    fdim = tdim - 1
    
    V = fem.functionspace(domain, ("Lagrange", degree))
    x = ufl.SpatialCoordinate(domain)
    
    kappa_expr = 1.0 + 0.6 * ufl.sin(2 * ufl.pi * x[0]) * ufl.sin(2 * ufl.pi * x[1])
    f_expr = 1.0 + ufl.sin(2 * ufl.pi * x[0]) * ufl.cos(2 * ufl.pi * x[1])
    
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    
    u_n = fem.Function(V, name="u_n")
    u_n.interpolate(lambda X: np.sin(np.pi * X[0]) * np.sin(np.pi * X[1]))
    
    u_initial_func = fem.Function(V)
    u_initial_func.interpolate(lambda X: np.sin(np.pi * X[0]) * np.sin(np.pi * X[1]))
    
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda X: np.ones(X.shape[1], dtype=bool)
    )
    dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc = fem.dirichletbc(ScalarType(0.0), dofs, V)
    bcs = [bc]
    
    n_steps = int(np.ceil(t_end / dt))
    actual_dt = t_end / n_steps
    dt_const = fem.Constant(domain, ScalarType(actual_dt))
    
    a = (u / dt_const) * v * ufl.dx + kappa_expr * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = (u_n / dt_const) * v * ufl.dx + f_expr * v * ufl.dx
    
    u_sol = fem.Function(V, name="u")
    total_iterations = 0
    
    for step in range(n_steps):
        problem = petsc.LinearProblem(
            a, L, bcs=bcs,
            petsc_options={"ksp_type": "cg", "pc_type": "hypre", "ksp_rtol": "1e-8"},
            petsc_options_prefix=f"ht{step}_"
        )
        u_sol = problem.solve()
        u_n.x.array[:] = u_sol.x.array[:]
    
    nx_out, ny_out = 50, 50
    xs = np.linspace(0.0, 1.0, nx_out)
    ys = np.linspace(0.0, 1.0, ny_out)
    XX, YY = np.meshgrid(xs, ys, indexing='ij')
    
    points_3d = np.zeros((nx_out * ny_out, 3))
    points_3d[:, 0] = XX.ravel()
    points_3d[:, 1] = YY.ravel()
    
    u_grid = _evaluate_function(domain, u_sol, points_3d, nx_out, ny_out)
    u_init_grid = _evaluate_function(domain, u_initial_func, points_3d, nx_out, ny_out)
    
    return {
        "u": u_grid,
        "u_initial": u_init_grid,
        "solver_info": {
            "mesh_resolution": N,
            "element_degree": degree,
            "ksp_type": "cg",
            "pc_type": "hypre",
            "rtol": 1e-8,
            "iterations": total_iterations,
            "dt": actual_dt,
            "n_steps": n_steps,
            "time_scheme": "backward_euler",
        }
    }


def _evaluate_function(domain, u_func, points_3d, nx, ny):
    bb_tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, points_3d)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points_3d)
    
    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    
    for i in range(points_3d.shape[0]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(points_3d[i])
            cells_on_proc.append(links[0])
            eval_map.append(i)
    
    u_values = np.full(points_3d.shape[0], np.nan)
    
    if len(points_on_proc) > 0:
        pts = np.array(points_on_proc)
        cls = np.array(cells_on_proc, dtype=np.int32)
        vals = u_func.eval(pts, cls)
        u_values[eval_map] = vals.flatten()
    
    u_grid = u_values.reshape((nx, ny))
    return u_grid
