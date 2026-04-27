import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc

ScalarType = PETSc.ScalarType

def solve(case_spec: dict) -> dict:
    """Solve the Helmholtz equation: -nabla^2 u - k^2 u = f with Dirichlet BCs."""
    
    k_val = case_spec.get("pde", {}).get("wavenumber", 20.0)
    
    grid_spec = case_spec["output"]["grid"]
    nx_out = grid_spec["nx"]
    ny_out = grid_spec["ny"]
    bbox = grid_spec["bbox"]
    
    element_degree = 3
    mesh_n = 80
    
    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_square(comm, mesh_n, mesh_n, cell_type=mesh.CellType.triangle)
    
    V = fem.functionspace(domain, ("Lagrange", element_degree))
    
    x = ufl.SpatialCoordinate(domain)
    pi_val = ufl.pi
    
    u_exact = ufl.sin(6 * pi_val * x[0]) * ufl.sin(5 * pi_val * x[1])
    
    k_sq = k_val * k_val
    f_expr = (36.0 * pi_val**2 + 25.0 * pi_val**2 - k_sq) * u_exact
    
    tdim = domain.topology.dim
    fdim = tdim - 1
    
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
    )
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    
    u_bc = fem.Function(V)
    u_exact_expr_interp = fem.Expression(u_exact, V.element.interpolation_points)
    u_bc.interpolate(u_exact_expr_interp)
    
    bc = fem.dirichletbc(u_bc, boundary_dofs)
    
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    
    a = ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx - k_sq * ufl.inner(u, v) * ufl.dx
    L = ufl.inner(f_expr, v) * ufl.dx
    
    problem = petsc.LinearProblem(
        a, L, bcs=[bc],
        petsc_options={
            "ksp_type": "preonly",
            "pc_type": "lu",
        },
        petsc_options_prefix="helmholtz_"
    )
    u_sol = problem.solve()
    
    error_form = fem.form(ufl.inner(u_sol - u_exact, u_sol - u_exact) * ufl.dx)
    error_local = fem.assemble_scalar(error_form)
    error_global = np.sqrt(comm.allreduce(error_local, op=MPI.SUM))
    print(f"L2 error: {error_global:.6e}")
    
    xs = np.linspace(bbox[0], bbox[1], nx_out)
    ys = np.linspace(bbox[2], bbox[3], ny_out)
    XX, YY = np.meshgrid(xs, ys)
    
    points = np.zeros((3, nx_out * ny_out))
    points[0] = XX.ravel()
    points[1] = YY.ravel()
    
    bb_tree = geometry.bb_tree(domain, tdim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, points.T)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points.T)
    
    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    
    for i in range(points.shape[1]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(points.T[i])
            cells_on_proc.append(links[0])
            eval_map.append(i)
    
    u_grid = np.full(nx_out * ny_out, np.nan)
    if len(points_on_proc) > 0:
        vals = u_sol.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u_grid[eval_map] = vals.flatten()
    
    u_grid = u_grid.reshape(ny_out, nx_out)
    
    nan_count = np.sum(np.isnan(u_grid))
    if nan_count > 0:
        print(f"WARNING: {nan_count} NaN values in output grid")
    
    solver_info = {
        "mesh_resolution": mesh_n,
        "element_degree": element_degree,
        "ksp_type": "preonly",
        "pc_type": "lu",
        "rtol": 1e-12,
        "iterations": 1,
    }
    
    return {
        "u": u_grid,
        "solver_info": solver_info,
    }


if __name__ == "__main__":
    import time
    
    case_spec = {
        "pde": {
            "wavenumber": 20.0,
        },
        "output": {
            "grid": {
                "nx": 100,
                "ny": 100,
                "bbox": [0.0, 1.0, 0.0, 1.0],
            }
        }
    }
    
    t0 = time.time()
    result = solve(case_spec)
    t1 = time.time()
    
    print(f"Wall time: {t1 - t0:.2f}s")
    print(f"Output shape: {result['u'].shape}")
    print(f"NaN count: {np.sum(np.isnan(result['u']))}")
    print(f"Min/Max: {np.nanmin(result['u']):.6f}, {np.nanmax(result['u']):.6f}")
    
    xs = np.linspace(0, 1, 100)
    ys = np.linspace(0, 1, 100)
    XX, YY = np.meshgrid(xs, ys)
    u_exact_grid = np.sin(6 * np.pi * XX) * np.sin(5 * np.pi * YY)
    
    err = np.sqrt(np.nanmean((result['u'] - u_exact_grid)**2))
    rel_err = err / np.sqrt(np.nanmean(u_exact_grid**2))
    print(f"Grid RMSE: {err:.6e}")
    print(f"Grid relative RMSE: {rel_err:.6e}")
