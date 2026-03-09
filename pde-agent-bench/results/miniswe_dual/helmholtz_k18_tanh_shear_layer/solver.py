import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc

def solve(case_spec: dict) -> dict:
    """Solve Helmholtz equation: -nabla^2 u - k^2 u = f with Dirichlet BCs."""
    
    k_val = 18.0
    if 'pde' in case_spec and 'wavenumber' in case_spec['pde']:
        k_val = float(case_spec['pde']['wavenumber'])
    
    nx_out = 50
    ny_out = 50
    if 'output' in case_spec:
        nx_out = case_spec['output'].get('nx', 50)
        ny_out = case_spec['output'].get('ny', 50)
    
    element_degree = 2
    resolutions = [64, 96, 128]
    
    prev_error = None
    final_u_sol = None
    final_domain = None
    final_N = None
    
    for N in resolutions:
        comm = MPI.COMM_WORLD
        domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
        V = fem.functionspace(domain, ("Lagrange", element_degree))
        x = ufl.SpatialCoordinate(domain)
        
        u_exact_ufl = ufl.tanh(6.0 * (x[0] - 0.5)) * ufl.sin(ufl.pi * x[1])
        grad_u_ex = ufl.grad(u_exact_ufl)
        lap_u_ex = ufl.div(grad_u_ex)
        f_expr = -lap_u_ex - k_val**2 * u_exact_ufl
        
        u = ufl.TrialFunction(V)
        v = ufl.TestFunction(V)
        
        a = ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx - k_val**2 * ufl.inner(u, v) * ufl.dx
        L = ufl.inner(f_expr, v) * ufl.dx
        
        tdim = domain.topology.dim
        fdim = tdim - 1
        boundary_facets = mesh.locate_entities_boundary(
            domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
        )
        
        u_bc = fem.Function(V)
        u_bc.interpolate(lambda x: np.tanh(6.0 * (x[0] - 0.5)) * np.sin(np.pi * x[1]))
        dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
        bc = fem.dirichletbc(u_bc, dofs)
        
        problem = petsc.LinearProblem(
            a, L, bcs=[bc],
            petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
            petsc_options_prefix="helmholtz_"
        )
        u_sol = problem.solve()
        
        error_form = fem.form(ufl.inner(u_sol - u_exact_ufl, u_sol - u_exact_ufl) * ufl.dx)
        error_local = fem.assemble_scalar(error_form)
        error_global = np.sqrt(domain.comm.allreduce(error_local, op=MPI.SUM))
        
        final_u_sol = u_sol
        final_domain = domain
        final_N = N
        
        if error_global < 5e-4:
            break
        if prev_error is not None and abs(error_global - prev_error) / (abs(error_global) + 1e-15) < 0.05:
            break
        prev_error = error_global
    
    x_coords = np.linspace(0, 1, nx_out)
    y_coords = np.linspace(0, 1, ny_out)
    X, Y = np.meshgrid(x_coords, y_coords, indexing='ij')
    points_3d = np.zeros((nx_out * ny_out, 3))
    points_3d[:, 0] = X.ravel()
    points_3d[:, 1] = Y.ravel()
    
    bb_tree = geometry.bb_tree(final_domain, final_domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, points_3d)
    colliding_cells = geometry.compute_colliding_cells(final_domain, cell_candidates, points_3d)
    
    u_values = np.full(points_3d.shape[0], np.nan)
    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    
    for i in range(points_3d.shape[0]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(points_3d[i])
            cells_on_proc.append(links[0])
            eval_map.append(i)
    
    if len(points_on_proc) > 0:
        pts_arr = np.array(points_on_proc)
        cells_arr = np.array(cells_on_proc, dtype=np.int32)
        vals = final_u_sol.eval(pts_arr, cells_arr)
        u_values[eval_map] = vals.flatten()
    
    u_grid = u_values.reshape((nx_out, ny_out))
    
    return {
        "u": u_grid,
        "solver_info": {
            "mesh_resolution": final_N,
            "element_degree": element_degree,
            "ksp_type": "preonly",
            "pc_type": "lu",
            "rtol": 1e-10,
            "iterations": 1,
        }
    }


if __name__ == "__main__":
    import time
    case_spec = {"pde": {"wavenumber": 18.0}, "output": {"nx": 50, "ny": 50}}
    t0 = time.time()
    result = solve(case_spec)
    elapsed = time.time() - t0
    print(f"Elapsed time: {elapsed:.2f}s")
    print(f"Solution shape: {result['u'].shape}")
    print(f"Solver info: {result['solver_info']}")
    nx, ny = 50, 50
    xc = np.linspace(0, 1, nx)
    yc = np.linspace(0, 1, ny)
    X, Y = np.meshgrid(xc, yc, indexing='ij')
    u_exact = np.tanh(6.0 * (X - 0.5)) * np.sin(np.pi * Y)
    error = np.sqrt(np.mean((result['u'] - u_exact)**2))
    max_error = np.max(np.abs(result['u'] - u_exact))
    print(f"RMS error on grid: {error:.6e}")
    print(f"Max error on grid: {max_error:.6e}")
    print(f"Any NaN: {np.any(np.isnan(result['u']))}")
