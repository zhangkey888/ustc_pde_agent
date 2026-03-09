import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc

def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD
    
    # Extract parameters
    k_val = 8.0
    if 'pde' in case_spec and 'wavenumber' in case_spec['pde']:
        k_val = float(case_spec['pde']['wavenumber'])
    
    # Extract output grid size
    nx_out = 50
    ny_out = 50
    if 'output' in case_spec:
        nx_out = case_spec['output'].get('nx', 50)
        ny_out = case_spec['output'].get('ny', 50)
    
    element_degree = 2
    mesh_resolution = 48
    
    domain = mesh.create_unit_square(comm, mesh_resolution, mesh_resolution, 
                                      cell_type=mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", element_degree))
    
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    x = ufl.SpatialCoordinate(domain)
    pi_val = np.pi
    k2 = PETSc.ScalarType(k_val**2)
    
    f_expr = (2.0 * pi_val**2 - k_val**2) * ufl.cos(pi_val * x[0]) * ufl.sin(pi_val * x[1])
    
    a = ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx - k2 * ufl.inner(u, v) * ufl.dx
    L = ufl.inner(f_expr, v) * ufl.dx
    
    tdim = domain.topology.dim
    fdim = tdim - 1
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
    )
    u_bc = fem.Function(V)
    u_bc.interpolate(lambda x: np.cos(np.pi * x[0]) * np.sin(np.pi * x[1]))
    dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc = fem.dirichletbc(u_bc, dofs)
    
    ksp_type = "preonly"
    pc_type = "lu"
    rtol = 1e-10
    
    problem = petsc.LinearProblem(
        a, L, bcs=[bc],
        petsc_options={"ksp_type": ksp_type, "pc_type": pc_type},
        petsc_options_prefix="helmholtz_"
    )
    u_sol = problem.solve()
    iterations = problem.solver.getIterationNumber()
    
    x_coords = np.linspace(0, 1, nx_out)
    y_coords = np.linspace(0, 1, ny_out)
    xx, yy = np.meshgrid(x_coords, y_coords, indexing='ij')
    points = np.zeros((nx_out * ny_out, 3))
    points[:, 0] = xx.flatten()
    points[:, 1] = yy.flatten()
    
    bb_tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, points)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points)
    
    u_values = np.full(nx_out * ny_out, np.nan)
    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    for i in range(len(points)):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(points[i])
            cells_on_proc.append(links[0])
            eval_map.append(i)
    if len(points_on_proc) > 0:
        pts_arr = np.array(points_on_proc)
        cells_arr = np.array(cells_on_proc, dtype=np.int32)
        vals = u_sol.eval(pts_arr, cells_arr)
        u_values[eval_map] = vals.flatten()
    
    u_grid = u_values.reshape((nx_out, ny_out))
    
    return {
        "u": u_grid,
        "solver_info": {
            "mesh_resolution": mesh_resolution,
            "element_degree": element_degree,
            "ksp_type": ksp_type,
            "pc_type": pc_type,
            "rtol": rtol,
            "iterations": iterations,
        }
    }

if __name__ == "__main__":
    import time
    case_spec = {"pde": {"wavenumber": 8.0}, "output": {"nx": 50, "ny": 50}}
    t0 = time.time()
    result = solve(case_spec)
    elapsed = time.time() - t0
    u_grid = result["u"]
    nx, ny = u_grid.shape
    x_coords = np.linspace(0, 1, nx)
    y_coords = np.linspace(0, 1, ny)
    xx, yy = np.meshgrid(x_coords, y_coords, indexing='ij')
    u_exact = np.cos(np.pi * xx) * np.sin(np.pi * yy)
    error = np.sqrt(np.mean((u_grid - u_exact)**2))
    max_error = np.max(np.abs(u_grid - u_exact))
    print(f"Shape: {u_grid.shape}, Time: {elapsed:.3f}s")
    print(f"L2 error: {error:.6e}, Max error: {max_error:.6e}, NaN: {np.any(np.isnan(u_grid))}")
