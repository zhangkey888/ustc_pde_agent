import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc
import math

def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD
    
    grid_spec = case_spec["output"]["grid"]
    nx_out = grid_spec["nx"]
    ny_out = grid_spec["ny"]
    bbox = grid_spec["bbox"]
    xmin, xmax, ymin, ymax = bbox[0], bbox[1], bbox[2], bbox[3]
    
    kappa = 1.0
    pi_val = math.pi
    f_val = (pi_val**2 - 36.0)
    
    mesh_res = 280
    element_degree = 3
    
    domain = mesh.create_unit_square(comm, mesh_res, mesh_res, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", element_degree))
    
    x = ufl.SpatialCoordinate(domain)
    u_exact_expr = ufl.exp(6.0 * x[0]) * ufl.sin(ufl.pi * x[1])
    f_expr = f_val * ufl.exp(6.0 * x[0]) * ufl.sin(ufl.pi * x[1])
    
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    
    a = kappa * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = ufl.inner(f_expr, v) * ufl.dx
    
    fdim = domain.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool))
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    
    u_bc = fem.Function(V)
    u_bc.interpolate(fem.Expression(u_exact_expr, V.element.interpolation_points))
    bc = fem.dirichletbc(u_bc, boundary_dofs)
    
    ksp_type = "cg"
    pc_type = "hypre"
    rtol = 1e-10
    
    problem = petsc.LinearProblem(a, L, bcs=[bc],
        petsc_options={"ksp_type": ksp_type, "pc_type": pc_type, "ksp_rtol": rtol, "ksp_atol": 1e-12},
        petsc_options_prefix="poisson_")
    
    u_sol = problem.solve()
    iterations = problem.solver.getIterationNumber()
    
    xs = np.linspace(xmin, xmax, nx_out)
    ys = np.linspace(ymin, ymax, ny_out)
    XX, YY = np.meshgrid(xs, ys)
    pts = np.column_stack([XX.ravel(), YY.ravel(), np.zeros(nx_out * ny_out)])
    
    bb_tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, pts)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, pts)
    
    points_on_proc = []
    cells_on_proc = []
    idx_on_proc = []
    for i in range(pts.shape[0]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(pts[i])
            cells_on_proc.append(links[0])
            idx_on_proc.append(i)
    
    u_values = np.full((pts.shape[0],), np.nan)
    if len(points_on_proc) > 0:
        vals = u_sol.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u_values[idx_on_proc] = vals.flatten()
    
    u_grid = u_values.reshape(ny_out, nx_out)
    
    if comm.size > 1:
        all_grids = comm.allgather(u_grid)
        combined = np.full_like(u_grid, np.nan)
        for g in all_grids:
            mask = ~np.isnan(g)
            combined[mask] = g[mask]
        u_grid = combined
    
    if comm.rank == 0:
        exact_grid = np.exp(6.0 * XX) * np.sin(np.pi * YY)
        error_grid = u_grid - exact_grid
        valid = ~np.isnan(error_grid)
        if np.any(valid):
            l2_error = np.sqrt(np.mean(error_grid[valid]**2))
            max_error = np.max(np.abs(error_grid[valid]))
            print(f"L2 error (RMS): {l2_error:.6e}, Max error: {max_error:.6e}, Iterations: {iterations}")
    
    solver_info = {
        "mesh_resolution": mesh_res,
        "element_degree": element_degree,
        "ksp_type": ksp_type,
        "pc_type": pc_type,
        "rtol": rtol,
        "iterations": iterations,
    }
    
    return {"u": u_grid, "solver_info": solver_info}

if __name__ == "__main__":
    case_spec = {"output": {"grid": {"nx": 50, "ny": 50, "bbox": [0.0, 1.0, 0.0, 1.0]}}, "pde": {}}
    result = solve(case_spec)
    print(f"Output shape: {result['u'].shape}")
