import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc

def solve(case_spec: dict) -> dict:
    """
    Solve biharmonic equation Δ²u = f using mixed formulation (two Poisson solves).
    Simply supported BCs: u = 0, Δu = 0 on ∂Ω.
    
    Splitting: let v = Δu
    Step 1: Δv = f with v = 0 on ∂Ω  →  -Δv = -f with v = 0
    Step 2: Δu = v with u = 0 on ∂Ω   →  -Δu = -v with u = 0
    """
    comm = MPI.COMM_WORLD
    
    # Extract output grid spec
    nx_out = case_spec["output"]["grid"]["nx"]
    ny_out = case_spec["output"]["grid"]["ny"]
    bbox = case_spec["output"]["grid"]["bbox"]
    xmin, xmax, ymin, ymax = bbox
    
    # Mesh resolution - high enough for oscillatory source
    mesh_res = 160
    
    # Create mesh
    domain = mesh.create_unit_square(comm, mesh_res, mesh_res, cell_type=mesh.CellType.triangle)
    tdim = domain.topology.dim
    fdim = tdim - 1
    
    # Function space - P3 for good accuracy with oscillatory source
    degree = 3
    V = fem.functionspace(domain, ("Lagrange", degree))
    
    # Boundary facets (all boundary for u=0)
    boundary_facets = mesh.locate_entities_boundary(domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool))
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    
    # Source term f = sin(10*pi*x)*sin(8*pi*y)
    x = ufl.SpatialCoordinate(domain)
    f_expr = ufl.sin(10 * ufl.pi * x[0]) * ufl.sin(8 * ufl.pi * x[1])
    
    # === Step 1: Solve -Δv = -f with v = 0 on ∂Ω ===
    v_trial = ufl.TrialFunction(V)
    w_test = ufl.TestFunction(V)
    
    a1 = ufl.inner(ufl.grad(v_trial), ufl.grad(w_test)) * ufl.dx
    L1 = -f_expr * w_test * ufl.dx
    
    v_bc_func = fem.Function(V)
    v_bc_func.x.array[:] = 0.0
    bc_v = fem.dirichletbc(v_bc_func, boundary_dofs)
    
    problem1 = petsc.LinearProblem(
        a1, L1, bcs=[bc_v],
        petsc_options={
            "ksp_type": "cg",
            "pc_type": "hypre",
            "pc_hypre_type": "boomeramg",
            "ksp_rtol": 1e-12,
            "ksp_atol": 1e-15,
            "ksp_max_it": 2000,
        },
        petsc_options_prefix="biharm1_"
    )
    
    v_sol = problem1.solve()
    v_sol.x.scatter_forward()
    
    its1 = problem1.solver.getIterationNumber()
    
    # === Step 2: Solve -Δu = -v with u = 0 on ∂Ω ===
    u_trial = ufl.TrialFunction(V)
    
    a2 = ufl.inner(ufl.grad(u_trial), ufl.grad(w_test)) * ufl.dx
    L2 = -v_sol * w_test * ufl.dx
    
    u_bc_func = fem.Function(V)
    u_bc_func.x.array[:] = 0.0
    bc_u = fem.dirichletbc(u_bc_func, boundary_dofs)
    
    problem2 = petsc.LinearProblem(
        a2, L2, bcs=[bc_u],
        petsc_options={
            "ksp_type": "cg",
            "pc_type": "hypre",
            "pc_hypre_type": "boomeramg",
            "ksp_rtol": 1e-12,
            "ksp_atol": 1e-15,
            "ksp_max_it": 2000,
        },
        petsc_options_prefix="biharm2_"
    )
    
    u_sol = problem2.solve()
    u_sol.x.scatter_forward()
    
    its2 = problem2.solver.getIterationNumber()
    
    # === Sample solution on output grid ===
    xs = np.linspace(xmin, xmax, nx_out)
    ys = np.linspace(ymin, ymax, ny_out)
    XX, YY = np.meshgrid(xs, ys)
    
    points = np.zeros((nx_out * ny_out, 3))
    points[:, 0] = XX.ravel()
    points[:, 1] = YY.ravel()
    
    bb_tree = geometry.bb_tree(domain, tdim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, points)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points)
    
    u_values = np.full((nx_out * ny_out,), np.nan)
    
    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    
    for i in range(points.shape[0]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(points[i])
            cells_on_proc.append(links[0])
            eval_map.append(i)
    
    if len(points_on_proc) > 0:
        pts_array = np.array(points_on_proc)
        cells_array = np.array(cells_on_proc, dtype=np.int32)
        vals = u_sol.eval(pts_array, cells_array)
        u_values[eval_map] = vals.flatten()
    
    u_grid = u_values.reshape(ny_out, nx_out)
    
    # Fill any remaining NaN (boundary edge cases)
    nan_mask = np.isnan(u_grid)
    if np.any(nan_mask):
        u_grid[nan_mask] = 0.0
    
    # Compute solver info
    total_iterations = its1 + its2
    
    solver_info = {
        "mesh_resolution": mesh_res,
        "element_degree": degree,
        "ksp_type": "cg",
        "pc_type": "hypre",
        "rtol": 1e-12,
        "iterations": total_iterations,
    }
    
    # Check if time-dependent info needed
    pde = case_spec.get("pde", {})
    if "time" in pde:
        solver_info["dt"] = 1.0
        solver_info["n_steps"] = 1
        solver_info["time_scheme"] = "none"
    
    result = {
        "u": u_grid,
        "solver_info": solver_info,
    }
    
    return result

if __name__ == "__main__":
    import time
    case_spec = {
        "output": {
            "grid": {
                "nx": 64,
                "ny": 64,
                "bbox": [0.0, 1.0, 0.0, 1.0]
            }
        },
        "pde": {}
    }
    t0 = time.time()
    result = solve(case_spec)
    t1 = time.time()
    print(f"Wall time: {t1-t0:.3f}s")
    print(f"Solution shape: {result['u'].shape}")
    print(f"Solution min: {np.nanmin(result['u']):.6e}, max: {np.nanmax(result['u']):.6e}")
    print(f"Solver info: {result['solver_info']}")
    
    # Verify against analytical solution for simply supported biharmonic
    xs = np.linspace(0, 1, 64)
    ys = np.linspace(0, 1, 64)
    XX, YY = np.meshgrid(xs, ys)
    u_exact = np.sin(10*np.pi*XX) * np.sin(8*np.pi*YY) / (164*np.pi**2)**2
    err = np.abs(result['u'] - u_exact)
    print(f"Analytical max error: {np.max(err):.6e}")
    print(f"Analytical L2 error: {np.sqrt(np.mean(err**2)):.6e}")
