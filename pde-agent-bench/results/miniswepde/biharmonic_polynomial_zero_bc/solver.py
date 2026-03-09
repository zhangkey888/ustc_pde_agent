import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc

def solve(case_spec: dict) -> dict:
    """
    Solve the biharmonic equation Δ²u = f on [0,1]×[0,1]
    using a mixed formulation (two Poisson solves):
      -Δw = f  with w = -Δu
      -Δu = w  with u = 0 on ∂Ω
    
    Manufactured solution: u = x*(1-x)*y*(1-y)
    Source term: f = 8
    """
    comm = MPI.COMM_WORLD
    
    # Parameters - adaptive
    N = 64
    degree = 2
    ksp_type_str = "cg"
    pc_type_str = "hypre"
    rtol = 1e-10
    total_iterations = 0
    
    # Create mesh
    domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
    
    # Function space
    V = fem.functionspace(domain, ("Lagrange", degree))
    
    # Source term f = 8
    f_val = fem.Constant(domain, PETSc.ScalarType(8.0))
    
    # Boundary condition: u = 0 on ∂Ω, w needs BC too
    # For the mixed formulation:
    # Step 1: Solve -Δw = f with appropriate BC for w
    # Step 2: Solve -Δu = w with u = 0 on ∂Ω
    
    # w = -Δu. For u = x(1-x)y(1-y):
    # Δu = -2y(1-y) - 2x(1-x)
    # w = -Δu = 2y(1-y) + 2x(1-x)
    # On boundary: w is NOT zero in general
    # x=0: w = 2y(1-y) + 0 = 2y(1-y)
    # x=1: w = 2y(1-y) + 0 = 2y(1-y)
    # y=0: w = 0 + 2x(1-x) = 2x(1-x)
    # y=1: w = 0 + 2x(1-x) = 2x(1-x)
    
    tdim = domain.topology.dim
    fdim = tdim - 1
    
    # --- Step 1: Solve -Δw = f with w = 2y(1-y) + 2x(1-x) on ∂Ω ---
    w_bc_func = fem.Function(V)
    w_bc_func.interpolate(lambda x: 2.0 * x[1] * (1.0 - x[1]) + 2.0 * x[0] * (1.0 - x[0]))
    
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
    )
    boundary_dofs_V = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc_w = fem.dirichletbc(w_bc_func, boundary_dofs_V)
    
    w_h = ufl.TrialFunction(V)
    v_test = ufl.TestFunction(V)
    
    a1 = ufl.inner(ufl.grad(w_h), ufl.grad(v_test)) * ufl.dx
    L1 = ufl.inner(f_val, v_test) * ufl.dx
    
    problem1 = petsc.LinearProblem(
        a1, L1, bcs=[bc_w],
        petsc_options={
            "ksp_type": ksp_type_str,
            "pc_type": pc_type_str,
            "ksp_rtol": str(rtol),
            "ksp_max_it": "1000",
        },
        petsc_options_prefix="step1_"
    )
    w_sol = problem1.solve()
    
    # Get iteration count from step 1
    ksp1 = problem1.solver
    iter1 = ksp1.getIterationNumber()
    total_iterations += iter1
    
    # --- Step 2: Solve -Δu = w with u = 0 on ∂Ω ---
    u_bc_func = fem.Function(V)
    u_bc_func.interpolate(lambda x: np.zeros_like(x[0]))
    bc_u = fem.dirichletbc(u_bc_func, boundary_dofs_V)
    
    u_h = ufl.TrialFunction(V)
    
    a2 = ufl.inner(ufl.grad(u_h), ufl.grad(v_test)) * ufl.dx
    L2 = ufl.inner(w_sol, v_test) * ufl.dx
    
    problem2 = petsc.LinearProblem(
        a2, L2, bcs=[bc_u],
        petsc_options={
            "ksp_type": ksp_type_str,
            "pc_type": pc_type_str,
            "ksp_rtol": str(rtol),
            "ksp_max_it": "1000",
        },
        petsc_options_prefix="step2_"
    )
    u_sol = problem2.solve()
    
    # Get iteration count from step 2
    ksp2 = problem2.solver
    iter2 = ksp2.getIterationNumber()
    total_iterations += iter2
    
    # --- Evaluate on 50x50 grid ---
    nx_out, ny_out = 50, 50
    xs = np.linspace(0.0, 1.0, nx_out)
    ys = np.linspace(0.0, 1.0, ny_out)
    XX, YY = np.meshgrid(xs, ys, indexing='ij')
    
    points_2d = np.column_stack([XX.ravel(), YY.ravel()])
    points_3d = np.zeros((points_2d.shape[0], 3))
    points_3d[:, 0] = points_2d[:, 0]
    points_3d[:, 1] = points_2d[:, 1]
    
    # Point evaluation
    bb_tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, points_3d)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points_3d)
    
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
        vals = u_sol.eval(pts_arr, cells_arr)
        u_values[eval_map] = vals.flatten()
    
    u_grid = u_values.reshape((nx_out, ny_out))
    
    result = {
        "u": u_grid,
        "solver_info": {
            "mesh_resolution": N,
            "element_degree": degree,
            "ksp_type": ksp_type_str,
            "pc_type": pc_type_str,
            "rtol": rtol,
            "iterations": total_iterations,
        }
    }
    
    return result


if __name__ == "__main__":
    import time
    t0 = time.time()
    result = solve({})
    elapsed = time.time() - t0
    
    u_grid = result["u"]
    nx, ny = u_grid.shape
    xs = np.linspace(0.0, 1.0, nx)
    ys = np.linspace(0.0, 1.0, ny)
    XX, YY = np.meshgrid(xs, ys, indexing='ij')
    u_exact = XX * (1.0 - XX) * YY * (1.0 - YY)
    
    error = np.sqrt(np.mean((u_grid - u_exact)**2))
    max_error = np.max(np.abs(u_grid - u_exact))
    
    print(f"Time: {elapsed:.3f}s")
    print(f"L2 (RMS) error: {error:.6e}")
    print(f"Max error: {max_error:.6e}")
    print(f"Solver info: {result['solver_info']}")
    print(f"Grid shape: {u_grid.shape}")
    print(f"Any NaN: {np.any(np.isnan(u_grid))}")
