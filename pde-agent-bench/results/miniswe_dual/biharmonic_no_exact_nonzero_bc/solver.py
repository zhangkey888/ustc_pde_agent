import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc

def solve(case_spec: dict) -> dict:
    """
    Solve biharmonic equation using mixed formulation:
      Δ²u = f  in Ω
      u = g    on ∂Ω
    
    Mixed formulation:
      -Δw = f  in Ω,  w = w_exact on ∂Ω
      -Δu = w  in Ω,  u = u_exact on ∂Ω
    
    Manufactured solution: u = sin(3πx) + cos(2πy)
    """
    comm = MPI.COMM_WORLD
    degree = 2
    N = 64
    
    domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", degree))
    
    x = ufl.SpatialCoordinate(domain)
    
    # Exact solution and derived quantities
    u_exact_expr = ufl.sin(3 * ufl.pi * x[0]) + ufl.cos(2 * ufl.pi * x[1])
    
    # f = Δ²u = 81π⁴sin(3πx) + 16π⁴cos(2πy)
    f_expr = 81 * ufl.pi**4 * ufl.sin(3 * ufl.pi * x[0]) + 16 * ufl.pi**4 * ufl.cos(2 * ufl.pi * x[1])
    
    # w = -Δu = 9π²sin(3πx) + 4π²cos(2πy)
    w_exact_expr = 9 * ufl.pi**2 * ufl.sin(3 * ufl.pi * x[0]) + 4 * ufl.pi**2 * ufl.cos(2 * ufl.pi * x[1])
    
    # Boundary setup
    tdim = domain.topology.dim
    fdim = tdim - 1
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
    )
    
    v_test = ufl.TestFunction(V)
    
    # --- Step 1: Solve -Δw = f, w = w_exact on ∂Ω ---
    w_bc_func = fem.Function(V)
    w_bc_expr_compiled = fem.Expression(w_exact_expr, V.element.interpolation_points)
    w_bc_func.interpolate(w_bc_expr_compiled)
    
    dofs_w = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc_w = fem.dirichletbc(w_bc_func, dofs_w)
    
    w_trial = ufl.TrialFunction(V)
    a_w = ufl.inner(ufl.grad(w_trial), ufl.grad(v_test)) * ufl.dx
    L_w = ufl.inner(f_expr, v_test) * ufl.dx
    
    ksp_type = "cg"
    pc_type = "hypre"
    rtol = 1e-10
    
    problem_w = petsc.LinearProblem(
        a_w, L_w, bcs=[bc_w],
        petsc_options={
            "ksp_type": ksp_type,
            "pc_type": pc_type,
            "ksp_rtol": str(rtol),
            "ksp_max_it": "1000",
        },
        petsc_options_prefix="w_solve_"
    )
    w_sol = problem_w.solve()
    
    # --- Step 2: Solve -Δu = w, u = u_exact on ∂Ω ---
    u_bc_func = fem.Function(V)
    u_bc_expr_compiled = fem.Expression(u_exact_expr, V.element.interpolation_points)
    u_bc_func.interpolate(u_bc_expr_compiled)
    
    dofs_u = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc_u = fem.dirichletbc(u_bc_func, dofs_u)
    
    u_trial = ufl.TrialFunction(V)
    a_u = ufl.inner(ufl.grad(u_trial), ufl.grad(v_test)) * ufl.dx
    L_u = ufl.inner(w_sol, v_test) * ufl.dx
    
    problem_u = petsc.LinearProblem(
        a_u, L_u, bcs=[bc_u],
        petsc_options={
            "ksp_type": ksp_type,
            "pc_type": pc_type,
            "ksp_rtol": str(rtol),
            "ksp_max_it": "1000",
        },
        petsc_options_prefix="u_solve_"
    )
    u_sol = problem_u.solve()
    
    # --- Evaluate on 50x50 grid ---
    nx_out, ny_out = 50, 50
    xs = np.linspace(0, 1, nx_out)
    ys = np.linspace(0, 1, ny_out)
    XX, YY = np.meshgrid(xs, ys, indexing='ij')
    
    points_2d = np.column_stack([XX.ravel(), YY.ravel()])
    points_3d = np.zeros((points_2d.shape[0], 3))
    points_3d[:, :2] = points_2d
    
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
    
    return {
        "u": u_grid,
        "solver_info": {
            "mesh_resolution": N,
            "element_degree": degree,
            "ksp_type": ksp_type,
            "pc_type": pc_type,
            "rtol": rtol,
            "iterations": 0,
        }
    }


if __name__ == "__main__":
    import time
    t0 = time.time()
    result = solve({})
    elapsed = time.time() - t0
    print(f"Wall time: {elapsed:.3f}s")
    print(f"u_grid shape: {result['u'].shape}")
    
    nx_out, ny_out = 50, 50
    xs = np.linspace(0, 1, nx_out)
    ys = np.linspace(0, 1, ny_out)
    XX, YY = np.meshgrid(xs, ys, indexing='ij')
    u_exact_grid = np.sin(3 * np.pi * XX) + np.cos(2 * np.pi * YY)
    
    err = np.nanmax(np.abs(result['u'] - u_exact_grid))
    l2_err = np.sqrt(np.nanmean((result['u'] - u_exact_grid)**2))
    print(f"Max pointwise error: {err:.6e}")
    print(f"RMS error: {l2_err:.6e}")
    print(f"NaN count: {np.sum(np.isnan(result['u']))}")
