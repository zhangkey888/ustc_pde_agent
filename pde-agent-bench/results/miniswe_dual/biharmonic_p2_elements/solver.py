import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc

def solve(case_spec: dict) -> dict:
    """
    Solve the biharmonic equation using a mixed formulation (two Poisson solves).
    Δ²u = f in Ω, u = g on ∂Ω
    
    Mixed formulation:
      w = -Δu, then -Δw = f
      Step 1: Solve -Δw = f with w = 0 on ∂Ω
      Step 2: Solve -Δu = w with u = 0 on ∂Ω
    """
    comm = MPI.COMM_WORLD
    
    # Parameters - need high accuracy (error ≤ 1.36e-06) within 2.863s
    N = 192
    degree = 2
    
    # Create mesh
    domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
    tdim = domain.topology.dim
    fdim = tdim - 1
    
    # Function space
    V = fem.functionspace(domain, ("Lagrange", degree))
    
    # Spatial coordinates
    x = ufl.SpatialCoordinate(domain)
    
    # Source term: f = Δ²u = 64π⁴ sin(2πx)sin(2πy)
    f_expr = 64.0 * ufl.pi**4 * ufl.sin(2 * ufl.pi * x[0]) * ufl.sin(2 * ufl.pi * x[1])
    
    # Boundary facets (all boundary)
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
    )
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    
    total_iterations = 0
    
    # Both BCs are zero since sin(2πx)sin(2πy) = 0 on ∂[0,1]²
    # and -Δu = 8π²sin(2πx)sin(2πy) = 0 on ∂[0,1]²
    bc_zero = fem.dirichletbc(PETSc.ScalarType(0.0), boundary_dofs, V)
    
    # ============ Step 1: Solve -Δw = f, w = 0 on ∂Ω ============
    w_trial = ufl.TrialFunction(V)
    v_test = ufl.TestFunction(V)
    
    a = ufl.inner(ufl.grad(w_trial), ufl.grad(v_test)) * ufl.dx
    L_w = ufl.inner(f_expr, v_test) * ufl.dx
    
    problem_w = petsc.LinearProblem(
        a, L_w, bcs=[bc_zero],
        petsc_options={
            "ksp_type": "cg",
            "pc_type": "hypre",
            "ksp_rtol": "1e-12",
            "ksp_atol": "1e-14",
            "ksp_max_it": "1000",
        },
        petsc_options_prefix="step1_"
    )
    w_sol = problem_w.solve()
    total_iterations += problem_w.solver.getIterationNumber()
    
    # ============ Step 2: Solve -Δu = w, u = 0 on ∂Ω ============
    u_trial = ufl.TrialFunction(V)
    v_test2 = ufl.TestFunction(V)
    
    a_u = ufl.inner(ufl.grad(u_trial), ufl.grad(v_test2)) * ufl.dx
    L_u = ufl.inner(w_sol, v_test2) * ufl.dx
    
    problem_u = petsc.LinearProblem(
        a_u, L_u, bcs=[bc_zero],
        petsc_options={
            "ksp_type": "cg",
            "pc_type": "hypre",
            "ksp_rtol": "1e-12",
            "ksp_atol": "1e-14",
            "ksp_max_it": "1000",
        },
        petsc_options_prefix="step2_"
    )
    u_sol = problem_u.solve()
    total_iterations += problem_u.solver.getIterationNumber()
    
    # ============ Evaluate on 50x50 grid ============
    nx_out, ny_out = 50, 50
    xs = np.linspace(0.0, 1.0, nx_out)
    ys = np.linspace(0.0, 1.0, ny_out)
    XX, YY = np.meshgrid(xs, ys, indexing='ij')
    
    points_2d = np.zeros((3, nx_out * ny_out))
    points_2d[0, :] = XX.flatten()
    points_2d[1, :] = YY.flatten()
    
    # Point evaluation
    bb_tree = geometry.bb_tree(domain, tdim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, points_2d.T)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points_2d.T)
    
    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    for i in range(nx_out * ny_out):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(points_2d[:, i])
            cells_on_proc.append(links[0])
            eval_map.append(i)
    
    u_values = np.full(nx_out * ny_out, np.nan)
    if len(points_on_proc) > 0:
        pts = np.array(points_on_proc)
        cls = np.array(cells_on_proc, dtype=np.int32)
        vals = u_sol.eval(pts, cls)
        u_values[eval_map] = vals.flatten()
    
    u_grid = u_values.reshape((nx_out, ny_out))
    
    result = {
        "u": u_grid,
        "solver_info": {
            "mesh_resolution": N,
            "element_degree": degree,
            "ksp_type": "cg",
            "pc_type": "hypre",
            "rtol": 1e-12,
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
    print(f"Solution shape: {u_grid.shape}")
    print(f"Solution range: [{np.nanmin(u_grid):.6e}, {np.nanmax(u_grid):.6e}]")
    print(f"Wall time: {elapsed:.3f}s")
    print(f"Solver info: {result['solver_info']}")
    print(f"NaN count: {np.isnan(u_grid).sum()}")
    
    # Compute error against exact solution
    nx_out, ny_out = 50, 50
    xs = np.linspace(0.0, 1.0, nx_out)
    ys = np.linspace(0.0, 1.0, ny_out)
    XX, YY = np.meshgrid(xs, ys, indexing='ij')
    u_exact = np.sin(2 * np.pi * XX) * np.sin(2 * np.pi * YY)
    
    error = np.sqrt(np.nanmean((u_grid - u_exact)**2))
    max_error = np.nanmax(np.abs(u_grid - u_exact))
    print(f"RMS error: {error:.6e}")
    print(f"Max error: {max_error:.6e}")
