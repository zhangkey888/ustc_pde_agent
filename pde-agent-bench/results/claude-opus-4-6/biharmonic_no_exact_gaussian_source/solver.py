import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc


def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD
    
    # Parameters
    N = 80
    degree = 2
    ksp_type = "cg"
    pc_type = "hypre"
    rtol = 1e-10
    
    # Create mesh
    domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
    
    # Function space
    V = fem.functionspace(domain, ("Lagrange", degree))
    
    # Spatial coordinates
    x = ufl.SpatialCoordinate(domain)
    
    # Source term: f = 10*exp(-80*((x-0.35)**2 + (y-0.55)**2))
    f_expr = 10.0 * ufl.exp(-80.0 * ((x[0] - 0.35)**2 + (x[1] - 0.55)**2))
    
    # Mixed formulation for biharmonic:
    # Δ²u = f  =>  -Δw = f,  -Δu = w  (with w = -Δu)
    # Actually: Δ²u = f means Δ(Δu) = f
    # Let w = -Δu, then -Δu = w and -Δw = -f ... let me be careful.
    #
    # Δ²u = f
    # Let w = Δu, then Δw = f
    # Step 1: Solve Δw = f with w=0 on ∂Ω (since u=g=0 on ∂Ω, we need to think about BCs for w)
    # Actually for the mixed method with u=0 on ∂Ω:
    # Let w = -Δu. Then:
    #   -Δu = w   (with u = 0 on ∂Ω)
    #   -Δw = -f  (i.e., Δw = f, but we need BC for w)
    #
    # For the standard biharmonic with u=0 and Δu=0 on ∂Ω (simply supported):
    # But the problem only specifies u=g on ∂Ω. For the biharmonic, we typically need
    # two boundary conditions. The problem states u=g on ∂Ω only.
    # 
    # Common interpretation: u=g on ∂Ω and ∂u/∂n = 0 on ∂Ω (clamped plate)
    # OR u=g and Δu=0 on ∂Ω (simply supported)
    #
    # For the mixed formulation, the natural choice is:
    # w = Δu, solve:
    #   Δw = f in Ω, w = 0 on ∂Ω  (this corresponds to Δu = 0 on ∂Ω)
    #   Δu = w in Ω, u = 0 on ∂Ω  (g=0 since no explicit g given)
    #
    # Actually looking at the problem: u = g on ∂Ω. No g is specified explicitly,
    # so g = 0 (homogeneous Dirichlet).
    # The second BC is not stated. For the mixed method, the simply supported
    # condition (Δu = 0 on ∂Ω) is natural.
    
    # Boundary conditions: u = 0 on ∂Ω
    tdim = domain.topology.dim
    fdim = tdim - 1
    
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
    )
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc_zero = fem.dirichletbc(PETSc.ScalarType(0.0), boundary_dofs, V)
    
    total_iterations = 0
    
    # Step 1: Solve -Δw = f with w = 0 on ∂Ω
    # (This gives w such that -Δw = f, i.e., w corresponds to an intermediate variable)
    # Actually let's use: w = Δu. Then Δw = Δ²u = f.
    # Weak form of Δw = f: ∫∇w·∇v dx = -∫f·v dx ... no.
    # Weak form: ∫Δw · v dx = ∫f · v dx
    # Integration by parts: -∫∇w·∇v dx + ∫(∂w/∂n)v ds = ∫f·v dx
    # With w=0 on ∂Ω (test function v=0 on ∂Ω for essential BC):
    # -∫∇w·∇v dx = ∫f·v dx
    # So: ∫∇w·∇v dx = -∫f·v dx
    #
    # Hmm, that gives a negative RHS. Let me reconsider.
    #
    # Standard approach: Let σ = -Δu. Then:
    #   σ = -Δu  =>  ∫σv dx = -∫Δu·v dx = ∫∇u·∇v dx (with u=0 or v=0 on ∂Ω)
    #   Δ²u = f  =>  -Δσ = f  =>  ∫∇σ·∇v dx = ∫f·v dx (with σ=0 on ∂Ω for simply supported)
    #
    # So:
    # Step 1: Solve ∫∇σ·∇v dx = ∫f·v dx, σ=0 on ∂Ω
    # Step 2: Solve ∫∇u·∇v dx = ∫σ·v dx, u=0 on ∂Ω
    
    # Step 1: Solve for σ
    sigma_h = fem.Function(V)
    w_trial = ufl.TrialFunction(V)
    v_test = ufl.TestFunction(V)
    
    a1 = ufl.inner(ufl.grad(w_trial), ufl.grad(v_test)) * ufl.dx
    L1 = ufl.inner(f_expr, v_test) * ufl.dx
    
    problem1 = petsc.LinearProblem(
        a1, L1, bcs=[bc_zero],
        petsc_options={
            "ksp_type": ksp_type,
            "pc_type": pc_type,
            "ksp_rtol": str(rtol),
            "ksp_max_it": "1000",
        },
        petsc_options_prefix="step1_"
    )
    sigma_h = problem1.solve()
    
    # Get iteration count for step 1
    iter1 = problem1.solver.getIterationNumber()
    total_iterations += iter1
    
    # Step 2: Solve for u
    u_h = fem.Function(V)
    a2 = ufl.inner(ufl.grad(w_trial), ufl.grad(v_test)) * ufl.dx
    L2 = ufl.inner(sigma_h, v_test) * ufl.dx
    
    problem2 = petsc.LinearProblem(
        a2, L2, bcs=[bc_zero],
        petsc_options={
            "ksp_type": ksp_type,
            "pc_type": pc_type,
            "ksp_rtol": str(rtol),
            "ksp_max_it": "1000",
        },
        petsc_options_prefix="step2_"
    )
    u_h = problem2.solve()
    
    iter2 = problem2.solver.getIterationNumber()
    total_iterations += iter2
    
    # Evaluate on 50x50 grid
    nx_out, ny_out = 50, 50
    xs = np.linspace(0.0, 1.0, nx_out)
    ys = np.linspace(0.0, 1.0, ny_out)
    XX, YY = np.meshgrid(xs, ys, indexing='ij')
    points = np.zeros((3, nx_out * ny_out))
    points[0, :] = XX.ravel()
    points[1, :] = YY.ravel()
    
    bb_tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, points.T)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points.T)
    
    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    for i in range(points.shape[1]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(points[:, i])
            cells_on_proc.append(links[0])
            eval_map.append(i)
    
    u_values = np.full(nx_out * ny_out, np.nan)
    if len(points_on_proc) > 0:
        pts_arr = np.array(points_on_proc)
        cells_arr = np.array(cells_on_proc, dtype=np.int32)
        vals = u_h.eval(pts_arr, cells_arr)
        u_values[eval_map] = vals.flatten()
    
    u_grid = u_values.reshape((nx_out, ny_out))
    
    solver_info = {
        "mesh_resolution": N,
        "element_degree": degree,
        "ksp_type": ksp_type,
        "pc_type": pc_type,
        "rtol": rtol,
        "iterations": total_iterations,
    }
    
    return {
        "u": u_grid,
        "solver_info": solver_info,
    }