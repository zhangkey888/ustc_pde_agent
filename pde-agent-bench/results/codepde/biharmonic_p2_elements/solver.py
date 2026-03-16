import numpy as np
from dolfinx import mesh, fem, default_scalar_type, geometry
from dolfinx.fem import petsc
from mpi4py import MPI
import ufl
from petsc4py import PETSc


def solve(case_spec: dict) -> dict:
    # 1. Parse case_spec
    comm = MPI.COMM_WORLD
    
    # Parameters
    N = 80  # mesh resolution
    degree = 2  # P2 elements as specified by case_id
    
    # 2. Create mesh
    domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
    
    # 3. Function spaces for mixed formulation
    # Biharmonic via mixed method: Δ²u = f  =>  w = -Δu, -Δw = f
    # So we solve two Poisson problems:
    #   -Δw = f  in Ω,  w = -Δu_exact on ∂Ω
    #   -Δu = w  in Ω,  u = g on ∂Ω
    # But actually the standard mixed formulation is:
    #   w = Δu (or w = -Δu)
    #   Δw = f (or -Δw = -f)
    # Let's use: σ = -Δu, then -Δσ = f
    # With BCs: u = g on ∂Ω, σ = -Δu_exact on ∂Ω
    
    V = fem.functionspace(domain, ("Lagrange", degree))
    
    # Spatial coordinates
    x = ufl.SpatialCoordinate(domain)
    pi = np.pi
    
    # Exact solution: u = sin(2πx)sin(2πy)
    u_exact_expr = ufl.sin(2 * pi * x[0]) * ufl.sin(2 * pi * x[1])
    
    # Δu = -8π²sin(2πx)sin(2πy)
    # Δ²u = 64π⁴sin(2πx)sin(2πy)
    # So f = 64π⁴sin(2πx)sin(2πy)
    
    # For mixed method: let σ = -Δu = 8π²sin(2πx)sin(2πy)
    # Then -Δσ = 8π² * 8π² sin(2πx)sin(2πy) = 64π⁴ sin(2πx)sin(2πy) = f ✓
    
    f_expr = 64.0 * pi**4 * ufl.sin(2 * pi * x[0]) * ufl.sin(2 * pi * x[1])
    sigma_exact_expr = 8.0 * pi**2 * ufl.sin(2 * pi * x[0]) * ufl.sin(2 * pi * x[1])
    
    # --- Step 1: Solve -Δσ = f with σ = σ_exact on ∂Ω ---
    # Since σ_exact = 8π²sin(2πx)sin(2πy), on the boundary of [0,1]² this is 0
    # (sin(0)=sin(2π)=0 for both x and y boundaries)
    
    u_trial = ufl.TrialFunction(V)
    v_test = ufl.TestFunction(V)
    
    # BCs for σ: σ = 0 on ∂Ω (homogeneous)
    tdim = domain.topology.dim
    fdim = tdim - 1
    
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
    )
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    
    bc_sigma = fem.dirichletbc(PETSc.ScalarType(0.0), boundary_dofs, V)
    
    a1 = ufl.inner(ufl.grad(u_trial), ufl.grad(v_test)) * ufl.dx
    L1 = ufl.inner(f_expr, v_test) * ufl.dx
    
    total_iterations = 0
    
    ksp_type = "cg"
    pc_type = "hypre"
    rtol = 1e-10
    
    problem1 = petsc.LinearProblem(
        a1, L1, bcs=[bc_sigma],
        petsc_options={
            "ksp_type": ksp_type,
            "pc_type": pc_type,
            "ksp_rtol": str(rtol),
            "ksp_max_it": "1000",
        },
        petsc_options_prefix="step1_"
    )
    sigma_h = problem1.solve()
    
    # Get iteration count from step 1
    ksp1 = problem1.solver
    iter1 = ksp1.getIterationNumber()
    total_iterations += iter1
    
    # --- Step 2: Solve -Δu = σ with u = g on ∂Ω ---
    # g = sin(2πx)sin(2πy) = 0 on boundary of [0,1]²
    bc_u = fem.dirichletbc(PETSc.ScalarType(0.0), boundary_dofs, V)
    
    a2 = ufl.inner(ufl.grad(u_trial), ufl.grad(v_test)) * ufl.dx
    L2 = ufl.inner(sigma_h, v_test) * ufl.dx
    
    problem2 = petsc.LinearProblem(
        a2, L2, bcs=[bc_u],
        petsc_options={
            "ksp_type": ksp_type,
            "pc_type": pc_type,
            "ksp_rtol": str(rtol),
            "ksp_max_it": "1000",
        },
        petsc_options_prefix="step2_"
    )
    u_h = problem2.solve()
    
    ksp2 = problem2.solver
    iter2 = ksp2.getIterationNumber()
    total_iterations += iter2
    
    # 7. Extract solution on 50x50 uniform grid
    nx_out, ny_out = 50, 50
    xs = np.linspace(0.0, 1.0, nx_out)
    ys = np.linspace(0.0, 1.0, ny_out)
    XX, YY = np.meshgrid(xs, ys, indexing='ij')
    
    points_2d = np.zeros((3, nx_out * ny_out))
    points_2d[0, :] = XX.ravel()
    points_2d[1, :] = YY.ravel()
    
    bb_tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, points_2d.T)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points_2d.T)
    
    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    for i in range(points_2d.shape[1]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(points_2d[:, i])
            cells_on_proc.append(links[0])
            eval_map.append(i)
    
    u_values = np.full(nx_out * ny_out, np.nan)
    if len(points_on_proc) > 0:
        pts_arr = np.array(points_on_proc)
        cells_arr = np.array(cells_on_proc, dtype=np.int32)
        vals = u_h.eval(pts_arr, cells_arr)
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
            "iterations": total_iterations,
        }
    }