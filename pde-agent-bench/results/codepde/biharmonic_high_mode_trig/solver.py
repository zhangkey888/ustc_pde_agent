import numpy as np
from dolfinx import mesh, fem, default_scalar_type, geometry
from dolfinx.fem import petsc
from mpi4py import MPI
import ufl
from petsc4py import PETSc


def solve(case_spec: dict) -> dict:
    # 1. Parse case_spec
    comm = MPI.COMM_WORLD
    
    # Biharmonic equation: Δ²u = f
    # Manufactured solution: u = sin(3*pi*x)*sin(2*pi*y)
    # Mixed formulation: introduce w = -Δu
    # Then: -Δu = w  and  -Δw = f
    # With appropriate BCs
    
    # Mesh resolution - use higher resolution for accuracy with high-mode trig
    N = 80
    degree = 2
    
    # 2. Create mesh
    domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
    
    # 3. Function spaces
    V = fem.functionspace(domain, ("Lagrange", degree))
    
    # Spatial coordinates
    x = ufl.SpatialCoordinate(domain)
    
    # Exact solution
    u_exact_expr = ufl.sin(3 * ufl.pi * x[0]) * ufl.sin(2 * ufl.pi * x[1])
    
    # Compute Laplacian of exact solution:
    # Δu = -(9π² + 4π²) sin(3πx) sin(2πy) = -13π² sin(3πx) sin(2πy)
    # So w = -Δu = 13π² sin(3πx) sin(2πy)
    w_exact_expr = 13.0 * ufl.pi**2 * ufl.sin(3 * ufl.pi * x[0]) * ufl.sin(2 * ufl.pi * x[1])
    
    # Source term f = Δ²u = Δ(Δu) = Δ(-13π² sin(3πx) sin(2πy))
    # = -13π² * (-(9π² + 4π²)) sin(3πx) sin(2πy) = 13π² * 13π² sin(3πx) sin(2πy)
    # = 169π⁴ sin(3πx) sin(2πy)
    f_expr = 169.0 * ufl.pi**4 * ufl.sin(3 * ufl.pi * x[0]) * ufl.sin(2 * ufl.pi * x[1])
    
    # Mixed formulation:
    # Step 1: Solve -Δw = f with w = w_exact on ∂Ω
    # Step 2: Solve -Δu = w with u = u_exact on ∂Ω
    
    # Since u_exact = sin(3πx)*sin(2πy), on boundary u = 0 (sin(0) or sin(nπ) = 0)
    # Similarly w_exact = 13π² sin(3πx)*sin(2πy), on boundary w = 0
    
    # --- Step 1: Solve for w ---
    w_trial = ufl.TrialFunction(V)
    v_test = ufl.TestFunction(V)
    
    a1 = ufl.inner(ufl.grad(w_trial), ufl.grad(v_test)) * ufl.dx
    L1 = ufl.inner(f_expr, v_test) * ufl.dx
    
    # BCs for w: w = 0 on all boundaries
    tdim = domain.topology.dim
    fdim = tdim - 1
    
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
    )
    
    dofs_w = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc_w = fem.dirichletbc(PETSc.ScalarType(0.0), dofs_w, V)
    
    total_iterations = 0
    ksp_type = "cg"
    pc_type = "hypre"
    rtol = 1e-10
    
    problem1 = petsc.LinearProblem(
        a1, L1, bcs=[bc_w],
        petsc_options={
            "ksp_type": ksp_type,
            "pc_type": pc_type,
            "ksp_rtol": str(rtol),
            "ksp_max_it": "1000",
        },
        petsc_options_prefix="step1_"
    )
    wh = problem1.solve()
    
    # Get iteration count from step 1
    solver1 = problem1.solver
    its1 = solver1.getIterationNumber()
    total_iterations += its1
    
    # --- Step 2: Solve for u ---
    u_trial = ufl.TrialFunction(V)
    v_test2 = ufl.TestFunction(V)
    
    a2 = ufl.inner(ufl.grad(u_trial), ufl.grad(v_test2)) * ufl.dx
    L2 = ufl.inner(wh, v_test2) * ufl.dx
    
    dofs_u = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc_u = fem.dirichletbc(PETSc.ScalarType(0.0), dofs_u, V)
    
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
    uh = problem2.solve()
    
    solver2 = problem2.solver
    its2 = solver2.getIterationNumber()
    total_iterations += its2
    
    # 7. Extract solution on 50x50 uniform grid
    nx_out, ny_out = 50, 50
    xs = np.linspace(0.0, 1.0, nx_out)
    ys = np.linspace(0.0, 1.0, ny_out)
    X, Y = np.meshgrid(xs, ys, indexing='ij')
    
    points_2d = np.zeros((3, nx_out * ny_out))
    points_2d[0, :] = X.ravel()
    points_2d[1, :] = Y.ravel()
    
    # Use geometry utilities for point evaluation
    bb_tree = geometry.bb_tree(domain, domain.topology.dim)
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
        vals = uh.eval(pts, cls)
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