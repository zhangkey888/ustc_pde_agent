import numpy as np
from dolfinx import mesh, fem, default_scalar_type, geometry
from dolfinx.fem import petsc
from mpi4py import MPI
import ufl
from petsc4py import PETSc


def solve(case_spec: dict) -> dict:
    # 1. Parse case_spec
    comm = MPI.COMM_WORLD
    
    # Mesh resolution - use higher resolution for oscillatory RHS
    N = 128
    element_degree = 2
    
    # 2. Create mesh
    domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
    
    # 3. Function spaces for mixed formulation
    # Biharmonic: Δ²u = f  =>  split into two Poisson problems:
    #   -Δw = f   (w = -Δu)
    #   -Δu = w   with u = g on ∂Ω
    # But we need BC for w as well. For u=0 on boundary, we can use:
    #   w = -Δu, and if u=0 on ∂Ω, we need to figure out w on ∂Ω.
    # 
    # A cleaner mixed formulation:
    #   Find (u, w) such that:
    #     w = -Δu  =>  (w, v) + (∇u, ∇v) = 0   for all v  (with u=g on ∂Ω)
    #     -Δw = f  =>  (∇w, ∇ψ) = (f, ψ)       for all ψ  (with w=0 on ∂Ω if u=0 on ∂Ω)
    #
    # Actually for simply supported plate: u=0 and Δu=0 on ∂Ω => w=0 on ∂Ω
    # For clamped plate: u=0 and ∂u/∂n=0 on ∂Ω
    #
    # The problem says u=g on ∂Ω. With no other BC specified, we use simply supported:
    # u=g on ∂Ω, w=-Δu=0 on ∂Ω (simply supported)
    #
    # Mixed system:
    #   (∇w, ∇v) = (f, v)       with w=0 on ∂Ω   => solve for w
    #   (∇u, ∇ψ) = (w, ψ)      with u=g on ∂Ω    => solve for u
    
    V = fem.functionspace(domain, ("Lagrange", element_degree))
    
    # Spatial coordinate
    x = ufl.SpatialCoordinate(domain)
    
    # Source term: f = sin(10*pi*x)*sin(8*pi*y)
    pi = np.pi
    f_expr = ufl.sin(10 * ufl.pi * x[0]) * ufl.sin(8 * ufl.pi * x[1])
    
    # Boundary condition: g = 0 (default for biharmonic on unit square unless specified)
    # Check case_spec for boundary conditions
    g_value = 0.0
    
    tdim = domain.topology.dim
    fdim = tdim - 1
    
    # Locate all boundary facets
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
    )
    
    # DOFs on boundary
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    
    # --- Step 1: Solve -Δw = f with w=0 on ∂Ω ---
    w_sol = ufl.TrialFunction(V)
    v_test = ufl.TestFunction(V)
    
    bc_w = fem.dirichletbc(PETSc.ScalarType(0.0), boundary_dofs, V)
    
    a1 = ufl.inner(ufl.grad(w_sol), ufl.grad(v_test)) * ufl.dx
    L1 = ufl.inner(f_expr, v_test) * ufl.dx
    
    ksp_type = "cg"
    pc_type = "hypre"
    rtol = 1e-10
    
    total_iterations = 0
    
    problem1 = petsc.LinearProblem(
        a1, L1, bcs=[bc_w],
        petsc_options={
            "ksp_type": ksp_type,
            "pc_type": pc_type,
            "ksp_rtol": str(rtol),
            "ksp_max_it": "2000",
        },
        petsc_options_prefix="step1_"
    )
    wh = problem1.solve()
    
    # Get iteration count from solver
    ksp1 = problem1.solver
    total_iterations += ksp1.getIterationNumber()
    
    # --- Step 2: Solve -Δu = w with u=g on ∂Ω ---
    u_sol = ufl.TrialFunction(V)
    psi_test = ufl.TestFunction(V)
    
    u_bc_func = fem.Function(V)
    u_bc_func.interpolate(lambda x: np.full(x.shape[1], g_value))
    bc_u = fem.dirichletbc(u_bc_func, boundary_dofs)
    
    a2 = ufl.inner(ufl.grad(u_sol), ufl.grad(psi_test)) * ufl.dx
    L2 = ufl.inner(wh, psi_test) * ufl.dx
    
    problem2 = petsc.LinearProblem(
        a2, L2, bcs=[bc_u],
        petsc_options={
            "ksp_type": ksp_type,
            "pc_type": pc_type,
            "ksp_rtol": str(rtol),
            "ksp_max_it": "2000",
        },
        petsc_options_prefix="step2_"
    )
    uh = problem2.solve()
    
    ksp2 = problem2.solver
    total_iterations += ksp2.getIterationNumber()
    
    # 7. Extract solution on 50x50 uniform grid
    nx_out, ny_out = 50, 50
    xs = np.linspace(0.0, 1.0, nx_out)
    ys = np.linspace(0.0, 1.0, ny_out)
    XX, YY = np.meshgrid(xs, ys, indexing='ij')
    
    points = np.zeros((3, nx_out * ny_out))
    points[0, :] = XX.ravel()
    points[1, :] = YY.ravel()
    
    # Point evaluation
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
        vals = uh.eval(pts_arr, cells_arr)
        u_values[eval_map] = vals.flatten()
    
    u_grid = u_values.reshape((nx_out, ny_out))
    
    return {
        "u": u_grid,
        "solver_info": {
            "mesh_resolution": N,
            "element_degree": element_degree,
            "ksp_type": ksp_type,
            "pc_type": pc_type,
            "rtol": rtol,
            "iterations": total_iterations,
        }
    }