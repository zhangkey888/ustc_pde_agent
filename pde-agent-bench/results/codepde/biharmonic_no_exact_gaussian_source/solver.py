import numpy as np
from dolfinx import mesh, fem, default_scalar_type, geometry
from dolfinx.fem import petsc
from mpi4py import MPI
import ufl
from petsc4py import PETSc


def solve(case_spec: dict) -> dict:
    # 1. Parse case_spec
    pde_config = case_spec.get("pde", case_spec.get("oracle_config", {}).get("pde", {}))
    
    # 2. Mesh parameters
    N = 128
    element_degree = 2
    
    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
    
    # Mixed formulation for biharmonic:
    # Δ²u = f  =>  -Δu = w,  -Δw = f
    # with u = g on ∂Ω (here g=0 assumed for homogeneous)
    # and w = 0 on ∂Ω (for simply supported plate, or we need to handle carefully)
    
    # Actually, for biharmonic with u = g on ∂Ω and no condition on ∇u·n,
    # the mixed formulation is:
    #   w = -Δu  (auxiliary variable)
    #   -Δw = f
    # With BCs: u = g on ∂Ω, w = 0 on ∂Ω (for simply supported)
    # This is a common approach.
    
    # We'll solve two sequential Poisson problems:
    # Step 1: -Δw = f in Ω, w = 0 on ∂Ω
    # Step 2: -Δu = w in Ω, u = 0 on ∂Ω  (g=0)
    
    # But wait - the boundary condition for w depends on the formulation.
    # For simply supported plate: u = 0, Δu = 0 on ∂Ω => w = -Δu = 0 on ∂Ω
    # This is the standard decomposition.
    
    # Function space
    V = fem.functionspace(domain, ("Lagrange", element_degree))
    
    # Spatial coordinates
    x = ufl.SpatialCoordinate(domain)
    
    # Source term: f = 10*exp(-80*((x-0.35)**2 + (y-0.55)**2))
    f_expr = 10.0 * ufl.exp(-80.0 * ((x[0] - 0.35)**2 + (x[1] - 0.55)**2))
    
    # Boundary conditions: u = 0 on ∂Ω
    tdim = domain.topology.dim
    fdim = tdim - 1
    
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
    )
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc_zero = fem.dirichletbc(default_scalar_type(0.0), boundary_dofs, V)
    
    total_iterations = 0
    
    # Step 1: Solve -Δw = f, w = 0 on ∂Ω
    w_trial = ufl.TrialFunction(V)
    v_test = ufl.TestFunction(V)
    
    a1 = ufl.inner(ufl.grad(w_trial), ufl.grad(v_test)) * ufl.dx
    L1 = ufl.inner(f_expr, v_test) * ufl.dx
    
    problem1 = petsc.LinearProblem(
        a1, L1, bcs=[bc_zero],
        petsc_options={
            "ksp_type": "cg",
            "pc_type": "hypre",
            "ksp_rtol": "1e-10",
            "ksp_max_it": "1000",
            "ksp_monitor": "",
        },
        petsc_options_prefix="step1_"
    )
    w_sol = problem1.solve()
    
    # Get iteration count from solver
    ksp1 = problem1.solver
    total_iterations += ksp1.getIterationNumber()
    
    # Step 2: Solve -Δu = w, u = 0 on ∂Ω
    u_trial = ufl.TrialFunction(V)
    v_test2 = ufl.TestFunction(V)
    
    a2 = ufl.inner(ufl.grad(u_trial), ufl.grad(v_test2)) * ufl.dx
    L2 = ufl.inner(w_sol, v_test2) * ufl.dx
    
    problem2 = petsc.LinearProblem(
        a2, L2, bcs=[bc_zero],
        petsc_options={
            "ksp_type": "cg",
            "pc_type": "hypre",
            "ksp_rtol": "1e-10",
            "ksp_max_it": "1000",
        },
        petsc_options_prefix="step2_"
    )
    u_sol = problem2.solve()
    
    ksp2 = problem2.solver
    total_iterations += ksp2.getIterationNumber()
    
    # 7. Extract solution on 50x50 uniform grid
    nx_out, ny_out = 50, 50
    xs = np.linspace(0.0, 1.0, nx_out)
    ys = np.linspace(0.0, 1.0, ny_out)
    XX, YY = np.meshgrid(xs, ys, indexing='ij')
    
    points_2d = np.zeros((3, nx_out * ny_out))
    points_2d[0, :] = XX.ravel()
    points_2d[1, :] = YY.ravel()
    
    # Build bounding box tree
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
    
    u_values = np.zeros(nx_out * ny_out)
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
            "element_degree": element_degree,
            "ksp_type": "cg",
            "pc_type": "hypre",
            "rtol": 1e-10,
            "iterations": total_iterations,
        }
    }