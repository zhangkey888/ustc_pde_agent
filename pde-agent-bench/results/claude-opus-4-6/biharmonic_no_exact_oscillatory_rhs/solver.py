import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc


def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD
    
    # Parameters
    nx = ny = 80
    degree = 2
    ksp_type = "cg"
    pc_type = "hypre"
    rtol = 1e-10
    
    # Create mesh
    domain = mesh.create_unit_square(comm, nx, ny, cell_type=mesh.CellType.triangle)
    
    # Function space
    V = fem.functionspace(domain, ("Lagrange", degree))
    
    # Mixed formulation for biharmonic:
    # Δ²u = f  =>  -Δu = w,  -Δw = f
    # with u = 0 on ∂Ω (given), and w = -Δu on ∂Ω (we set w = 0 as natural/assumed)
    
    # Actually, for the mixed method:
    # Find (u, w) such that:
    #   -Δu = w  in Ω,  u = g on ∂Ω
    #   -Δw = f  in Ω,  w = 0 on ∂Ω (assuming simply supported plate: u=0, Δu=0 on boundary)
    
    # The problem states u = g on ∂Ω. Since no g is specified explicitly beyond the BC,
    # and the case is "biharmonic_no_exact", we assume g = 0 (homogeneous Dirichlet).
    # For simply supported: u = 0 and Δu = 0 on ∂Ω => w = -Δu = 0 on ∂Ω
    
    # Step 1: Solve -Δw = f with w = 0 on ∂Ω
    # Step 2: Solve -Δu = w with u = 0 on ∂Ω
    
    x = ufl.SpatialCoordinate(domain)
    f_expr = ufl.sin(10 * ufl.pi * x[0]) * ufl.sin(8 * ufl.pi * x[1])
    
    # Boundary conditions: all boundary, homogeneous
    tdim = domain.topology.dim
    fdim = tdim - 1
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
    )
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    
    bc_zero = fem.dirichletbc(PETSc.ScalarType(0.0), boundary_dofs, V)
    
    total_iterations = 0
    
    # Step 1: Solve -Δw = f, w = 0 on ∂Ω
    w_sol = fem.Function(V)
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
            "ksp_max_it": "2000",
            "ksp_converged_reason": "",
        },
        petsc_options_prefix="step1_",
    )
    w_sol = problem1.solve()
    
    # Get iteration count from step 1
    ksp1 = problem1.solver
    its1 = ksp1.getIterationNumber()
    total_iterations += its1
    
    # Step 2: Solve -Δu = w, u = 0 on ∂Ω
    u_sol = fem.Function(V)
    u_trial = ufl.TrialFunction(V)
    
    a2 = ufl.inner(ufl.grad(u_trial), ufl.grad(v_test)) * ufl.dx
    L2 = ufl.inner(w_sol, v_test) * ufl.dx
    
    problem2 = petsc.LinearProblem(
        a2, L2, bcs=[bc_zero],
        petsc_options={
            "ksp_type": ksp_type,
            "pc_type": pc_type,
            "ksp_rtol": str(rtol),
            "ksp_max_it": "2000",
            "ksp_converged_reason": "",
        },
        petsc_options_prefix="step2_",
    )
    u_sol = problem2.solve()
    
    ksp2 = problem2.solver
    its2 = ksp2.getIterationNumber()
    total_iterations += its2
    
    # Evaluate on 50x50 grid
    n_grid = 50
    xs = np.linspace(0, 1, n_grid)
    ys = np.linspace(0, 1, n_grid)
    XX, YY = np.meshgrid(xs, ys, indexing='ij')
    
    points = np.zeros((3, n_grid * n_grid))
    points[0, :] = XX.ravel()
    points[1, :] = YY.ravel()
    
    # Point evaluation
    bb_tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, points.T)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points.T)
    
    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    for i in range(n_grid * n_grid):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(points[:, i])
            cells_on_proc.append(links[0])
            eval_map.append(i)
    
    u_values = np.full(n_grid * n_grid, np.nan)
    if len(points_on_proc) > 0:
        pts_arr = np.array(points_on_proc)
        cells_arr = np.array(cells_on_proc, dtype=np.int32)
        vals = u_sol.eval(pts_arr, cells_arr)
        u_values[eval_map] = vals.flatten()
    
    u_grid = u_values.reshape((n_grid, n_grid))
    
    return {
        "u": u_grid,
        "solver_info": {
            "mesh_resolution": nx,
            "element_degree": degree,
            "ksp_type": ksp_type,
            "pc_type": pc_type,
            "rtol": rtol,
            "iterations": total_iterations,
        },
    }