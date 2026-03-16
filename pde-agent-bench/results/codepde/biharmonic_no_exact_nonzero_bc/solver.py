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
    # Manufactured solution: u = sin(3*pi*x) + cos(2*pi*y)
    # We use mixed formulation: introduce w = -Δu
    # Then: -Δu = w  and  -Δw = f
    # With appropriate BCs for both u and w.
    
    # Parameters
    nx = ny = 80
    degree = 2
    
    # 2. Create mesh
    domain = mesh.create_unit_square(comm, nx, ny, cell_type=mesh.CellType.triangle)
    
    # 3. Function spaces
    V = fem.functionspace(domain, ("Lagrange", degree))
    
    # Spatial coordinates
    x = ufl.SpatialCoordinate(domain)
    pi = np.pi
    
    # Exact solution
    u_exact_expr = ufl.sin(3 * pi * x[0]) + ufl.cos(2 * pi * x[1])
    
    # Compute w_exact = -Δu_exact
    # Δu = -9π²sin(3πx) - 4π²cos(2πy)
    # w = -Δu = 9π²sin(3πx) + 4π²cos(2πy)
    w_exact_expr = 9 * pi**2 * ufl.sin(3 * pi * x[0]) + 4 * pi**2 * ufl.cos(2 * pi * x[1])
    
    # Source term f = Δ²u = Δ(-w) = -Δw
    # Δw = -81π⁴sin(3πx) - 16π⁴cos(2πy)
    # f = -Δw = 81π⁴sin(3πx) + 16π⁴cos(2πy)
    f_expr = 81 * pi**4 * ufl.sin(3 * pi * x[0]) + 16 * pi**4 * ufl.cos(2 * pi * x[1])
    
    # 4. Boundary conditions
    tdim = domain.topology.dim
    fdim = tdim - 1
    
    # All boundary facets
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
    )
    
    # BC for u: u = u_exact on ∂Ω
    u_bc_func = fem.Function(V)
    u_exact_fem_expr = fem.Expression(u_exact_expr, V.element.interpolation_points)
    u_bc_func.interpolate(u_exact_fem_expr)
    
    dofs_u = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc_u = fem.dirichletbc(u_bc_func, dofs_u)
    
    # BC for w: w = w_exact on ∂Ω
    w_bc_func = fem.Function(V)
    w_exact_fem_expr = fem.Expression(w_exact_expr, V.element.interpolation_points)
    w_bc_func.interpolate(w_exact_fem_expr)
    
    dofs_w = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc_w = fem.dirichletbc(w_bc_func, dofs_w)
    
    # 5. Mixed formulation - two sequential Poisson solves
    # Step 1: Solve -Δw = f with w = w_exact on ∂Ω
    # Weak form: ∫ grad(w)·grad(v) dx = ∫ f*v dx
    
    w_trial = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    
    a_w = ufl.inner(ufl.grad(w_trial), ufl.grad(v)) * ufl.dx
    L_w = ufl.inner(f_expr, v) * ufl.dx
    
    ksp_type = "cg"
    pc_type = "hypre"
    rtol = 1e-10
    
    petsc_opts = {
        "ksp_type": ksp_type,
        "pc_type": pc_type,
        "ksp_rtol": str(rtol),
        "ksp_max_it": "1000",
        "ksp_converged_reason": "",
    }
    
    problem_w = petsc.LinearProblem(
        a_w, L_w, bcs=[bc_w],
        petsc_options=petsc_opts,
        petsc_options_prefix="step1_"
    )
    wh = problem_w.solve()
    
    # Get iteration count for step 1
    iter1 = problem_w.solver.getIterationNumber()
    
    # Step 2: Solve -Δu = w with u = u_exact on ∂Ω
    # Weak form: ∫ grad(u)·grad(v) dx = ∫ w*v dx
    
    u_trial = ufl.TrialFunction(V)
    
    a_u = ufl.inner(ufl.grad(u_trial), ufl.grad(v)) * ufl.dx
    L_u = ufl.inner(wh, v) * ufl.dx
    
    problem_u = petsc.LinearProblem(
        a_u, L_u, bcs=[bc_u],
        petsc_options=petsc_opts,
        petsc_options_prefix="step2_"
    )
    uh = problem_u.solve()
    
    # Get iteration count for step 2
    iter2 = problem_u.solver.getIterationNumber()
    
    total_iterations = iter1 + iter2
    
    # 6. Extract solution on 50x50 uniform grid
    n_eval = 50
    xs = np.linspace(0.0, 1.0, n_eval)
    ys = np.linspace(0.0, 1.0, n_eval)
    XX, YY = np.meshgrid(xs, ys, indexing='ij')
    
    points_2d = np.zeros((3, n_eval * n_eval))
    points_2d[0, :] = XX.ravel()
    points_2d[1, :] = YY.ravel()
    points_2d[2, :] = 0.0
    
    # Point evaluation
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
    
    u_values = np.full(n_eval * n_eval, np.nan)
    if len(points_on_proc) > 0:
        pts_arr = np.array(points_on_proc)
        cells_arr = np.array(cells_on_proc, dtype=np.int32)
        vals = uh.eval(pts_arr, cells_arr)
        u_values[eval_map] = vals.flatten()
    
    u_grid = u_values.reshape((n_eval, n_eval))
    
    return {
        "u": u_grid,
        "solver_info": {
            "mesh_resolution": nx,
            "element_degree": degree,
            "ksp_type": ksp_type,
            "pc_type": pc_type,
            "rtol": rtol,
            "iterations": total_iterations,
        }
    }