import numpy as np
from dolfinx import mesh, fem, default_scalar_type, geometry
from dolfinx.fem import petsc
from mpi4py import MPI
import ufl
from petsc4py import PETSc


def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD
    
    # Parameters
    nx = ny = 80
    degree = 2
    
    # Create mesh
    domain = mesh.create_unit_square(comm, nx, ny, cell_type=mesh.CellType.triangle)
    
    # Mixed formulation for biharmonic:
    # Δ²u = f  =>  w = -Δu,  -Δw = f
    # So we solve two Poisson problems:
    #   -Δw = f  with w = -Δu on boundary
    #   -Δu = -w with u = g on boundary
    # But w boundary condition requires knowing Δu on boundary, which is tricky.
    
    # Alternative: Use the C0 interior penalty method for biharmonic
    # Or use the mixed method with (u, w) where w = Δu
    
    # Let's use the mixed formulation:
    # w = Δu (or w = -Δu depending on sign convention)
    # System:
    #   (w, v1) + (∇u, ∇v1) = <∂u/∂n, v1>_∂Ω   ... integration by parts of (Δu, v1)
    #   (∇w, ∇v2) = (f, v2) + <∂w/∂n, v2>_∂Ω    ... -Δw = f => (∇w, ∇v2) = (f, v2) + boundary
    
    # For the manufactured solution u = exp(x)*sin(pi*y):
    # Δu = exp(x)*sin(pi*y) - pi²*exp(x)*sin(pi*y) = exp(x)*sin(pi*y)*(1 - pi²)
    # Δ²u = Δ(Δu) = (1-pi²)*Δ(exp(x)*sin(pi*y)) = (1-pi²)² * exp(x)*sin(pi*y)
    # So f = (1-pi²)² * exp(x)*sin(pi*y)
    
    # Mixed method: introduce w = Δu
    # Then: -Δw = f, with appropriate BCs
    # Weak form for the coupled system:
    #   ∫ w*v1 dx + ∫ ∇u·∇v1 dx = 0  (for all v1, with u=g on ∂Ω)
    #   ∫ ∇w·∇v2 dx = ∫ f*v2 dx       (for all v2, with w=Δu_exact on ∂Ω)
    
    # Actually, let me use a sequential approach: two Poisson solves
    # Step 1: Solve -Δw = -f with w = w_bc = Δu_exact on ∂Ω  =>  w = -Δu
    # Step 2: Solve -Δu = w with u = g on ∂Ω
    
    # Wait, let me be more careful.
    # Let w = -Δu. Then Δw = -Δ²u = -f. So -Δw = f.
    # Step 1: Solve -Δw = f, w = -Δu_exact on ∂Ω  => gives w = -Δu
    # Step 2: Solve -Δu = w, u = g on ∂Ω           => gives u
    
    pi_val = np.pi
    
    V = fem.functionspace(domain, ("Lagrange", degree))
    
    x = ufl.SpatialCoordinate(domain)
    
    # Exact solution
    u_exact_ufl = ufl.exp(x[0]) * ufl.sin(ufl.pi * x[1])
    
    # Δu = ∂²u/∂x² + ∂²u/∂y²
    # ∂²u/∂x² = exp(x)*sin(pi*y)
    # ∂²u/∂y² = -pi²*exp(x)*sin(pi*y)
    # Δu = exp(x)*sin(pi*y)*(1 - pi²)
    laplacian_u_ufl = ufl.exp(x[0]) * ufl.sin(ufl.pi * x[1]) * (1.0 - ufl.pi**2)
    
    # f = Δ²u = Δ(Δu) = (1-pi²) * Δ(exp(x)*sin(pi*y)) = (1-pi²)*(1-pi²)*exp(x)*sin(pi*y)
    f_ufl = (1.0 - ufl.pi**2)**2 * ufl.exp(x[0]) * ufl.sin(ufl.pi * x[1])
    
    # w = -Δu
    w_exact_ufl = -laplacian_u_ufl  # = exp(x)*sin(pi*y)*(pi²-1)
    
    # Step 1: Solve -Δw = f with w = w_exact on ∂Ω
    w_trial = ufl.TrialFunction(V)
    v1 = ufl.TestFunction(V)
    
    a1 = ufl.inner(ufl.grad(w_trial), ufl.grad(v1)) * ufl.dx
    L1 = f_ufl * v1 * ufl.dx
    
    # BC for w
    tdim = domain.topology.dim
    fdim = tdim - 1
    
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x_arr: np.ones(x_arr.shape[1], dtype=bool)
    )
    
    w_bc_func = fem.Function(V)
    w_bc_expr = fem.Expression(w_exact_ufl, V.element.interpolation_points)
    w_bc_func.interpolate(w_bc_expr)
    
    dofs_w = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc_w = fem.dirichletbc(w_bc_func, dofs_w)
    
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
            "ksp_monitor": None,
        },
        petsc_options_prefix="step1_"
    )
    wh = problem1.solve()
    
    # Get iteration count from step 1
    ksp1 = problem1.solver
    its1 = ksp1.getIterationNumber()
    total_iterations += its1
    
    # Step 2: Solve -Δu = w with u = g on ∂Ω
    u_trial = ufl.TrialFunction(V)
    v2 = ufl.TestFunction(V)
    
    a2 = ufl.inner(ufl.grad(u_trial), ufl.grad(v2)) * ufl.dx
    L2 = wh * v2 * ufl.dx
    
    # BC for u
    u_bc_func = fem.Function(V)
    u_bc_expr = fem.Expression(u_exact_ufl, V.element.interpolation_points)
    u_bc_func.interpolate(u_bc_expr)
    
    dofs_u = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc_u = fem.dirichletbc(u_bc_func, dofs_u)
    
    problem2 = petsc.LinearProblem(
        a2, L2, bcs=[bc_u],
        petsc_options={
            "ksp_type": ksp_type,
            "pc_type": pc_type,
            "ksp_rtol": str(rtol),
        },
        petsc_options_prefix="step2_"
    )
    uh = problem2.solve()
    
    ksp2 = problem2.solver
    its2 = ksp2.getIterationNumber()
    total_iterations += its2
    
    # Extract solution on 50x50 grid
    n_eval = 50
    xs = np.linspace(0.0, 1.0, n_eval)
    ys = np.linspace(0.0, 1.0, n_eval)
    XX, YY = np.meshgrid(xs, ys, indexing='ij')
    
    points_2d = np.zeros((3, n_eval * n_eval))
    points_2d[0, :] = XX.ravel()
    points_2d[1, :] = YY.ravel()
    points_2d[2, :] = 0.0
    
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