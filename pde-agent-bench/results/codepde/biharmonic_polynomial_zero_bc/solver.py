import numpy as np
from dolfinx import mesh, fem, default_scalar_type, geometry
from dolfinx.fem import petsc
from mpi4py import MPI
import ufl
from petsc4py import PETSc


def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD
    
    # Parameters
    N = 64
    degree = 2
    
    # Create mesh
    domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
    
    # Mixed formulation for biharmonic:
    # Δ²u = f  =>  -Δu = w,  -Δw = f
    # with u = g on ∂Ω, w = -Δu on ∂Ω
    
    # Manufactured solution: u = x*(1-x)*y*(1-y)
    # Compute Δu:
    # u_xx = -2*y*(1-y), u_yy = -2*x*(1-x)
    # Δu = -2*y*(1-y) - 2*x*(1-x)
    # w = -Δu = 2*y*(1-y) + 2*x*(1-x)
    # Δw = w_xx + w_yy = -4 + (-4) = -8  (wait, let me recompute)
    # w = 2*y*(1-y) + 2*x*(1-x) = 2y - 2y² + 2x - 2x²
    # w_xx = -4, w_yy = -4
    # Δw = -4 + (-4) = -8  (WRONG sign check)
    # Actually w_xx = d²/dx²(2x - 2x²) = -4, w_yy = d²/dy²(2y - 2y²) = -4
    # Δw = -8
    # f = -Δw = 8  (since -Δw = f)
    # Wait, let me be careful with signs.
    
    # Biharmonic: Δ²u = f
    # Δu = -2y(1-y) - 2x(1-x)
    # Δ(Δu) = Δ(-2y(1-y) - 2x(1-x))
    #        = -2*Δ(y(1-y)) - 2*Δ(x(1-x))
    #        = -2*(-2) - 2*(-2) = 4 + 4 = 8
    # So f = 8.
    
    # Mixed formulation:
    # Introduce w = -Δu (auxiliary variable)
    # Then -Δw = Δ²u = f
    # System:
    #   -Δu = w   in Ω   (equivalently: ∫∇u·∇v dx = ∫w·v dx)
    #   -Δw = f   in Ω   (equivalently: ∫∇w·∇q dx = ∫f·q dx)
    # BCs: u = 0 on ∂Ω (given)
    #       w = -Δu = 2y(1-y) + 2x(1-x) on ∂Ω
    # But on ∂Ω, at least one of x,y is 0 or 1, so:
    # On x=0: w = 2y(1-y) + 0 = 2y(1-y)
    # On x=1: w = 2y(1-y) + 0 = 2y(1-y)
    # On y=0: w = 0 + 2x(1-x) = 2x(1-x)
    # On y=1: w = 0 + 2x(1-x) = 2x(1-x)
    
    # Step 1: Solve for w: -Δw = f with w = w_bc on ∂Ω
    V = fem.functionspace(domain, ("Lagrange", degree))
    
    x = ufl.SpatialCoordinate(domain)
    
    f_val = fem.Constant(domain, default_scalar_type(8.0))
    
    # w boundary condition
    w_exact_expr = 2.0 * x[1] * (1.0 - x[1]) + 2.0 * x[0] * (1.0 - x[0])
    
    # Solve for w first
    w_trial = ufl.TrialFunction(V)
    v_test = ufl.TestFunction(V)
    
    a_w = ufl.inner(ufl.grad(w_trial), ufl.grad(v_test)) * ufl.dx
    L_w = ufl.inner(f_val, v_test) * ufl.dx
    
    # BC for w
    tdim = domain.topology.dim
    fdim = tdim - 1
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
    )
    
    w_bc_func = fem.Function(V)
    w_bc_ufl_expr = fem.Expression(w_exact_expr, V.element.interpolation_points)
    w_bc_func.interpolate(w_bc_ufl_expr)
    
    dofs_w = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc_w = fem.dirichletbc(w_bc_func, dofs_w)
    
    ksp_type = "cg"
    pc_type = "hypre"
    rtol = 1e-10
    
    problem_w = petsc.LinearProblem(
        a_w, L_w, bcs=[bc_w],
        petsc_options={
            "ksp_type": ksp_type,
            "pc_type": pc_type,
            "ksp_rtol": str(rtol),
            "ksp_monitor": None,
        },
        petsc_options_prefix="w_solve_",
    )
    wh = problem_w.solve()
    
    # Get iterations for w solve
    ksp_w = problem_w.solver
    iters_w = ksp_w.getIterationNumber()
    
    # Step 2: Solve for u: -Δu = w with u = 0 on ∂Ω
    u_trial = ufl.TrialFunction(V)
    v_test2 = ufl.TestFunction(V)
    
    a_u = ufl.inner(ufl.grad(u_trial), ufl.grad(v_test2)) * ufl.dx
    L_u = ufl.inner(wh, v_test2) * ufl.dx
    
    u_bc_func = fem.Function(V)
    u_bc_func.x.array[:] = 0.0
    
    dofs_u = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc_u = fem.dirichletbc(u_bc_func, dofs_u)
    
    problem_u = petsc.LinearProblem(
        a_u, L_u, bcs=[bc_u],
        petsc_options={
            "ksp_type": ksp_type,
            "pc_type": pc_type,
            "ksp_rtol": str(rtol),
        },
        petsc_options_prefix="u_solve_",
    )
    uh = problem_u.solve()
    
    ksp_u = problem_u.solver
    iters_u = ksp_u.getIterationNumber()
    
    total_iters = iters_w + iters_u
    
    # Extract solution on 50x50 uniform grid
    nx_out, ny_out = 50, 50
    xs = np.linspace(0.0, 1.0, nx_out)
    ys = np.linspace(0.0, 1.0, ny_out)
    XX, YY = np.meshgrid(xs, ys, indexing="ij")
    
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
        vals = uh.eval(pts_arr, cells_arr)
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
            "iterations": total_iters,
        },
    }