import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc


def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD
    
    # Parameters
    nx = ny = 64
    degree = 2
    ksp_type = "cg"
    pc_type = "hypre"
    rtol = 1e-10
    
    # Create mesh
    domain = mesh.create_unit_square(comm, nx, ny, cell_type=mesh.CellType.triangle)
    
    # Manufactured solution: u = sin(3*pi*x) + cos(2*pi*y)
    # Laplacian: -(3*pi)^2 * sin(3*pi*x) - (2*pi)^2 * cos(2*pi*y)
    #          = -9*pi^2*sin(3*pi*x) - 4*pi^2*cos(2*pi*y)
    # Bilaplacian (Laplacian of Laplacian):
    # Lap of (-9*pi^2*sin(3*pi*x)) = -9*pi^2 * (-(3*pi)^2 * sin(3*pi*x)) = 81*pi^4*sin(3*pi*x)
    # Lap of (-4*pi^2*cos(2*pi*y)) = -4*pi^2 * (-(2*pi)^2 * cos(2*pi*y)) = 16*pi^4*cos(2*pi*y)
    # f = 81*pi^4*sin(3*pi*x) + 16*pi^4*cos(2*pi*y)
    
    # Mixed formulation: introduce w = -Lap(u)
    # -Lap(u) = w  =>  Lap(u) + w = 0
    # -Lap(w) = f  =>  Lap(w) + f = 0  (but we want Lap^2 u = f, so -Lap(w) = f)
    # 
    # Actually: Δ²u = f means Δ(Δu) = f
    # Let w = Δu (or w = -Δu depending on sign convention)
    # 
    # Using w = -Δu:
    #   -Δu = w   =>  (grad u, grad v) = (w, v)  for all v  (+ BCs)
    #   -Δw = -f  =>  Hmm, let me be more careful.
    #
    # Δ²u = f
    # Let w = -Δu. Then -Δw = Δ²u = f, so -Δw = f.
    #
    # System:
    #   -Δu = w   with u = g on ∂Ω
    #   -Δw = f   with w = -Δu_exact on ∂Ω
    #
    # Weak forms:
    #   (grad w, grad v) = (f, v)  with w = w_bc on ∂Ω
    #   (grad u, grad v) = (w, v)  with u = g on ∂Ω
    
    # First solve for w, then solve for u
    
    V = fem.functionspace(domain, ("Lagrange", degree))
    
    x = ufl.SpatialCoordinate(domain)
    pi = np.pi
    
    # Exact solution
    u_exact_expr = ufl.sin(3 * pi * x[0]) + ufl.cos(2 * pi * x[1])
    
    # w_exact = -Δu_exact = 9*pi^2*sin(3*pi*x) + 4*pi^2*cos(2*pi*y)
    w_exact_expr = 9 * pi**2 * ufl.sin(3 * pi * x[0]) + 4 * pi**2 * ufl.cos(2 * pi * x[1])
    
    # Source term f = Δ²u = 81*pi^4*sin(3*pi*x) + 16*pi^4*cos(2*pi*y)
    f_expr = 81 * pi**4 * ufl.sin(3 * pi * x[0]) + 16 * pi**4 * ufl.cos(2 * pi * x[1])
    
    # --- Step 1: Solve -Δw = f with w = w_exact on ∂Ω ---
    tdim = domain.topology.dim
    fdim = tdim - 1
    
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
    )
    
    # BC for w
    w_bc_func = fem.Function(V)
    w_bc_ufl_expr = fem.Expression(w_exact_expr, V.element.interpolation_points)
    w_bc_func.interpolate(w_bc_ufl_expr)
    
    dofs_w = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc_w = fem.dirichletbc(w_bc_func, dofs_w)
    
    w_trial = ufl.TrialFunction(V)
    v_test = ufl.TestFunction(V)
    
    a_w = ufl.inner(ufl.grad(w_trial), ufl.grad(v_test)) * ufl.dx
    L_w = ufl.inner(f_expr, v_test) * ufl.dx
    
    total_iterations = 0
    
    problem_w = petsc.LinearProblem(
        a_w, L_w, bcs=[bc_w],
        petsc_options={
            "ksp_type": ksp_type,
            "pc_type": pc_type,
            "ksp_rtol": str(rtol),
            "ksp_max_it": "1000",
        },
        petsc_options_prefix="solve_w_"
    )
    w_sol = problem_w.solve()
    
    # Get iteration count
    ksp_w = problem_w.solver
    total_iterations += ksp_w.getIterationNumber()
    
    # --- Step 2: Solve -Δu = w with u = g on ∂Ω ---
    u_bc_func = fem.Function(V)
    u_bc_ufl_expr = fem.Expression(u_exact_expr, V.element.interpolation_points)
    u_bc_func.interpolate(u_bc_ufl_expr)
    
    dofs_u = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc_u = fem.dirichletbc(u_bc_func, dofs_u)
    
    u_trial = ufl.TrialFunction(V)
    
    a_u = ufl.inner(ufl.grad(u_trial), ufl.grad(v_test)) * ufl.dx
    L_u = ufl.inner(w_sol, v_test) * ufl.dx
    
    problem_u = petsc.LinearProblem(
        a_u, L_u, bcs=[bc_u],
        petsc_options={
            "ksp_type": ksp_type,
            "pc_type": pc_type,
            "ksp_rtol": str(rtol),
            "ksp_max_it": "1000",
        },
        petsc_options_prefix="solve_u_"
    )
    u_sol = problem_u.solve()
    
    ksp_u = problem_u.solver
    total_iterations += ksp_u.getIterationNumber()
    
    # --- Evaluate on 50x50 grid ---
    n_eval = 50
    xs = np.linspace(0, 1, n_eval)
    ys = np.linspace(0, 1, n_eval)
    XX, YY = np.meshgrid(xs, ys, indexing='ij')
    
    points_2d = np.zeros((3, n_eval * n_eval))
    points_2d[0, :] = XX.ravel()
    points_2d[1, :] = YY.ravel()
    
    bb_tree = geometry.bb_tree(domain, tdim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, points_2d.T)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points_2d.T)
    
    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    for i in range(n_eval * n_eval):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(points_2d[:, i])
            cells_on_proc.append(links[0])
            eval_map.append(i)
    
    u_values = np.full(n_eval * n_eval, np.nan)
    if len(points_on_proc) > 0:
        pts_arr = np.array(points_on_proc)
        cells_arr = np.array(cells_on_proc, dtype=np.int32)
        vals = u_sol.eval(pts_arr, cells_arr)
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