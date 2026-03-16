import numpy as np
from dolfinx import mesh, fem, default_scalar_type, geometry
from dolfinx.fem import petsc
from mpi4py import MPI
import ufl
from petsc4py import PETSc


def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD
    
    # Mesh resolution and element degree
    N = 80
    degree = 2
    
    # Create mesh
    domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
    
    # Mixed formulation for biharmonic:
    # Δ²u = f  =>  -Δu = w,  -Δw = f
    # with u = g on ∂Ω, and w = -Δu on ∂Ω (from exact solution)
    
    # Spatial coordinate for UFL expressions
    x = ufl.SpatialCoordinate(domain)
    
    # Exact solution: u = tanh(6*(y-0.5))*sin(pi*x)
    pi = ufl.pi
    u_exact = ufl.tanh(6.0 * (x[1] - 0.5)) * ufl.sin(pi * x[0])
    
    # Compute w_exact = -Δu_exact
    # We need the Laplacian of u_exact
    # Δu = d²u/dx² + d²u/dy²
    # u = tanh(6(y-0.5)) * sin(πx)
    # Let T = tanh(6(y-0.5)), S = sin(πx)
    # d²u/dx² = -π² * T * S = -π² * u
    # du/dy = 6*sech²(6(y-0.5)) * S
    # d²u/dy² = -72*tanh(6(y-0.5))*sech²(6(y-0.5)) * S
    # So Δu = -π²*T*S + (-72*T*(1-T²)*S) = (-π² - 72*(1-T²))*T*S ... wait let me be more careful
    # 
    # Actually let's just use UFL to compute everything symbolically
    
    # Compute Laplacian of u_exact using UFL
    grad_u_exact = ufl.grad(u_exact)
    laplacian_u_exact = ufl.div(grad_u_exact)
    
    # w_exact = -Δu_exact
    w_exact = -laplacian_u_exact
    
    # f = Δ²u = -Δw = Δ(-Δu)
    # So f = -Δw_exact ... but w = -Δu, so Δ²u = ΔΔu = Δ(Δu) = -Δw
    # Actually: Δ²u = f means ΔΔu = f
    # We set w = -Δu, so -Δw = ΔΔu = f
    # Hence f = -Δw_exact
    grad_w_exact = ufl.grad(w_exact)
    laplacian_w_exact = ufl.div(grad_w_exact)
    f_expr = -laplacian_w_exact
    
    # Function space
    V = fem.functionspace(domain, ("Lagrange", degree))
    
    # --- Step 1: Solve -Δw = f with w = w_exact on ∂Ω ---
    
    # Boundary conditions for w
    tdim = domain.topology.dim
    fdim = tdim - 1
    
    domain.topology.create_connectivity(fdim, tdim)
    boundary_facets = mesh.exterior_facets(domain.topology)
    
    # w BC
    w_bc_func = fem.Function(V)
    w_bc_expr = fem.Expression(w_exact, V.element.interpolation_points)
    w_bc_func.interpolate(w_bc_expr)
    
    dofs_w = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc_w = fem.dirichletbc(w_bc_func, dofs_w)
    
    # Variational form for w: -Δw = f => ∫∇w·∇v dx = ∫f·v dx
    w_trial = ufl.TrialFunction(V)
    v_test = ufl.TestFunction(V)
    
    a_w = ufl.inner(ufl.grad(w_trial), ufl.grad(v_test)) * ufl.dx
    L_w = f_expr * v_test * ufl.dx
    
    ksp_type = "cg"
    pc_type = "hypre"
    rtol = 1e-10
    
    problem_w = petsc.LinearProblem(
        a_w, L_w, bcs=[bc_w],
        petsc_options={
            "ksp_type": ksp_type,
            "pc_type": pc_type,
            "ksp_rtol": str(rtol),
            "ksp_max_it": "2000",
        },
        petsc_options_prefix="solve_w_"
    )
    wh = problem_w.solve()
    
    # Get iteration count for w solve
    iter_w = problem_w.solver.getIterationNumber()
    
    # --- Step 2: Solve -Δu = w with u = g on ∂Ω ---
    
    # u BC
    u_bc_func = fem.Function(V)
    u_bc_expr = fem.Expression(u_exact, V.element.interpolation_points)
    u_bc_func.interpolate(u_bc_expr)
    
    dofs_u = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc_u = fem.dirichletbc(u_bc_func, dofs_u)
    
    u_trial = ufl.TrialFunction(V)
    v_test2 = ufl.TestFunction(V)
    
    a_u = ufl.inner(ufl.grad(u_trial), ufl.grad(v_test2)) * ufl.dx
    L_u = wh * v_test2 * ufl.dx
    
    problem_u = petsc.LinearProblem(
        a_u, L_u, bcs=[bc_u],
        petsc_options={
            "ksp_type": ksp_type,
            "pc_type": pc_type,
            "ksp_rtol": str(rtol),
            "ksp_max_it": "2000",
        },
        petsc_options_prefix="solve_u_"
    )
    uh = problem_u.solve()
    
    iter_u = problem_u.solver.getIterationNumber()
    total_iterations = iter_w + iter_u
    
    # --- Extract solution on 50x50 uniform grid ---
    nx_out, ny_out = 50, 50
    xs = np.linspace(0.0, 1.0, nx_out)
    ys = np.linspace(0.0, 1.0, ny_out)
    XX, YY = np.meshgrid(xs, ys, indexing='ij')
    points = np.zeros((3, nx_out * ny_out))
    points[0, :] = XX.ravel()
    points[1, :] = YY.ravel()
    
    bb_tree = geometry.bb_tree(domain, tdim)
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
            "element_degree": degree,
            "ksp_type": ksp_type,
            "pc_type": pc_type,
            "rtol": rtol,
            "iterations": total_iterations,
        }
    }