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
    # with u = g on ∂Ω, w = -Δu on ∂Ω (from exact solution)
    
    # Exact solution: u = exp(5*(x-1))*sin(pi*y)
    # Compute derivatives:
    # u_x = 5*exp(5*(x-1))*sin(pi*y)
    # u_xx = 25*exp(5*(x-1))*sin(pi*y)
    # u_y = pi*exp(5*(x-1))*cos(pi*y)
    # u_yy = -pi^2*exp(5*(x-1))*sin(pi*y)
    # Δu = (25 - pi^2)*exp(5*(x-1))*sin(pi*y)
    # w = -Δu = -(25 - pi^2)*exp(5*(x-1))*sin(pi*y) = (pi^2 - 25)*exp(5*(x-1))*sin(pi*y)
    # Δw = (25 - pi^2)*(25 - pi^2)*exp(5*(x-1))*sin(pi*y) = (25 - pi^2)^2 * exp(5*(x-1))*sin(pi*y)
    # Wait, let me redo:
    # w = -Δu = (pi^2 - 25)*exp(5*(x-1))*sin(pi*y)
    # w_xx = (pi^2 - 25)*25*exp(5*(x-1))*sin(pi*y)
    # w_yy = (pi^2 - 25)*(-pi^2)*exp(5*(x-1))*sin(pi*y)
    # Δw = (pi^2 - 25)*(25 - pi^2)*exp(5*(x-1))*sin(pi*y) = -(25 - pi^2)^2 * exp(5*(x-1))*sin(pi*y)
    # -Δw = (25 - pi^2)^2 * exp(5*(x-1))*sin(pi*y) = f
    # So f = (25 - pi^2)^2 * exp(5*(x-1))*sin(pi*y)
    
    V = fem.functionspace(domain, ("Lagrange", degree))
    
    x = ufl.SpatialCoordinate(domain)
    pi_val = np.pi
    
    # Exact solution in UFL
    u_exact_ufl = ufl.exp(5.0 * (x[0] - 1.0)) * ufl.sin(ufl.pi * x[1])
    
    # w_exact = -Δu = (pi^2 - 25)*exp(5*(x-1))*sin(pi*y)
    w_exact_ufl = (ufl.pi**2 - 25.0) * ufl.exp(5.0 * (x[0] - 1.0)) * ufl.sin(ufl.pi * x[1])
    
    # Source term f = Δ²u = (25 - pi^2)^2 * exp(5*(x-1))*sin(pi*y)
    f_ufl = (25.0 - ufl.pi**2)**2 * ufl.exp(5.0 * (x[0] - 1.0)) * ufl.sin(ufl.pi * x[1])
    
    # --- Step 1: Solve -Δw = f with w = w_exact on ∂Ω ---
    w_trial = ufl.TrialFunction(V)
    v_test = ufl.TestFunction(V)
    
    a1 = ufl.inner(ufl.grad(w_trial), ufl.grad(v_test)) * ufl.dx
    L1 = f_ufl * v_test * ufl.dx
    
    # BC for w
    tdim = domain.topology.dim
    fdim = tdim - 1
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
    )
    
    w_bc_func = fem.Function(V)
    w_bc_expr = fem.Expression(w_exact_ufl, V.element.interpolation_points)
    w_bc_func.interpolate(w_bc_expr)
    
    dofs_w = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc_w = fem.dirichletbc(w_bc_func, dofs_w)
    
    ksp_type = "cg"
    pc_type = "hypre"
    rtol = 1e-10
    
    problem1 = petsc.LinearProblem(
        a1, L1, bcs=[bc_w],
        petsc_options={
            "ksp_type": ksp_type,
            "pc_type": pc_type,
            "ksp_rtol": str(rtol),
            "ksp_max_it": "2000",
        },
        petsc_options_prefix="solve1_"
    )
    wh = problem1.solve()
    
    # Get iteration count for first solve
    iter1 = problem1.solver.getIterationNumber()
    
    # --- Step 2: Solve -Δu = w with u = u_exact on ∂Ω ---
    u_trial = ufl.TrialFunction(V)
    
    a2 = ufl.inner(ufl.grad(u_trial), ufl.grad(v_test)) * ufl.dx
    L2 = wh * v_test * ufl.dx
    
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
            "ksp_max_it": "2000",
        },
        petsc_options_prefix="solve2_"
    )
    uh = problem2.solve()
    
    iter2 = problem2.solver.getIterationNumber()
    
    # --- Extract solution on 50x50 grid ---
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
            "iterations": iter1 + iter2,
        }
    }