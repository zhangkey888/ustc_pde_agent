import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc

ScalarType = PETSc.ScalarType


def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD
    
    # Parameters
    N = 64
    degree = 2
    ksp_type = "cg"
    pc_type = "hypre"
    rtol = 1e-10
    
    # Create mesh
    domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
    
    # Function space
    V = fem.functionspace(domain, ("Lagrange", degree))
    
    # Manufactured solution: u = exp(x)*sin(pi*y)
    # Laplacian: Δu = exp(x)*sin(pi*y) - pi^2*exp(x)*sin(pi*y) = exp(x)*sin(pi*y)*(1 - pi^2)
    # Bilaplacian: Δ²u = Δ(Δu) = (1-pi^2) * Δ(exp(x)*sin(pi*y)) = (1-pi^2)*(1-pi^2)*exp(x)*sin(pi*y)
    # = (1-pi^2)^2 * exp(x)*sin(pi*y)
    
    x = ufl.SpatialCoordinate(domain)
    pi_val = np.pi
    
    u_exact_ufl = ufl.exp(x[0]) * ufl.sin(ufl.pi * x[1])
    
    # f = Δ²u = (1 - pi^2)^2 * exp(x)*sin(pi*y)
    f_ufl = (1.0 - ufl.pi**2)**2 * ufl.exp(x[0]) * ufl.sin(ufl.pi * x[1])
    
    # Mixed formulation: introduce w = -Δu
    # -Δu = w  =>  (grad u, grad v) = (w, v) for all v  (with u = g on ∂Ω)
    # -Δw = f  =>  (grad w, grad v) = (f, v) for all v  (with w = -Δu on ∂Ω)
    
    # w_exact = -Δu = -(1 - pi^2)*exp(x)*sin(pi*y) = (pi^2 - 1)*exp(x)*sin(pi*y)
    w_exact_ufl = (ufl.pi**2 - 1.0) * ufl.exp(x[0]) * ufl.sin(ufl.pi * x[1])
    
    # Boundary conditions
    tdim = domain.topology.dim
    fdim = tdim - 1
    
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
    )
    
    # Step 1: Solve -Δw = f with w = w_exact on ∂Ω
    w_bc_func = fem.Function(V)
    w_bc_expr = fem.Expression(w_exact_ufl, V.element.interpolation_points)
    w_bc_func.interpolate(w_bc_expr)
    
    dofs_w = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc_w = fem.dirichletbc(w_bc_func, dofs_w)
    
    w_trial = ufl.TrialFunction(V)
    v_test = ufl.TestFunction(V)
    
    a_w = ufl.inner(ufl.grad(w_trial), ufl.grad(v_test)) * ufl.dx
    L_w = ufl.inner(f_ufl, v_test) * ufl.dx
    
    total_iterations = 0
    
    problem_w = petsc.LinearProblem(
        a_w, L_w, bcs=[bc_w],
        petsc_options={
            "ksp_type": ksp_type,
            "pc_type": pc_type,
            "ksp_rtol": str(rtol),
            "ksp_max_it": "1000",
        },
        petsc_options_prefix="step1_"
    )
    w_sol = problem_w.solve()
    
    # Get iteration count for step 1
    ksp1 = problem_w.solver
    total_iterations += ksp1.getIterationNumber()
    
    # Step 2: Solve -Δu = w with u = u_exact on ∂Ω
    u_bc_func = fem.Function(V)
    u_bc_expr = fem.Expression(u_exact_ufl, V.element.interpolation_points)
    u_bc_func.interpolate(u_bc_expr)
    
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
        petsc_options_prefix="step2_"
    )
    u_sol = problem_u.solve()
    
    ksp2 = problem_u.solver
    total_iterations += ksp2.getIterationNumber()
    
    # Evaluate on 50x50 grid
    nx_out, ny_out = 50, 50
    xs = np.linspace(0, 1, nx_out)
    ys = np.linspace(0, 1, ny_out)
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
        vals = u_sol.eval(pts_arr, cells_arr)
        u_values[eval_map] = vals.flatten()
    
    u_grid = u_values.reshape((nx_out, ny_out))
    
    solver_info = {
        "mesh_resolution": N,
        "element_degree": degree,
        "ksp_type": ksp_type,
        "pc_type": pc_type,
        "rtol": rtol,
        "iterations": total_iterations,
    }
    
    return {
        "u": u_grid,
        "solver_info": solver_info,
    }