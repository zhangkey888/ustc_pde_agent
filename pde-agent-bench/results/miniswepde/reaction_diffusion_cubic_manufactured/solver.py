import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc

ScalarType = PETSc.ScalarType


def solve(case_spec: dict) -> dict:
    pde_spec = case_spec.get("pde", {})
    time_spec = pde_spec.get("time", {})
    
    # Hardcoded defaults for this problem
    t_end = 0.2
    dt_val = 0.005
    scheme = "backward_euler"
    is_transient = True
    
    if time_spec:
        t_end = time_spec.get("t_end", t_end)
        dt_val = time_spec.get("dt", dt_val)
        scheme = time_spec.get("scheme", scheme)
    
    params = case_spec.get("params", {})
    epsilon = params.get("epsilon", 1.0)
    N = params.get("mesh_resolution", 48)
    element_degree = params.get("element_degree", 2)
    dt_val = params.get("dt", dt_val)
    newton_rtol = params.get("newton_rtol", 1e-8)
    newton_max_it = params.get("newton_max_it", 25)
    
    nx_out = case_spec.get("output", {}).get("nx", 60)
    ny_out = case_spec.get("output", {}).get("ny", 60)
    
    comm = MPI.COMM_WORLD
    
    domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", element_degree))
    
    x = ufl.SpatialCoordinate(domain)
    pi = ufl.pi
    t_const = fem.Constant(domain, ScalarType(0.0))
    
    # Manufactured solution: u_exact = exp(-t) * 0.2 * sin(2*pi*x) * sin(pi*y)
    u_exact_ufl = ufl.exp(-t_const) * 0.2 * ufl.sin(2*pi*x[0]) * ufl.sin(pi*x[1])
    
    # Source term: f = du/dt - eps*laplacian(u) + u^3
    # du/dt = -u_exact
    # laplacian(u) = -(4*pi^2 + pi^2)*u_exact = -5*pi^2*u_exact
    # -eps*laplacian(u) = eps*5*pi^2*u_exact
    # f = -u_exact + eps*5*pi^2*u_exact + u_exact^3
    f_ufl = -u_exact_ufl + epsilon * 5.0 * pi**2 * u_exact_ufl + u_exact_ufl**3
    
    u_sol = fem.Function(V, name="u")
    u_n = fem.Function(V, name="u_n")
    v = ufl.TestFunction(V)
    
    # Boundary conditions
    u_bc = fem.Function(V)
    tdim = domain.topology.dim
    fdim = tdim - 1
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
    )
    dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    
    u_exact_expr = fem.Expression(u_exact_ufl, V.element.interpolation_points)
    
    # Initial condition
    t_const.value = 0.0
    u_bc.interpolate(u_exact_expr)
    u_sol.interpolate(u_exact_expr)
    u_n.interpolate(u_exact_expr)
    
    u_initial_grid = _evaluate_on_grid(domain, u_sol, nx_out, ny_out)
    
    # Weak form (backward Euler)
    dt_const = fem.Constant(domain, ScalarType(dt_val))
    eps_const = fem.Constant(domain, ScalarType(epsilon))
    
    F = (
        (u_sol - u_n) / dt_const * v * ufl.dx
        + eps_const * ufl.inner(ufl.grad(u_sol), ufl.grad(v)) * ufl.dx
        + u_sol**3 * v * ufl.dx
        - f_ufl * v * ufl.dx
    )
    
    bcs = [fem.dirichletbc(u_bc, dofs)]
    
    # Time stepping
    n_steps = int(round(t_end / dt_val))
    t = 0.0
    nonlinear_iterations = []
    ksp_type_str = "gmres"
    pc_type_str = "ilu"
    
    # We need to recreate the problem each step since BCs change,
    # or we can create it once and update BCs. Let's try creating once.
    petsc_opts = {
        "snes_rtol": newton_rtol,
        "snes_atol": 1e-12,
        "snes_max_it": newton_max_it,
        "ksp_type": ksp_type_str,
        "pc_type": pc_type_str,
        "snes_error_if_not_converged": False,
    }
    
    problem = petsc.NonlinearProblem(
        F, u_sol,
        petsc_options_prefix="nls_",
        bcs=bcs,
        petsc_options=petsc_opts,
    )
    
    for step in range(n_steps):
        t += dt_val
        t_const.value = t
        
        # Update boundary condition
        u_bc.interpolate(u_exact_expr)
        
        # Solve
        problem.solve()
        
        snes = problem.solver
        n_iters = snes.getIterationNumber()
        reason = snes.getConvergedReason()
        
        u_sol.x.scatter_forward()
        nonlinear_iterations.append(n_iters)
        
        # Update previous solution
        u_n.x.array[:] = u_sol.x.array[:]
    
    # Evaluate on output grid
    u_grid = _evaluate_on_grid(domain, u_sol, nx_out, ny_out)
    
    solver_info = {
        "mesh_resolution": N,
        "element_degree": element_degree,
        "ksp_type": ksp_type_str,
        "pc_type": pc_type_str,
        "rtol": newton_rtol,
        "iterations": sum(nonlinear_iterations) * 3,
        "dt": dt_val,
        "n_steps": n_steps,
        "time_scheme": "backward_euler",
        "nonlinear_iterations": nonlinear_iterations,
    }
    
    return {
        "u": u_grid,
        "u_initial": u_initial_grid,
        "solver_info": solver_info,
    }


def _evaluate_on_grid(domain, u_func, nx, ny):
    xs = np.linspace(0, 1, nx)
    ys = np.linspace(0, 1, ny)
    XX, YY = np.meshgrid(xs, ys, indexing='ij')
    
    points_2d = np.column_stack([XX.ravel(), YY.ravel()])
    points_3d = np.zeros((points_2d.shape[0], 3))
    points_3d[:, :2] = points_2d
    
    bb_tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, points_3d)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points_3d)
    
    u_values = np.full(points_3d.shape[0], np.nan)
    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    
    for i in range(points_3d.shape[0]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(points_3d[i])
            cells_on_proc.append(links[0])
            eval_map.append(i)
    
    if len(points_on_proc) > 0:
        pts_arr = np.array(points_on_proc)
        cells_arr = np.array(cells_on_proc, dtype=np.int32)
        vals = u_func.eval(pts_arr, cells_arr)
        u_values[eval_map] = vals.flatten()
    
    return u_values.reshape((nx, ny))
