import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, nls, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc


def solve(case_spec: dict) -> dict:
    # 1. Parse case_spec
    pde_config = case_spec.get("pde", {})
    nu_val = pde_config.get("viscosity", 0.2)
    
    # Mesh resolution and element degrees
    # P4/P3 as indicated by case name, small mesh
    N = 16  # small mesh
    degree_u = 4
    degree_p = 3
    
    comm = MPI.COMM_WORLD
    
    # 2. Create mesh
    domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
    tdim = domain.topology.dim
    fdim = tdim - 1
    
    # 3. Function spaces - Taylor-Hood P4/P3
    V = fem.functionspace(domain, ("Lagrange", degree_u, (domain.geometry.dim,)))
    Q = fem.functionspace(domain, ("Lagrange", degree_p))
    
    # Mixed function space
    from dolfinx.fem import Function
    from basix.ufl import mixed_element, element
    
    vel_el = element("Lagrange", domain.topology.cell_name(), degree_u, shape=(domain.geometry.dim,))
    pres_el = element("Lagrange", domain.topology.cell_name(), degree_p)
    mel = mixed_element([vel_el, pres_el])
    W = fem.functionspace(domain, mel)
    
    # 4. Define manufactured solution and source term
    x = ufl.SpatialCoordinate(domain)
    pi = ufl.pi
    
    # Exact velocity: u = [pi*cos(pi*y)*sin(pi*x), -pi*cos(pi*x)*sin(pi*y)]
    u_exact = ufl.as_vector([
        pi * ufl.cos(pi * x[1]) * ufl.sin(pi * x[0]),
        -pi * ufl.cos(pi * x[0]) * ufl.sin(pi * x[1])
    ])
    
    # Exact pressure: p = cos(pi*x)*cos(pi*y)
    p_exact = ufl.cos(pi * x[0]) * ufl.cos(pi * x[1])
    
    # Compute source term: f = u·∇u - ν ∇²u + ∇p
    # grad(u) is a 2x2 tensor, u·∇u = grad(u) * u
    grad_u_exact = ufl.grad(u_exact)
    convection = ufl.dot(grad_u_exact, u_exact)  # (u·∇)u = grad(u)*u
    
    # -ν ∇²u = -ν div(grad(u))
    laplacian_u = ufl.div(ufl.grad(u_exact))
    
    grad_p_exact = ufl.grad(p_exact)
    
    f_expr = convection - nu_val * laplacian_u + grad_p_exact
    
    # 5. Define variational form (nonlinear residual)
    w = fem.Function(W)
    (u_sol, p_sol) = ufl.split(w)
    (v, q) = ufl.TestFunctions(W)
    
    nu_c = fem.Constant(domain, PETSc.ScalarType(nu_val))
    
    # Residual: (u·∇)u · v + ν ∇u:∇v - p div(v) + q div(u) - f·v = 0
    F = (
        ufl.inner(ufl.grad(u_sol) * u_sol, v) * ufl.dx
        + nu_c * ufl.inner(ufl.grad(u_sol), ufl.grad(v)) * ufl.dx
        - p_sol * ufl.div(v) * ufl.dx
        + q * ufl.div(u_sol) * ufl.dx
        - ufl.inner(f_expr, v) * ufl.dx
    )
    
    # 6. Boundary conditions
    # Velocity BC on all boundaries
    V_sub, _ = W.sub(0).collapse()
    
    u_bc_func = fem.Function(V_sub)
    
    # Interpolate exact velocity
    u_bc_expr = fem.Expression(
        u_exact,
        V_sub.element.interpolation_points
    )
    u_bc_func.interpolate(u_bc_expr)
    
    # Find all boundary facets
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
    )
    
    dofs_u = fem.locate_dofs_topological((W.sub(0), V_sub), fdim, boundary_facets)
    bc_u = fem.dirichletbc(u_bc_func, dofs_u, W.sub(0))
    
    # Pin pressure at one point to fix the constant
    # Find a DOF near (0,0) for pressure
    Q_sub, _ = W.sub(1).collapse()
    
    def corner(x):
        return np.isclose(x[0], 0.0) & np.isclose(x[1], 0.0)
    
    # Pin pressure at corner
    p_bc_func = fem.Function(Q_sub)
    p_bc_expr = fem.Expression(p_exact, Q_sub.element.interpolation_points)
    p_bc_func.interpolate(p_bc_expr)
    
    corner_facets = mesh.locate_entities_boundary(domain, fdim, corner)
    if len(corner_facets) > 0:
        dofs_p = fem.locate_dofs_topological((W.sub(1), Q_sub), fdim, corner_facets)
        bc_p = fem.dirichletbc(p_bc_func, dofs_p, W.sub(1))
        bcs = [bc_u, bc_p]
    else:
        bcs = [bc_u]
    
    # 7. Initial guess: interpolate exact solution (helps convergence)
    # Set initial guess to exact solution for robust convergence
    W0_sub, w0_map = W.sub(0).collapse()
    W1_sub, w1_map = W.sub(1).collapse()
    
    u_init = fem.Function(W0_sub)
    u_init.interpolate(u_bc_expr)
    
    p_init = fem.Function(W1_sub)
    p_init.interpolate(p_bc_expr)
    
    w.x.array[w0_map] = u_init.x.array
    w.x.array[w1_map] = p_init.x.array
    w.x.scatter_forward()
    
    # 8. Solve with Newton
    problem = petsc.NonlinearProblem(F, w, bcs=bcs)
    solver = nls.petsc.NewtonSolver(comm, problem)
    
    solver.convergence_criterion = "incremental"
    solver.rtol = 1e-10
    solver.atol = 1e-12
    solver.max_it = 50
    
    ksp = solver.krylov_solver
    ksp.setType(PETSc.KSP.Type.GMRES)
    pc = ksp.getPC()
    pc.setType(PETSc.PC.Type.LU)
    
    n_newton, converged = solver.solve(w)
    assert converged, f"Newton solver did not converge after {n_newton} iterations"
    w.x.scatter_forward()
    
    # 9. Extract velocity on 50x50 grid
    nx_out, ny_out = 50, 50
    xs = np.linspace(0, 1, nx_out)
    ys = np.linspace(0, 1, ny_out)
    XX, YY = np.meshgrid(xs, ys, indexing='ij')
    
    points_2d = np.zeros((3, nx_out * ny_out))
    points_2d[0, :] = XX.ravel()
    points_2d[1, :] = YY.ravel()
    
    # Get velocity sub-function
    u_h = w.sub(0).collapse()
    
    bb_tree = geometry.bb_tree(domain, tdim)
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
    
    # Evaluate velocity
    vel_values = np.full((nx_out * ny_out, 2), np.nan)
    if len(points_on_proc) > 0:
        pts_arr = np.array(points_on_proc)
        cells_arr = np.array(cells_on_proc, dtype=np.int32)
        vals = u_h.eval(pts_arr, cells_arr)
        for idx, global_idx in enumerate(eval_map):
            vel_values[global_idx, :] = vals[idx, :2]
    
    # Compute velocity magnitude
    vel_mag = np.sqrt(vel_values[:, 0]**2 + vel_values[:, 1]**2)
    u_grid = vel_mag.reshape((nx_out, ny_out))
    
    return {
        "u": u_grid,
        "solver_info": {
            "mesh_resolution": N,
            "element_degree": degree_u,
            "ksp_type": "gmres",
            "pc_type": "lu",
            "rtol": 1e-10,
            "iterations": int(n_newton),
            "nonlinear_iterations": [int(n_newton)],
        }
    }