import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, nls, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc


def solve(case_spec: dict) -> dict:
    # 1. Parse case_spec
    pde_config = case_spec.get("pde", case_spec.get("oracle_config", {}).get("pde", {}))
    nu_val = pde_config.get("viscosity", 0.15)
    
    # Mesh resolution and element degrees
    N = 40  # mesh resolution
    degree_u = 2
    degree_p = 1
    
    comm = MPI.COMM_WORLD
    
    # 2. Create mesh
    domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
    tdim = domain.topology.dim
    fdim = tdim - 1
    
    # 3. Function spaces (Taylor-Hood P2/P1)
    V = fem.functionspace(domain, ("Lagrange", degree_u, (domain.geometry.dim,)))
    Q = fem.functionspace(domain, ("Lagrange", degree_p))
    
    # Mixed function space
    from dolfinx.fem import Function
    from basix.ufl import element, mixed_element
    
    vel_el = element("Lagrange", domain.topology.cell_name(), degree_u, shape=(domain.geometry.dim,))
    pres_el = element("Lagrange", domain.topology.cell_name(), degree_p)
    mixed_el = mixed_element([vel_el, pres_el])
    
    W = fem.functionspace(domain, mixed_el)
    
    # 4. Define manufactured solution symbolically
    x = ufl.SpatialCoordinate(domain)
    pi = ufl.pi
    
    # Exact velocity: u = [pi*exp(2*x)*cos(pi*y), -2*exp(2*x)*sin(pi*y)]
    u_exact_0 = pi * ufl.exp(2 * x[0]) * ufl.cos(pi * x[1])
    u_exact_1 = -2.0 * ufl.exp(2 * x[0]) * ufl.sin(pi * x[1])
    u_exact = ufl.as_vector([u_exact_0, u_exact_1])
    
    # Exact pressure: p = exp(x)*cos(pi*y)
    p_exact = ufl.exp(x[0]) * ufl.cos(pi * x[1])
    
    nu = fem.Constant(domain, PETSc.ScalarType(nu_val))
    
    # Compute source term from manufactured solution
    # f = u·∇u - ν ∇²u + ∇p
    # Note: -ν ∇²u = -ν div(grad(u))
    # So f = (u·∇)u - ν Δu + ∇p
    # In weak form: (u·∇u, v) + ν(∇u, ∇v) - (p, ∇·v) = (f, v)
    # f = (u_exact·∇)u_exact - ν Δ(u_exact) + ∇(p_exact)
    
    grad_u_exact = ufl.grad(u_exact)
    convection_exact = ufl.dot(grad_u_exact, u_exact)  # (u·∇)u = grad(u)*u
    laplacian_u_exact = ufl.div(ufl.grad(u_exact))
    grad_p_exact = ufl.grad(p_exact)
    
    f = convection_exact - nu_val * laplacian_u_exact + grad_p_exact
    
    # 5. Define variational problem
    w = fem.Function(W)
    (u_test, p_test) = ufl.TestFunctions(W)
    (u_sol, p_sol) = ufl.split(w)
    
    # Residual: ν(∇u, ∇v) + ((u·∇)u, v) - (p, ∇·v) + (∇·u, q) - (f, v) = 0
    F = (
        nu * ufl.inner(ufl.grad(u_sol), ufl.grad(u_test)) * ufl.dx
        + ufl.inner(ufl.grad(u_sol) * u_sol, u_test) * ufl.dx
        - p_sol * ufl.div(u_test) * ufl.dx
        + ufl.div(u_sol) * p_test * ufl.dx
        - ufl.inner(f, u_test) * ufl.dx
    )
    
    # 6. Boundary conditions
    # Velocity BC on entire boundary
    u_bc_func = fem.Function(V)
    
    u_exact_expr = fem.Expression(
        u_exact, V.element.interpolation_points
    )
    u_bc_func.interpolate(u_exact_expr)
    
    # Find all boundary facets
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
    )
    
    # Locate DOFs for velocity sub-space
    W_sub0, sub0_map = W.sub(0).collapse()
    dofs_u = fem.locate_dofs_topological((W.sub(0), V), fdim, boundary_facets)
    bc_u = fem.dirichletbc(u_bc_func, dofs_u, W.sub(0))
    
    bcs = [bc_u]
    
    # 7. Set initial guess using Stokes-like approach (interpolate exact solution as guess)
    # Interpolate exact velocity into W.sub(0)
    w_sub0_func = fem.Function(W_sub0)
    w_sub0_func.interpolate(u_exact_expr)
    w.x.array[sub0_map] = w_sub0_func.x.array
    
    # Interpolate exact pressure into W.sub(1)
    W_sub1, sub1_map = W.sub(1).collapse()
    p_bc_func = fem.Function(W_sub1)
    p_exact_expr = fem.Expression(p_exact, W_sub1.element.interpolation_points)
    p_bc_func.interpolate(p_exact_expr)
    w.x.array[sub1_map] = p_bc_func.x.array
    
    # Actually, let's NOT use exact solution as initial guess (that's cheating for convergence test)
    # Instead use zero or a simple guess. But for manufactured solution verification, 
    # a nearby guess is fine. Let's use a perturbation.
    # Actually for correctness, let's just use zero and let Newton converge.
    w.x.array[:] = 0.0
    # Re-apply BC values
    # Better: interpolate exact on boundary only via BC application
    
    # 8. Solve nonlinear problem
    problem = petsc.NonlinearProblem(F, w, bcs=bcs)
    solver = nls.petsc.NewtonSolver(comm, problem)
    
    solver.convergence_criterion = "incremental"
    solver.rtol = 1e-10
    solver.atol = 1e-12
    solver.max_it = 50
    solver.relaxation_parameter = 1.0
    
    ksp = solver.krylov_solver
    ksp.setType(PETSc.KSP.Type.GMRES)
    ksp.setTolerances(rtol=1e-10, atol=1e-12, max_it=2000)
    pc = ksp.getPC()
    pc.setType(PETSc.PC.Type.LU)
    
    n_newton, converged = solver.solve(w)
    assert converged, f"Newton solver did not converge after {n_newton} iterations"
    w.x.scatter_forward()
    
    # 9. Extract velocity on 50x50 grid
    nx_out, ny_out = 50, 50
    xs = np.linspace(0.0, 1.0, nx_out)
    ys = np.linspace(0.0, 1.0, ny_out)
    X, Y = np.meshgrid(xs, ys, indexing='ij')
    
    points = np.zeros((3, nx_out * ny_out))
    points[0, :] = X.ravel()
    points[1, :] = Y.ravel()
    
    # Extract velocity sub-function
    u_h = w.sub(0).collapse()
    
    # Point evaluation
    bb_tree = geometry.bb_tree(domain, tdim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, points.T)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points.T)
    
    n_points = points.shape[1]
    velocity_mag = np.full(n_points, np.nan)
    
    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    
    for i in range(n_points):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(points[:, i])
            cells_on_proc.append(links[0])
            eval_map.append(i)
    
    if len(points_on_proc) > 0:
        pts_arr = np.array(points_on_proc)
        cells_arr = np.array(cells_on_proc, dtype=np.int32)
        vals = u_h.eval(pts_arr, cells_arr)
        # vals shape: (n_eval_points, 2) for 2D velocity
        for idx, global_idx in enumerate(eval_map):
            ux = vals[idx, 0]
            uy = vals[idx, 1]
            velocity_mag[global_idx] = np.sqrt(ux**2 + uy**2)
    
    u_grid = velocity_mag.reshape((nx_out, ny_out))
    
    # Get total linear iterations
    total_linear_its = 0  # LU solver does 1 iteration per Newton step
    total_linear_its = n_newton  # Each Newton step = 1 LU solve
    
    return {
        "u": u_grid,
        "solver_info": {
            "mesh_resolution": N,
            "element_degree": degree_u,
            "ksp_type": "gmres",
            "pc_type": "lu",
            "rtol": 1e-10,
            "iterations": int(total_linear_its),
            "nonlinear_iterations": [int(n_newton)],
        }
    }