import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, nls, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc


def solve(case_spec: dict) -> dict:
    # 1. Parse case_spec
    pde_config = case_spec.get("pde", {})
    nu_val = pde_config.get("viscosity", 0.08)
    
    # Mesh resolution - use high resolution for accuracy
    N = 80
    degree_u = 2
    degree_p = 1
    
    # 2. Create mesh
    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
    
    # 3. Mixed function spaces (Taylor-Hood P2/P1)
    V = fem.functionspace(domain, ("Lagrange", degree_u, (domain.geometry.dim,)))
    Q = fem.functionspace(domain, ("Lagrange", degree_p))
    
    # Create mixed element
    vel_el = ufl.VectorElement("Lagrange", domain.ufl_cell(), degree_u)
    pres_el = ufl.FiniteElement("Lagrange", domain.ufl_cell(), degree_p)
    mixed_el = ufl.MixedElement([vel_el, pres_el])
    W = fem.functionspace(domain, mixed_el)
    
    # 4. Define exact solutions for BCs and source term
    x = ufl.SpatialCoordinate(domain)
    pi = ufl.pi
    
    # Exact velocity: u = [pi*exp(6*(x-1))*cos(pi*y), -6*exp(6*(x-1))*sin(pi*y)]
    u_exact_0 = pi * ufl.exp(6.0 * (x[0] - 1.0)) * ufl.cos(pi * x[1])
    u_exact_1 = -6.0 * ufl.exp(6.0 * (x[0] - 1.0)) * ufl.sin(pi * x[1])
    u_exact = ufl.as_vector([u_exact_0, u_exact_1])
    
    # Exact pressure: p = sin(pi*x)*sin(pi*y)
    p_exact = ufl.sin(pi * x[0]) * ufl.sin(pi * x[1])
    
    nu = fem.Constant(domain, PETSc.ScalarType(nu_val))
    
    # Compute source term: f = u·∇u - ν ∇²u + ∇p
    grad_u_exact = ufl.grad(u_exact)
    convection = ufl.dot(grad_u_exact, u_exact)  # (u·∇)u = grad(u)*u
    laplacian_u = ufl.div(ufl.grad(u_exact))
    grad_p = ufl.grad(p_exact)
    
    f = convection - nu_val * laplacian_u + grad_p
    
    # 5. Define variational form
    w = fem.Function(W)
    (u, p) = ufl.split(w)
    (v, q) = ufl.TestFunctions(W)
    
    F = (
        nu * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
        + ufl.inner(ufl.dot(ufl.grad(u), u), v) * ufl.dx
        - p * ufl.div(v) * ufl.dx
        + ufl.div(u) * q * ufl.dx
        - ufl.inner(f, v) * ufl.dx
    )
    
    # 6. Boundary conditions
    tdim = domain.topology.dim
    fdim = tdim - 1
    
    # All boundary facets
    domain.topology.create_connectivity(fdim, tdim)
    boundary_facets = mesh.exterior_facet_indices(domain.topology)
    
    # Velocity BC
    u_bc_func = fem.Function(V)
    u_bc_expr = fem.Expression(u_exact, V.element.interpolation_points)
    u_bc_func.interpolate(u_bc_expr)
    
    dofs_u = fem.locate_dofs_topological((W.sub(0), V), fdim, boundary_facets)
    bc_u = fem.dirichletbc(u_bc_func, dofs_u, W.sub(0))
    
    # Pin pressure at one point to remove nullspace
    # Find a point, e.g., (0, 0)
    def corner(x):
        return np.isclose(x[0], 0.0) & np.isclose(x[1], 0.0)
    
    p_bc_func = fem.Function(Q)
    p_bc_expr = fem.Expression(p_exact, Q.element.interpolation_points)
    p_bc_func.interpolate(p_bc_expr)
    
    corner_facets = mesh.locate_entities_boundary(domain, fdim, corner)
    if len(corner_facets) > 0:
        dofs_p = fem.locate_dofs_topological((W.sub(1), Q), fdim, corner_facets)
        bc_p = fem.dirichletbc(p_bc_func, dofs_p, W.sub(1))
        bcs = [bc_u, bc_p]
    else:
        bcs = [bc_u]
    
    # 7. Initial guess: interpolate exact solution (helps Newton converge)
    # Use a Stokes-like initial guess or just zero
    # For better convergence, let's try to set initial guess closer to solution
    W0_sub, W0_map = W.sub(0).collapse()
    W1_sub, W1_map = W.sub(1).collapse()
    
    u_init = fem.Function(W0_sub)
    u_init_expr = fem.Expression(u_exact, W0_sub.element.interpolation_points)
    u_init.interpolate(u_init_expr)
    w.x.array[W0_map] = u_init.x.array[:]
    
    p_init = fem.Function(W1_sub)
    p_init_expr = fem.Expression(p_exact, W1_sub.element.interpolation_points)
    p_init.interpolate(p_init_expr)
    w.x.array[W1_map] = p_init.x.array[:]
    
    w.x.scatter_forward()
    
    # 8. Solve nonlinear problem
    problem = petsc.NonlinearProblem(F, w, bcs=bcs)
    solver = nls.petsc.NewtonSolver(comm, problem)
    solver.convergence_criterion = "incremental"
    solver.rtol = 1e-10
    solver.atol = 1e-12
    solver.max_it = 50
    
    ksp = solver.krylov_solver
    ksp.setType(PETSc.KSP.Type.GMRES)
    ksp.setTolerances(rtol=1e-10)
    pc = ksp.getPC()
    pc.setType(PETSc.PC.Type.LU)
    
    n_newton, converged = solver.solve(w)
    assert converged, f"Newton solver did not converge after {n_newton} iterations"
    w.x.scatter_forward()
    
    # 9. Extract velocity on 50x50 grid
    nx_out, ny_out = 50, 50
    xs = np.linspace(0.0, 1.0, nx_out)
    ys = np.linspace(0.0, 1.0, ny_out)
    XX, YY = np.meshgrid(xs, ys, indexing='ij')
    
    points_2d = np.zeros((3, nx_out * ny_out))
    points_2d[0, :] = XX.ravel()
    points_2d[1, :] = YY.ravel()
    
    # Point evaluation
    bb_tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, points_2d.T)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points_2d.T)
    
    # Extract velocity sub-function
    u_sol = w.sub(0).collapse()
    
    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    for i in range(nx_out * ny_out):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(points_2d[:, i])
            cells_on_proc.append(links[0])
            eval_map.append(i)
    
    vel_magnitude = np.full(nx_out * ny_out, np.nan)
    if len(points_on_proc) > 0:
        pts_arr = np.array(points_on_proc)
        cells_arr = np.array(cells_on_proc, dtype=np.int32)
        vals = u_sol.eval(pts_arr, cells_arr)
        # vals shape: (n_points, 2) for 2D velocity
        mag = np.sqrt(vals[:, 0]**2 + vals[:, 1]**2)
        for idx, global_idx in enumerate(eval_map):
            vel_magnitude[global_idx] = mag[idx]
    
    u_grid = vel_magnitude.reshape((nx_out, ny_out))
    
    return {
        "u": u_grid,
        "solver_info": {
            "mesh_resolution": N,
            "element_degree": degree_u,
            "ksp_type": "gmres",
            "pc_type": "lu",
            "rtol": 1e-10,
            "nonlinear_iterations": [int(n_newton)],
        }
    }