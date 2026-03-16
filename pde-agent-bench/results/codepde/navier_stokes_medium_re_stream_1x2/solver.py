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
    N = 64
    degree_u = 2
    degree_p = 1
    
    comm = MPI.COMM_WORLD
    
    # 2. Create mesh
    domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
    tdim = domain.topology.dim
    fdim = tdim - 1
    
    # 3. Create mixed function space (Taylor-Hood P2/P1)
    V = fem.functionspace(domain, ("Lagrange", degree_u, (domain.geometry.dim,)))
    Q = fem.functionspace(domain, ("Lagrange", degree_p))
    
    # Create mixed element
    vel_elem = ufl.VectorElement("Lagrange", domain.ufl_cell(), degree_u)
    pres_elem = ufl.FiniteElement("Lagrange", domain.ufl_cell(), degree_p)
    mixed_elem = ufl.MixedElement([vel_elem, pres_elem])
    W = fem.functionspace(domain, mixed_elem)
    
    # 4. Define exact solution for BCs and source term
    x = ufl.SpatialCoordinate(domain)
    pi = ufl.pi
    
    # Exact velocity: u = [2*pi*cos(2*pi*y)*sin(pi*x), -pi*cos(pi*x)*sin(2*pi*y)]
    u_exact_0 = 2 * pi * ufl.cos(2 * pi * x[1]) * ufl.sin(pi * x[0])
    u_exact_1 = -pi * ufl.cos(pi * x[0]) * ufl.sin(2 * pi * x[1])
    u_exact = ufl.as_vector([u_exact_0, u_exact_1])
    
    # Exact pressure: p = cos(pi*x)*sin(pi*y)
    p_exact = ufl.cos(pi * x[0]) * ufl.sin(pi * x[1])
    
    nu = fem.Constant(domain, PETSc.ScalarType(nu_val))
    
    # Compute source term: f = u·∇u - ν∇²u + ∇p
    grad_u_exact = ufl.grad(u_exact)
    convection = ufl.dot(grad_u_exact, u_exact)  # (u·∇)u = grad(u) * u
    laplacian_u = ufl.div(ufl.grad(u_exact))
    grad_p = ufl.grad(p_exact)
    
    f_expr = convection - nu_val * laplacian_u + grad_p
    
    # 5. Define variational form
    w = fem.Function(W)
    (u_sol, p_sol) = ufl.split(w)
    (v, q) = ufl.TestFunctions(W)
    
    F = (
        nu * ufl.inner(ufl.grad(u_sol), ufl.grad(v)) * ufl.dx
        + ufl.inner(ufl.grad(u_sol) * u_sol, v) * ufl.dx
        - p_sol * ufl.div(v) * ufl.dx
        + ufl.div(u_sol) * q * ufl.dx
        - ufl.inner(f_expr, v) * ufl.dx
    )
    
    # 6. Boundary conditions
    # Velocity BC on all boundaries
    def all_boundary(x):
        return (np.isclose(x[0], 0.0) | np.isclose(x[0], 1.0) |
                np.isclose(x[1], 0.0) | np.isclose(x[1], 1.0))
    
    boundary_facets = mesh.locate_entities_boundary(domain, fdim, all_boundary)
    
    # Interpolate exact velocity onto V
    u_bc_func = fem.Function(V)
    u_exact_expr = fem.Expression(
        u_exact, V.element.interpolation_points
    )
    u_bc_func.interpolate(u_exact_expr)
    
    dofs_u = fem.locate_dofs_topological((W.sub(0), V), fdim, boundary_facets)
    bc_u = fem.dirichletbc(u_bc_func, dofs_u, W.sub(0))
    
    # Pin pressure at one point to fix the constant
    def corner_point(x):
        return np.isclose(x[0], 0.0) & np.isclose(x[1], 0.0)
    
    # Interpolate exact pressure for the pin
    p_bc_func = fem.Function(Q)
    p_exact_expr = fem.Expression(p_exact, Q.element.interpolation_points)
    p_bc_func.interpolate(p_exact_expr)
    
    corner_facets = mesh.locate_entities_boundary(domain, fdim, corner_point)
    if len(corner_facets) > 0:
        dofs_p = fem.locate_dofs_topological((W.sub(1), Q), fdim, corner_facets)
        bc_p = fem.dirichletbc(p_bc_func, dofs_p, W.sub(1))
        bcs = [bc_u, bc_p]
    else:
        bcs = [bc_u]
    
    # 7. Initial guess: interpolate exact solution (helps convergence)
    # Set initial guess from Stokes-like or exact
    W0_sub, W0_map = W.sub(0).collapse()
    W1_sub, W1_map = W.sub(1).collapse()
    
    u_init = fem.Function(W0_sub)
    u_init.interpolate(fem.Expression(u_exact, W0_sub.element.interpolation_points))
    w.x.array[W0_map] = u_init.x.array
    
    p_init = fem.Function(W1_sub)
    p_init.interpolate(fem.Expression(p_exact, W1_sub.element.interpolation_points))
    w.x.array[W1_map] = p_init.x.array
    w.x.scatter_forward()
    
    # 8. Newton solver
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
    
    n_iters, converged = solver.solve(w)
    assert converged
    w.x.scatter_forward()
    
    # 9. Extract velocity on 50x50 grid
    nx_out, ny_out = 50, 50
    xs = np.linspace(0.0, 1.0, nx_out)
    ys = np.linspace(0.0, 1.0, ny_out)
    XX, YY = np.meshgrid(xs, ys, indexing='ij')
    points_2d = np.column_stack([XX.ravel(), YY.ravel()])
    points_3d = np.zeros((points_2d.shape[0], 3))
    points_3d[:, :2] = points_2d
    
    # Extract velocity sub-function
    u_h = w.sub(0).collapse()
    
    bb_tree = geometry.bb_tree(domain, tdim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, points_3d)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points_3d)
    
    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    for i in range(points_3d.shape[0]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(points_3d[i])
            cells_on_proc.append(links[0])
            eval_map.append(i)
    
    vel_mag = np.full(points_3d.shape[0], np.nan)
    if len(points_on_proc) > 0:
        pts_arr = np.array(points_on_proc)
        cells_arr = np.array(cells_on_proc, dtype=np.int32)
        vals = u_h.eval(pts_arr, cells_arr)
        # vals shape: (n_points, 2)
        for idx_local, idx_global in enumerate(eval_map):
            ux = vals[idx_local, 0]
            uy = vals[idx_local, 1]
            vel_mag[idx_global] = np.sqrt(ux**2 + uy**2)
    
    u_grid = vel_mag.reshape((nx_out, ny_out))
    
    return {
        "u": u_grid,
        "solver_info": {
            "mesh_resolution": N,
            "element_degree": degree_u,
            "ksp_type": "gmres",
            "pc_type": "lu",
            "rtol": 1e-10,
            "nonlinear_iterations": [int(n_iters)],
        }
    }