import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, nls, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc


def solve(case_spec: dict) -> dict:
    # 1. Parse case_spec
    pde = case_spec.get("pde", {})
    nu_val = pde.get("viscosity", 0.2)
    
    # Mesh resolution and element degrees
    N = 48
    degree_u = 2
    degree_p = 1
    
    # 2. Create mesh
    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
    tdim = domain.topology.dim
    fdim = tdim - 1
    
    # 3. Mixed function space (Taylor-Hood P2/P1)
    V = fem.functionspace(domain, ("Lagrange", degree_u, (domain.geometry.dim,)))
    Q = fem.functionspace(domain, ("Lagrange", degree_p))
    
    # Create mixed element
    vel_el = ufl.VectorElement("Lagrange", domain.ufl_cell(), degree_u)
    pres_el = ufl.FiniteElement("Lagrange", domain.ufl_cell(), degree_p)
    mixed_el = ufl.MixedElement([vel_el, pres_el])
    W = fem.functionspace(domain, mixed_el)
    
    # 4. Define exact solution for BCs and source term
    x = ufl.SpatialCoordinate(domain)
    pi = ufl.pi
    
    # Exact velocity
    u_exact_0 = pi * ufl.cos(pi * x[1]) * ufl.sin(2 * pi * x[0])
    u_exact_1 = -2 * pi * ufl.cos(2 * pi * x[0]) * ufl.sin(pi * x[1])
    u_exact = ufl.as_vector([u_exact_0, u_exact_1])
    
    # Exact pressure
    p_exact = ufl.sin(pi * x[0]) * ufl.cos(pi * x[1])
    
    # Compute source term: f = u·∇u - ν∇²u + ∇p
    # grad(u_exact) is a 2x2 tensor, (u·∇)u = grad(u)*u
    convection = ufl.grad(u_exact) * u_exact
    diffusion = -nu_val * ufl.div(ufl.grad(u_exact))
    pressure_grad = ufl.grad(p_exact)
    f = convection + diffusion + pressure_grad
    
    # 5. Define variational form
    w = fem.Function(W)
    (u, p) = ufl.split(w)
    (v, q) = ufl.TestFunctions(W)
    
    nu = fem.Constant(domain, PETSc.ScalarType(nu_val))
    
    F = (
        nu * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
        + ufl.inner(ufl.grad(u) * u, v) * ufl.dx
        - p * ufl.div(v) * ufl.dx
        + q * ufl.div(u) * ufl.dx
        - ufl.inner(f, v) * ufl.dx
    )
    
    # 6. Boundary conditions
    # Velocity BC on all boundaries
    def all_boundary(x):
        return (np.isclose(x[0], 0.0) | np.isclose(x[0], 1.0) |
                np.isclose(x[1], 0.0) | np.isclose(x[1], 1.0))
    
    boundary_facets = mesh.locate_entities_boundary(domain, fdim, all_boundary)
    
    # Interpolate exact velocity into V
    u_bc_func = fem.Function(V)
    u_exact_expr = fem.Expression(
        ufl.as_vector([u_exact_0, u_exact_1]),
        V.element.interpolation_points
    )
    u_bc_func.interpolate(u_exact_expr)
    
    # Locate DOFs for velocity sub-space
    W0, W0_to_W = W.sub(0).collapse()
    dofs_u = fem.locate_dofs_topological((W.sub(0), V), fdim, boundary_facets)
    bc_u = fem.dirichletbc(u_bc_func, dofs_u, W.sub(0))
    
    # Pin pressure at one point to fix the constant
    # Find a point, e.g., origin
    def origin_marker(x):
        return np.isclose(x[0], 0.0) & np.isclose(x[1], 0.0)
    
    # We'll pin pressure via a point constraint
    p_bc_val = PETSc.ScalarType(0.0)  # p_exact at (0,0) = sin(0)*cos(0) = 0
    
    # Find vertex at origin for pressure
    origin_vertices = mesh.locate_entities_boundary(domain, 0, origin_marker)
    W1, W1_to_W = W.sub(1).collapse()
    dofs_p = fem.locate_dofs_topological((W.sub(1), Q), 0, origin_vertices)
    
    p_bc_func = fem.Function(Q)
    p_bc_func.x.array[:] = 0.0
    # Set the value at the origin to p_exact(0,0) = 0
    bc_p = fem.dirichletbc(p_bc_func, dofs_p, W.sub(1))
    
    bcs = [bc_u, bc_p]
    
    # 7. Initial guess - use Stokes solution approach or just zero
    # For moderate Re, zero initial guess with Newton damping should work
    w.x.array[:] = 0.0
    
    # Better initial guess: interpolate exact solution
    # Interpolate velocity part
    w_sub0 = w.sub(0)
    w_sub0_collapsed = w_sub0.collapse()
    w_sub0_collapsed.interpolate(u_exact_expr)
    w.x.array[W0_to_W] = w_sub0_collapsed.x.array[:]
    
    # Interpolate pressure part
    p_exact_expr = fem.Expression(p_exact, Q.element.interpolation_points)
    w_sub1 = w.sub(1)
    w_sub1_collapsed = w_sub1.collapse()
    w_sub1_collapsed.interpolate(p_exact_expr)
    w.x.array[W1_to_W] = w_sub1_collapsed.x.array[:]
    
    # Reset to zero for a fair solve (or keep for faster convergence)
    # Actually, let's start from zero to be fair but use continuation if needed
    w.x.array[:] = 0.0
    
    # 8. Solve with Newton
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
    
    # Try direct solve first
    n_iters, converged = solver.solve(w)
    
    if not converged:
        # Continuation approach: start with high viscosity, decrease
        w.x.array[:] = 0.0
        nu.value = 1.0
        for nu_step in [1.0, 0.5, 0.2]:
            nu.value = nu_step
            n_iters, converged = solver.solve(w)
    
    w.x.scatter_forward()
    
    # 9. Extract velocity magnitude on 50x50 grid
    nx_out, ny_out = 50, 50
    xs = np.linspace(0.0, 1.0, nx_out)
    ys = np.linspace(0.0, 1.0, ny_out)
    X, Y = np.meshgrid(xs, ys, indexing='ij')
    points = np.zeros((3, nx_out * ny_out))
    points[0, :] = X.ravel()
    points[1, :] = Y.ravel()
    
    # Collapse velocity subspace
    u_out = w.sub(0).collapse()
    
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
    
    vel_magnitude = np.full(nx_out * ny_out, np.nan)
    
    if len(points_on_proc) > 0:
        pts_arr = np.array(points_on_proc)
        cells_arr = np.array(cells_on_proc, dtype=np.int32)
        vals = u_out.eval(pts_arr, cells_arr)
        # vals shape: (n_points, 2) for 2D velocity
        vel_mag = np.sqrt(vals[:, 0]**2 + vals[:, 1]**2)
        for idx, global_idx in enumerate(eval_map):
            vel_magnitude[global_idx] = vel_mag[idx]
    
    u_grid = vel_magnitude.reshape((nx_out, ny_out))
    
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