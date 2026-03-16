import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, nls, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc
import basix


def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD
    
    # Parameters
    nu_val = 0.1
    N = 32  # mesh resolution
    degree_u = 2
    degree_p = 1
    
    # Create mesh
    domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
    tdim = domain.topology.dim
    fdim = tdim - 1
    
    # Create mixed function space using basix elements
    vel_el = basix.ufl.element("Lagrange", domain.basix_cell(), degree_u, shape=(2,))
    pres_el = basix.ufl.element("Lagrange", domain.basix_cell(), degree_p)
    mixed_el = basix.ufl.mixed_element([vel_el, pres_el])
    W = fem.functionspace(domain, mixed_el)
    
    # Also create individual spaces for BC interpolation
    V = fem.functionspace(domain, ("Lagrange", degree_u, (2,)))
    Q = fem.functionspace(domain, ("Lagrange", degree_p))
    
    # Define unknown and test functions
    w = fem.Function(W)
    (u, p) = ufl.split(w)
    (v, q) = ufl.TestFunctions(W)
    
    # Spatial coordinates
    x = ufl.SpatialCoordinate(domain)
    pi = ufl.pi
    
    # Exact solution
    u_exact = ufl.as_vector([
        2*pi*ufl.cos(2*pi*x[1])*ufl.sin(3*pi*x[0]),
        -3*pi*ufl.cos(3*pi*x[0])*ufl.sin(2*pi*x[1])
    ])
    p_exact = ufl.cos(pi*x[0])*ufl.cos(2*pi*x[1])
    
    # Compute source term from manufactured solution
    # f = u·∇u - ν ∇²u + ∇p
    grad_u_exact = ufl.grad(u_exact)
    convection = ufl.dot(grad_u_exact, u_exact)  # (u·∇)u = grad(u) * u
    laplacian_u = ufl.div(ufl.grad(u_exact))
    grad_p_exact = ufl.grad(p_exact)
    
    f = convection - nu_val * laplacian_u + grad_p_exact
    
    # Residual form
    nu_c = fem.Constant(domain, PETSc.ScalarType(nu_val))
    
    F = (
        nu_c * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
        + ufl.inner(ufl.grad(u) * u, v) * ufl.dx
        - p * ufl.div(v) * ufl.dx
        + q * ufl.div(u) * ufl.dx
        - ufl.inner(f, v) * ufl.dx
    )
    
    # Boundary conditions - all boundaries
    def all_boundary(x):
        return (np.isclose(x[0], 0.0) | np.isclose(x[0], 1.0) |
                np.isclose(x[1], 0.0) | np.isclose(x[1], 1.0))
    
    boundary_facets = mesh.locate_entities_boundary(domain, fdim, all_boundary)
    
    # Interpolate exact velocity BC
    u_bc_func = fem.Function(V)
    u_bc_expr = fem.Expression(u_exact, V.element.interpolation_points)
    u_bc_func.interpolate(u_bc_expr)
    
    # Locate DOFs for velocity sub-space
    W0 = W.sub(0)
    dofs_u = fem.locate_dofs_topological((W0, V), fdim, boundary_facets)
    bc_u = fem.dirichletbc(u_bc_func, dofs_u, W0)
    
    # Pin pressure at one point to fix the constant
    # Find a vertex at (0,0)
    def corner_point(x):
        return np.isclose(x[0], 0.0) & np.isclose(x[1], 0.0)
    
    # Interpolate exact pressure for the pin
    p_bc_func = fem.Function(Q)
    p_bc_expr = fem.Expression(p_exact, Q.element.interpolation_points)
    p_bc_func.interpolate(p_bc_expr)
    
    corner_vertices = mesh.locate_entities_boundary(domain, 0, corner_point)
    W1 = W.sub(1)
    dofs_p = fem.locate_dofs_topological((W1, Q), 0, corner_vertices)
    bc_p = fem.dirichletbc(p_bc_func, dofs_p, W1)
    
    bcs = [bc_u, bc_p]
    
    # Set initial guess using exact solution (helps convergence)
    # Interpolate velocity into W.sub(0)
    w.sub(0).interpolate(u_bc_func)
    w.sub(1).interpolate(p_bc_func)
    w.x.scatter_forward()
    
    # Solve nonlinear problem
    problem = petsc.NonlinearProblem(F, w, bcs=bcs)
    solver = nls.petsc.NewtonSolver(comm, problem)
    solver.convergence_criterion = "incremental"
    solver.rtol = 1e-10
    solver.atol = 1e-12
    solver.max_it = 25
    
    ksp = solver.krylov_solver
    ksp.setType(PETSc.KSP.Type.GMRES)
    pc = ksp.getPC()
    pc.setType(PETSc.PC.Type.LU)
    
    n_newton, converged = solver.solve(w)
    assert converged, f"Newton solver did not converge after {n_newton} iterations"
    w.x.scatter_forward()
    
    # Extract velocity and pressure
    u_sol = w.sub(0).collapse()
    p_sol = w.sub(1).collapse()
    
    # Evaluate on 50x50 grid
    nx_out, ny_out = 50, 50
    xs = np.linspace(0, 1, nx_out)
    ys = np.linspace(0, 1, ny_out)
    XX, YY = np.meshgrid(xs, ys, indexing='ij')
    points = np.zeros((3, nx_out * ny_out))
    points[0, :] = XX.ravel()
    points[1, :] = YY.ravel()
    
    # Point evaluation
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
    
    u_values = np.full((points.shape[1], 2), np.nan)
    if len(points_on_proc) > 0:
        pts_arr = np.array(points_on_proc)
        cells_arr = np.array(cells_on_proc, dtype=np.int32)
        vals = u_sol.eval(pts_arr, cells_arr)
        for idx, global_idx in enumerate(eval_map):
            u_values[global_idx, :] = vals[idx, :]
    
    # Compute velocity magnitude
    vel_mag = np.sqrt(u_values[:, 0]**2 + u_values[:, 1]**2)
    u_grid = vel_mag.reshape((nx_out, ny_out))
    
    solver_info = {
        "mesh_resolution": N,
        "element_degree": degree_u,
        "ksp_type": "gmres",
        "pc_type": "lu",
        "rtol": 1e-10,
        "nonlinear_iterations": [int(n_newton)],
    }
    
    return {
        "u": u_grid,
        "solver_info": solver_info,
    }