import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, nls, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc


def solve(case_spec: dict) -> dict:
    # 1. Parse case_spec
    pde_config = case_spec.get("pde", {})
    nu_val = pde_config.get("viscosity", 0.25)
    
    # Mesh resolution and element degrees
    N = 40
    degree_u = 2
    degree_p = 1
    
    comm = MPI.COMM_WORLD
    
    # 2. Create mesh
    domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
    tdim = domain.topology.dim
    fdim = tdim - 1
    
    # 3. Mixed function spaces (Taylor-Hood P2/P1)
    V = fem.functionspace(domain, ("Lagrange", degree_u, (domain.geometry.dim,)))
    Q = fem.functionspace(domain, ("Lagrange", degree_p))
    
    # Create mixed element
    vel_elem = ufl.VectorElement("Lagrange", domain.ufl_cell(), degree_u)
    pres_elem = ufl.FiniteElement("Lagrange", domain.ufl_cell(), degree_p)
    mixed_elem = ufl.MixedElement([vel_elem, pres_elem])
    W = fem.functionspace(domain, mixed_elem)
    
    # 4. Define exact solution for BCs and source term
    x = ufl.SpatialCoordinate(domain)
    
    # Exact velocity: u = [x*(1-x)*(1-2*y), -y*(1-y)*(1-2*x)]
    u_exact_0 = x[0] * (1.0 - x[0]) * (1.0 - 2.0 * x[1])
    u_exact_1 = -x[1] * (1.0 - x[1]) * (1.0 - 2.0 * x[0])
    u_exact = ufl.as_vector([u_exact_0, u_exact_1])
    
    # Exact pressure: p = x - y
    p_exact = x[0] - x[1]
    
    # Compute source term: f = u·∇u - ν∇²u + ∇p
    grad_u_exact = ufl.grad(u_exact)
    convection = ufl.dot(grad_u_exact, u_exact)  # (grad u) * u = u·∇u in index notation
    laplacian_u = ufl.div(ufl.grad(u_exact))
    grad_p = ufl.grad(p_exact)
    
    nu = fem.Constant(domain, PETSc.ScalarType(nu_val))
    f = convection - nu * laplacian_u + grad_p
    
    # 5. Define variational form (nonlinear residual)
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
    # All boundary: u = u_exact (which is zero on boundary for this manufactured solution)
    # Let's verify: on x=0: u0 = 0*(1-0)*(1-2y) = 0, u1 = -y(1-y)*(1-0) = -y(1-y)*(1) -- wait, not zero
    # Actually u_exact on x=0: u0=0, u1=-y(1-y)*1 which is NOT zero
    # On x=1: u0=0, u1=-y(1-y)*(-1) = y(1-y)
    # On y=0: u0=x(1-x)*1=x(1-x), u1=0
    # On y=1: u0=x(1-x)*(-1)=-x(1-x), u1=0
    # So the BC is NOT zero everywhere. We need to apply the exact solution as BC.
    
    # Create BC function for velocity
    u_bc_func = fem.Function(V)
    u_bc_expr = fem.Expression(u_exact, V.element.interpolation_points)
    u_bc_func.interpolate(u_bc_expr)
    
    # Locate all boundary facets
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
    )
    
    # Locate DOFs for velocity on boundary
    dofs_u = fem.locate_dofs_topological((W.sub(0), V), fdim, boundary_facets)
    bc_u = fem.dirichletbc(u_bc_func, dofs_u, W.sub(0))
    
    # Pin pressure at one point to fix the constant
    # Find a DOF near (0,0) for pressure
    def corner_marker(x):
        return np.isclose(x[0], 0.0) & np.isclose(x[1], 0.0)
    
    # We'll pin pressure at corner (0,0) where p_exact = 0 - 0 = 0
    corner_facets = mesh.locate_entities_boundary(domain, fdim, corner_marker)
    
    # For pressure pinning, use a point approach
    p_bc_func = fem.Function(Q)
    # p_exact at (0,0) = 0
    p_bc_func.x.array[:] = 0.0
    
    # Actually, let's use a cleaner approach - pin pressure DOF at corner
    dofs_p_corner = fem.locate_dofs_topological((W.sub(1), Q), fdim, corner_facets)
    
    # Create a function with the exact pressure value at that point
    p_bc_val = fem.Function(Q)
    p_bc_expr = fem.Expression(p_exact, Q.element.interpolation_points)
    p_bc_val.interpolate(p_bc_expr)
    
    bc_p = fem.dirichletbc(p_bc_val, dofs_p_corner, W.sub(1))
    
    bcs = [bc_u, bc_p]
    
    # 7. Initial guess: interpolate exact solution (helps Newton converge fast)
    # Set initial guess to zero or close to exact
    w.x.array[:] = 0.0
    
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
    xs = np.linspace(0.0, 1.0, nx_out)
    ys = np.linspace(0.0, 1.0, ny_out)
    XX, YY = np.meshgrid(xs, ys, indexing='ij')
    
    points_2d = np.zeros((3, nx_out * ny_out))
    points_2d[0, :] = XX.ravel()
    points_2d[1, :] = YY.ravel()
    
    # Get velocity sub-function
    u_sol = w.sub(0).collapse()
    
    # Point evaluation
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
    
    # velocity_magnitude on grid
    vel_mag = np.full(nx_out * ny_out, np.nan)
    
    if len(points_on_proc) > 0:
        pts_arr = np.array(points_on_proc)
        cells_arr = np.array(cells_on_proc, dtype=np.int32)
        vals = u_sol.eval(pts_arr, cells_arr)
        # vals shape: (n_points, 2) for 2D velocity
        mag = np.sqrt(vals[:, 0]**2 + vals[:, 1]**2)
        for idx, global_idx in enumerate(eval_map):
            vel_mag[global_idx] = mag[idx]
    
    u_grid = vel_mag.reshape((nx_out, ny_out))
    
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