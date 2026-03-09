import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import basix.ufl
import ufl
from petsc4py import PETSc


def solve(case_spec: dict) -> dict:
    """Solve steady incompressible Navier-Stokes with manufactured solution."""
    
    comm = MPI.COMM_WORLD
    
    # Parse case spec
    nu_val = 0.01
    if 'pde' in case_spec and 'viscosity' in case_spec['pde']:
        nu_val = case_spec['pde']['viscosity']
    
    # Use P3/P2 Taylor-Hood at N=48 for high accuracy within time budget
    N = 48
    degree_u = 3
    degree_p = 2
    
    # Create mesh
    domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
    
    # Create mixed function space (Taylor-Hood P3/P2)
    P_u = basix.ufl.element("Lagrange", "triangle", degree_u, shape=(2,))
    P_p = basix.ufl.element("Lagrange", "triangle", degree_p)
    ME = basix.ufl.mixed_element([P_u, P_p])
    W = fem.functionspace(domain, ME)
    
    # Collapse subspaces for BCs
    V, V_to_W = W.sub(0).collapse()
    Q, Q_to_W = W.sub(1).collapse()
    
    # Define solution and test functions
    w = fem.Function(W)
    (u, p) = ufl.split(w)
    (v, q) = ufl.TestFunctions(W)
    
    # Spatial coordinates
    x = ufl.SpatialCoordinate(domain)
    pi_val = ufl.pi
    
    # Manufactured solution
    u_exact_0 = 0.2 * pi_val * ufl.cos(pi_val * x[1]) * ufl.sin(2 * pi_val * x[0])
    u_exact_1 = -0.4 * pi_val * ufl.cos(2 * pi_val * x[0]) * ufl.sin(pi_val * x[1])
    u_exact = ufl.as_vector([u_exact_0, u_exact_1])
    
    # Compute source term: f = u·∇u - ν∇²u + ∇p
    # Since p=0, ∇p = 0
    grad_u_exact = ufl.grad(u_exact)
    convection = ufl.dot(grad_u_exact, u_exact)  # grad(u)*u = (u·∇)u
    laplacian_u = ufl.div(ufl.grad(u_exact))
    f = convection - nu_val * laplacian_u
    
    # Viscosity constant
    nu = fem.Constant(domain, PETSc.ScalarType(nu_val))
    
    # Weak form (residual)
    F_form = (
        nu * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
        + ufl.inner(ufl.grad(u) * u, v) * ufl.dx
        - p * ufl.div(v) * ufl.dx
        + ufl.div(u) * q * ufl.dx
        - ufl.inner(f, v) * ufl.dx
    )
    
    # Boundary conditions: u = u_exact on all boundaries
    tdim = domain.topology.dim
    fdim = tdim - 1
    
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
    )
    
    u_bc_func = fem.Function(V)
    u_bc_expr = fem.Expression(u_exact, V.element.interpolation_points)
    u_bc_func.interpolate(u_bc_expr)
    
    dofs_u = fem.locate_dofs_topological((W.sub(0), V), fdim, boundary_facets)
    bc_u = fem.dirichletbc(u_bc_func, dofs_u, W.sub(0))
    
    # Pin pressure at one point to remove nullspace
    p_bc_func = fem.Function(Q)
    p_bc_func.x.array[:] = 0.0
    
    def corner(x):
        return np.logical_and(np.isclose(x[0], 0.0), np.isclose(x[1], 0.0))
    
    corner_facets = mesh.locate_entities_boundary(domain, fdim, corner)
    if len(corner_facets) > 0:
        dofs_p = fem.locate_dofs_topological((W.sub(1), Q), fdim, corner_facets)
        bc_p = fem.dirichletbc(p_bc_func, dofs_p, W.sub(1))
        bcs = [bc_u, bc_p]
    else:
        bcs = [bc_u]
    
    # Initial guess: interpolate exact solution as starting point
    w_init_u = fem.Function(V)
    w_init_u.interpolate(u_bc_expr)
    w.sub(0).interpolate(w_init_u)
    w.x.scatter_forward()
    
    # Set up nonlinear solver
    petsc_options = {
        "snes_type": "newtonls",
        "snes_rtol": 1e-10,
        "snes_atol": 1e-12,
        "snes_max_it": 50,
        "ksp_type": "preonly",
        "pc_type": "lu",
        "pc_factor_mat_solver_type": "mumps",
    }
    
    problem = petsc.NonlinearProblem(
        F_form, w,
        bcs=bcs,
        petsc_options_prefix="ns_",
        petsc_options=petsc_options,
    )
    
    problem.solve()
    
    snes = problem.solver
    converged_reason = snes.getConvergedReason()
    n_newton = snes.getIterationNumber()
    
    if converged_reason <= 0:
        raise RuntimeError(f"SNES did not converge, reason: {converged_reason}")
    
    # Extract velocity for evaluation
    u_h = w.sub(0).collapse()
    
    # Evaluate on 50x50 grid
    nx_eval, ny_eval = 50, 50
    xs = np.linspace(0, 1, nx_eval)
    ys = np.linspace(0, 1, ny_eval)
    XX, YY = np.meshgrid(xs, ys, indexing='ij')
    points_2d = np.column_stack([XX.ravel(), YY.ravel()])
    points_3d = np.zeros((points_2d.shape[0], 3))
    points_3d[:, :2] = points_2d
    
    # Point evaluation
    bb_tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, points_3d)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points_3d)
    
    vel_mag = np.full(points_3d.shape[0], np.nan)
    
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
        vals = u_h.eval(pts_arr, cells_arr)
        for idx, global_idx in enumerate(eval_map):
            ux = vals[idx, 0]
            uy = vals[idx, 1]
            vel_mag[global_idx] = np.sqrt(ux**2 + uy**2)
    
    u_grid = vel_mag.reshape((nx_eval, ny_eval))
    
    output = {
        "u": u_grid,
        "solver_info": {
            "mesh_resolution": N,
            "element_degree": degree_u,
            "ksp_type": "preonly",
            "pc_type": "lu",
            "rtol": 1e-10,
            "nonlinear_iterations": [int(n_newton)],
        }
    }
    
    return output
