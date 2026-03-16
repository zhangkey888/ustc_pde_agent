import numpy as np
from dolfinx import mesh, fem, default_scalar_type, geometry
from dolfinx.fem import petsc
from mpi4py import MPI
import ufl
from petsc4py import PETSc
import basix


def solve(case_spec: dict) -> dict:
    # 1. Parse case_spec
    pde_config = case_spec.get("pde", {})
    nu_val = pde_config.get("viscosity", 5.0)
    
    # 2. Create mesh - use sufficient resolution for accuracy
    N = 64
    domain = mesh.create_unit_square(MPI.COMM_WORLD, N, N, cell_type=mesh.CellType.triangle)
    
    # 3. Mixed function space (Taylor-Hood: P2/P1)
    gdim = domain.geometry.dim
    
    # Velocity space P2 vector
    V_el = basix.ufl.element("Lagrange", domain.basix_cell(), 2, shape=(gdim,))
    # Pressure space P1
    Q_el = basix.ufl.element("Lagrange", domain.basix_cell(), 1)
    # Mixed element
    ME = basix.ufl.mixed_element([V_el, Q_el])
    W = fem.functionspace(domain, ME)
    
    # Sub spaces for BCs
    V_sub, _ = W.sub(0).collapse()
    Q_sub, _ = W.sub(1).collapse()
    
    # 4. Define exact solution for BCs and source term
    x = ufl.SpatialCoordinate(domain)
    pi = ufl.pi
    
    # Exact velocity: u = (pi*cos(pi*y)*sin(pi*x), -pi*cos(pi*x)*sin(pi*y))
    u_exact_0 = pi * ufl.cos(pi * x[1]) * ufl.sin(pi * x[0])
    u_exact_1 = -pi * ufl.cos(pi * x[0]) * ufl.sin(pi * x[1])
    u_exact = ufl.as_vector([u_exact_0, u_exact_1])
    
    # Exact pressure: p = cos(pi*x)*cos(pi*y)
    p_exact = ufl.cos(pi * x[0]) * ufl.cos(pi * x[1])
    
    # Source term: f = -nu * laplacian(u_exact) + grad(p_exact)
    # For u_exact_0 = pi*cos(pi*y)*sin(pi*x):
    #   d^2/dx^2 = -pi^3*cos(pi*y)*sin(pi*x)
    #   d^2/dy^2 = -pi^3*cos(pi*y)*sin(pi*x) (wait, let me recheck)
    #   Actually: d^2(u0)/dx^2 = pi * cos(pi*y) * (-pi^2 * sin(pi*x)) = -pi^3 * cos(pi*y)*sin(pi*x)
    #            d^2(u0)/dy^2 = pi * (-pi^2 * cos(pi*y)) * sin(pi*x) = -pi^3 * sin(pi*x)*cos(pi*y)
    #   laplacian(u0) = -2*pi^3*sin(pi*x)*cos(pi*y)
    # Similarly for u1.
    # grad(p) = (-pi*sin(pi*x)*cos(pi*y), -pi*cos(pi*x)*sin(pi*y))
    
    # Let UFL compute the source term symbolically
    nu_c = fem.Constant(domain, default_scalar_type(nu_val))
    
    # f = -nu * div(grad(u_exact)) + grad(p_exact)
    f_expr = -nu_c * ufl.div(ufl.grad(u_exact)) + ufl.grad(p_exact)
    
    # 5. Trial and test functions
    (u, p) = ufl.TrialFunctions(W)
    (v, q) = ufl.TestFunctions(W)
    
    # 6. Variational form for Stokes:
    # a((u,p),(v,q)) = nu*(grad(u),grad(v)) - (p, div(v)) - (div(u), q)
    # L((v,q)) = (f, v)
    a = (nu_c * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
         - p * ufl.div(v) * ufl.dx
         - ufl.div(u) * q * ufl.dx)
    
    L = ufl.inner(f_expr, v) * ufl.dx
    
    # 7. Boundary conditions - apply exact velocity on all boundaries
    tdim = domain.topology.dim
    fdim = tdim - 1
    
    # Find all boundary facets
    domain.topology.create_connectivity(fdim, tdim)
    boundary_facets = mesh.exterior_facet_indices(domain.topology)
    
    # Interpolate exact velocity BC
    u_bc_func = fem.Function(V_sub)
    u_bc_expr = fem.Expression(u_exact, V_sub.element.interpolation_points)
    u_bc_func.interpolate(u_bc_expr)
    
    dofs_u = fem.locate_dofs_topological((W.sub(0), V_sub), fdim, boundary_facets)
    bc_u = fem.dirichletbc(u_bc_func, dofs_u, W.sub(0))
    
    # Pin pressure at one point to remove nullspace
    # Find a DOF near (0,0) for pressure
    def corner_marker(x):
        return np.isclose(x[0], 0.0) & np.isclose(x[1], 0.0)
    
    # Pin pressure DOF
    p_bc_func = fem.Function(Q_sub)
    p_bc_expr = fem.Expression(p_exact, Q_sub.element.interpolation_points)
    p_bc_func.interpolate(p_bc_expr)
    
    # Use a vertex-based approach for pressure pin
    pressure_dofs = fem.locate_dofs_geometrical((W.sub(1), Q_sub), corner_marker)
    bc_p = fem.dirichletbc(p_bc_func, pressure_dofs, W.sub(1))
    
    bcs = [bc_u, bc_p]
    
    # 8. Solve
    ksp_type = "gmres"
    pc_type = "lu"
    rtol = 1e-12
    
    problem = petsc.LinearProblem(
        a, L, bcs=bcs,
        petsc_options={
            "ksp_type": ksp_type,
            "pc_type": pc_type,
            "ksp_rtol": str(rtol),
            "ksp_max_it": "1000",
        },
        petsc_options_prefix="stokes_"
    )
    wh = problem.solve()
    
    # Get iteration count
    ksp = problem.solver
    iterations = ksp.getIterationNumber()
    
    # 9. Extract velocity sub-function
    uh = wh.sub(0).collapse()
    
    # 10. Evaluate velocity magnitude on 100x100 grid
    nx_eval, ny_eval = 100, 100
    xs = np.linspace(0, 1, nx_eval)
    ys = np.linspace(0, 1, ny_eval)
    XX, YY = np.meshgrid(xs, ys, indexing='ij')
    
    points = np.zeros((3, nx_eval * ny_eval))
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
    
    u_mag = np.full(nx_eval * ny_eval, np.nan)
    
    if len(points_on_proc) > 0:
        pts_arr = np.array(points_on_proc)
        cells_arr = np.array(cells_on_proc, dtype=np.int32)
        vals = uh.eval(pts_arr, cells_arr)
        # vals shape: (n_points, gdim)
        mag = np.sqrt(vals[:, 0]**2 + vals[:, 1]**2)
        for idx, global_idx in enumerate(eval_map):
            u_mag[global_idx] = mag[idx]
    
    u_grid = u_mag.reshape((nx_eval, ny_eval))
    
    return {
        "u": u_grid,
        "solver_info": {
            "mesh_resolution": N,
            "element_degree": 2,
            "ksp_type": ksp_type,
            "pc_type": pc_type,
            "rtol": rtol,
            "iterations": iterations,
            "pressure_fixing": "dirichlet_corner",
        }
    }