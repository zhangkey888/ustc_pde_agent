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
    nu_val = pde_config.get("viscosity", 1.0)
    
    # 2. Create mesh - quadrilateral mesh as specified
    N = 64
    domain = mesh.create_unit_square(MPI.COMM_WORLD, N, N, cell_type=mesh.CellType.quadrilateral)
    
    tdim = domain.topology.dim
    fdim = tdim - 1
    
    # 3. Define mixed function space (Taylor-Hood: P2/P1 on quads)
    # For quads, use Lagrange elements
    P2 = basix.ufl.element("Lagrange", domain.basix_cell(), 2, shape=(domain.geometry.dim,))
    P1 = basix.ufl.element("Lagrange", domain.basix_cell(), 1)
    TH = basix.ufl.mixed_element([P2, P1])
    
    W = fem.functionspace(domain, TH)
    
    # Extract sub-spaces for BCs
    V_sub, _ = W.sub(0).collapse()
    Q_sub, _ = W.sub(1).collapse()
    
    # 4. Define trial and test functions
    (u, p) = ufl.TrialFunctions(W)
    (v, q) = ufl.TestFunctions(W)
    
    # Spatial coordinates
    x = ufl.SpatialCoordinate(domain)
    pi = ufl.pi
    
    # Exact solution
    u_exact = ufl.as_vector([pi * ufl.cos(pi * x[1]) * ufl.sin(pi * x[0]),
                              -pi * ufl.cos(pi * x[0]) * ufl.sin(pi * x[1])])
    p_exact = ufl.cos(pi * x[0]) * ufl.cos(pi * x[1])
    
    # Compute source term: f = -nu * laplacian(u_exact) + grad(p_exact)
    nu_c = fem.Constant(domain, default_scalar_type(nu_val))
    
    # For the manufactured solution, compute f analytically
    # u1 = pi*cos(pi*y)*sin(pi*x)
    # u2 = -pi*cos(pi*x)*sin(pi*y)
    # laplacian(u1) = -2*pi^3*cos(pi*y)*sin(pi*x)
    # laplacian(u2) = 2*pi^3*cos(pi*x)*sin(pi*y)
    # grad(p) = (-pi*sin(pi*x)*cos(pi*y), -pi*cos(pi*x)*sin(pi*y))
    
    # f1 = -nu*(-2*pi^3*cos(pi*y)*sin(pi*x)) + (-pi*sin(pi*x)*cos(pi*y))
    #     = 2*nu*pi^3*cos(pi*y)*sin(pi*x) - pi*sin(pi*x)*cos(pi*y)
    #     = pi*cos(pi*y)*sin(pi*x)*(2*nu*pi^2 - 1)
    
    # f2 = -nu*(2*pi^3*cos(pi*x)*sin(pi*y)) + (-pi*cos(pi*x)*sin(pi*y))
    #     = -2*nu*pi^3*cos(pi*x)*sin(pi*y) - pi*cos(pi*x)*sin(pi*y)
    #     = -pi*cos(pi*x)*sin(pi*y)*(2*nu*pi^2 + 1)
    
    f = ufl.as_vector([
        pi * ufl.cos(pi * x[1]) * ufl.sin(pi * x[0]) * (2.0 * nu_val * pi**2 - 1.0),
        -pi * ufl.cos(pi * x[0]) * ufl.sin(pi * x[1]) * (2.0 * nu_val * pi**2 + 1.0)
    ])
    
    # 5. Variational forms
    a = (nu_c * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
         - p * ufl.div(v) * ufl.dx
         + q * ufl.div(u) * ufl.dx)
    
    L = ufl.inner(f, v) * ufl.dx
    
    # 6. Boundary conditions - apply exact velocity on all boundaries
    u_bc_func = fem.Function(V_sub)
    u_bc_func.interpolate(lambda x: np.vstack([
        np.pi * np.cos(np.pi * x[1]) * np.sin(np.pi * x[0]),
        -np.pi * np.cos(np.pi * x[0]) * np.sin(np.pi * x[1])
    ]))
    
    # Find all boundary facets
    domain.topology.create_connectivity(fdim, tdim)
    boundary_facets = mesh.exterior_facets(domain.topology)
    
    dofs_u = fem.locate_dofs_topological((W.sub(0), V_sub), fdim, boundary_facets)
    bc_u = fem.dirichletbc(u_bc_func, dofs_u, W.sub(0))
    
    # Pin pressure at one point to fix the nullspace
    # Find a DOF near (0, 0)
    p_bc_func = fem.Function(Q_sub)
    p_bc_func.interpolate(lambda x: np.cos(np.pi * x[0]) * np.cos(np.pi * x[1]))
    
    # Pin pressure at corner
    def corner(x):
        return np.logical_and(np.isclose(x[0], 0.0), np.isclose(x[1], 0.0))
    
    corner_facets = mesh.locate_entities_boundary(domain, fdim, corner)
    if len(corner_facets) > 0:
        dofs_p = fem.locate_dofs_topological((W.sub(1), Q_sub), fdim, corner_facets)
        bc_p = fem.dirichletbc(p_bc_func, dofs_p, W.sub(1))
        bcs = [bc_u, bc_p]
    else:
        bcs = [bc_u]
    
    # 7. Solve
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
            "ksp_monitor": None,
        },
        petsc_options_prefix="stokes_"
    )
    wh = problem.solve()
    
    # Get iteration count
    ksp = problem.solver
    iterations = ksp.getIterationNumber()
    
    # 8. Extract velocity and compute magnitude on 100x100 grid
    nx_out, ny_out = 100, 100
    xs = np.linspace(0, 1, nx_out)
    ys = np.linspace(0, 1, ny_out)
    X, Y = np.meshgrid(xs, ys, indexing='ij')
    points = np.zeros((3, nx_out * ny_out))
    points[0, :] = X.ravel()
    points[1, :] = Y.ravel()
    
    # Point evaluation
    bb_tree = geometry.bb_tree(domain, tdim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, points.T)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points.T)
    
    # Extract velocity sub-function
    uh = wh.sub(0).collapse()
    
    vel_mag = np.full(nx_out * ny_out, np.nan)
    
    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    for i in range(nx_out * ny_out):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(points[:, i])
            cells_on_proc.append(links[0])
            eval_map.append(i)
    
    if len(points_on_proc) > 0:
        pts_arr = np.array(points_on_proc)
        cells_arr = np.array(cells_on_proc, dtype=np.int32)
        vals = uh.eval(pts_arr, cells_arr)
        # vals shape: (n_points, gdim)
        for idx, global_idx in enumerate(eval_map):
            vx = vals[idx, 0]
            vy = vals[idx, 1]
            vel_mag[global_idx] = np.sqrt(vx**2 + vy**2)
    
    u_grid = vel_mag.reshape((nx_out, ny_out))
    
    return {
        "u": u_grid,
        "solver_info": {
            "mesh_resolution": N,
            "element_degree": 2,
            "ksp_type": ksp_type,
            "pc_type": pc_type,
            "rtol": rtol,
            "iterations": iterations,
        }
    }