import numpy as np
from dolfinx import mesh, fem, default_scalar_type, geometry
from dolfinx.fem import petsc
from mpi4py import MPI
import ufl
from petsc4py import PETSc


def solve(case_spec: dict) -> dict:
    # 1. Parse case_spec
    pde_config = case_spec.get("pde", {})
    material = pde_config.get("material", {})
    E_val = material.get("E", 1.0)
    nu_val = material.get("nu", 0.3)
    
    # Lamé parameters
    mu_val = E_val / (2.0 * (1.0 + nu_val))
    lam_val = E_val * nu_val / ((1.0 + nu_val) * (1.0 - 2.0 * nu_val))
    
    # Mesh and element parameters
    N = 64
    degree = 2
    
    # 2. Create mesh - quadrilateral as specified
    domain = mesh.create_unit_square(MPI.COMM_WORLD, N, N, cell_type=mesh.CellType.quadrilateral)
    
    # 3. Vector function space
    V = fem.functionspace(domain, ("Lagrange", degree, (domain.geometry.dim,)))
    
    # 4. Define variational problem
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    
    x = ufl.SpatialCoordinate(domain)
    pi = ufl.pi
    
    mu = fem.Constant(domain, PETSc.ScalarType(mu_val))
    lam = fem.Constant(domain, PETSc.ScalarType(lam_val))
    
    def epsilon(u):
        return ufl.sym(ufl.grad(u))
    
    def sigma(u):
        return 2.0 * mu * epsilon(u) + lam * ufl.tr(epsilon(u)) * ufl.Identity(domain.geometry.dim)
    
    # Exact solution
    u_exact = ufl.as_vector([
        ufl.sin(2 * pi * x[0]) * ufl.cos(3 * pi * x[1]),
        ufl.sin(pi * x[0]) * ufl.sin(2 * pi * x[1])
    ])
    
    # Compute source term from manufactured solution: f = -div(sigma(u_exact))
    f = -ufl.div(sigma(u_exact))
    
    # Bilinear and linear forms
    a = ufl.inner(sigma(u), epsilon(v)) * ufl.dx
    L = ufl.inner(f, v) * ufl.dx
    
    # 5. Boundary conditions - apply exact solution on all boundaries
    u_bc = fem.Function(V)
    
    # Create expression for exact solution
    u_exact_expr = fem.Expression(u_exact, V.element.interpolation_points)
    u_bc.interpolate(u_exact_expr)
    
    tdim = domain.topology.dim
    fdim = tdim - 1
    
    # All boundary facets
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
    )
    dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc = fem.dirichletbc(u_bc, dofs)
    
    # 6. Solve
    problem = petsc.LinearProblem(
        a, L, bcs=[bc],
        petsc_options={
            "ksp_type": "cg",
            "pc_type": "gamg",
            "ksp_rtol": 1e-10,
            "ksp_atol": 1e-12,
            "ksp_max_it": 1000,
        },
        petsc_options_prefix="elasticity_"
    )
    uh = problem.solve()
    
    # Get iteration count
    ksp = problem.solver
    iterations = ksp.getIterationNumber()
    
    # 7. Extract on 50x50 uniform grid
    nx_out, ny_out = 50, 50
    xs = np.linspace(0.0, 1.0, nx_out)
    ys = np.linspace(0.0, 1.0, ny_out)
    XX, YY = np.meshgrid(xs, ys, indexing='ij')
    
    points_2d = np.column_stack([XX.ravel(), YY.ravel()])
    points_3d = np.zeros((points_2d.shape[0], 3))
    points_3d[:, :2] = points_2d
    
    # Point evaluation
    bb_tree = geometry.bb_tree(domain, domain.topology.dim)
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
    
    n_points = points_3d.shape[0]
    u_magnitude = np.full(n_points, np.nan)
    
    if len(points_on_proc) > 0:
        pts_arr = np.array(points_on_proc)
        cells_arr = np.array(cells_on_proc, dtype=np.int32)
        vals = uh.eval(pts_arr, cells_arr)
        # vals shape: (n_eval_points, 2) for 2D vector
        mag = np.sqrt(vals[:, 0]**2 + vals[:, 1]**2)
        for idx, global_idx in enumerate(eval_map):
            u_magnitude[global_idx] = mag[idx]
    
    u_grid = u_magnitude.reshape((nx_out, ny_out))
    
    return {
        "u": u_grid,
        "solver_info": {
            "mesh_resolution": N,
            "element_degree": degree,
            "ksp_type": "cg",
            "pc_type": "gamg",
            "rtol": 1e-10,
            "iterations": int(iterations),
        }
    }