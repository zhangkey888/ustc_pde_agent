import numpy as np
from dolfinx import mesh, fem, default_scalar_type, geometry
from dolfinx.fem import petsc
from mpi4py import MPI
import ufl
from petsc4py import PETSc


def solve(case_spec: dict) -> dict:
    # 1. Parse case_spec
    pde_config = case_spec.get("pde", case_spec.get("oracle_config", {}).get("pde", {}))
    
    E = 1.0
    nu_val = 0.49
    
    # Lamé parameters
    lmbda = E * nu_val / ((1.0 + nu_val) * (1.0 - 2.0 * nu_val))
    mu = E / (2.0 * (1.0 + nu_val))
    
    # 2. Create mesh - use higher resolution for near-incompressible
    nx, ny = 80, 80
    degree = 2
    domain = mesh.create_unit_square(MPI.COMM_WORLD, nx, ny, cell_type=mesh.CellType.triangle)
    
    # 3. Vector function space
    V = fem.functionspace(domain, ("Lagrange", degree, (domain.geometry.dim,)))
    
    # 4. Spatial coordinates
    x = ufl.SpatialCoordinate(domain)
    
    # Exact solution
    u_exact = ufl.as_vector([
        ufl.sin(ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1]),
        ufl.sin(ufl.pi * x[0]) * ufl.cos(ufl.pi * x[1])
    ])
    
    # Strain and stress
    def epsilon(u):
        return ufl.sym(ufl.grad(u))
    
    def sigma(u):
        return 2.0 * mu * epsilon(u) + lmbda * ufl.tr(epsilon(u)) * ufl.Identity(domain.geometry.dim)
    
    # Source term from manufactured solution: f = -div(sigma(u_exact))
    f = -ufl.div(sigma(u_exact))
    
    # 5. Variational problem
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    
    a = ufl.inner(sigma(u), epsilon(v)) * ufl.dx
    L = ufl.inner(f, v) * ufl.dx
    
    # 6. Boundary conditions - all boundaries
    tdim = domain.topology.dim
    fdim = tdim - 1
    
    # Locate all boundary facets
    domain.topology.create_connectivity(fdim, tdim)
    boundary_facets = mesh.exterior_facet_indices(domain.topology)
    
    # Create BC function by interpolating exact solution
    u_bc = fem.Function(V)
    u_exact_expr = fem.Expression(u_exact, V.element.interpolation_points)
    u_bc.interpolate(u_exact_expr)
    
    dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc = fem.dirichletbc(u_bc, dofs)
    
    # 7. Solve - use direct solver for near-incompressible case
    ksp_type = "preonly"
    pc_type = "lu"
    rtol = 1e-10
    
    problem = petsc.LinearProblem(
        a, L, bcs=[bc],
        petsc_options={
            "ksp_type": ksp_type,
            "pc_type": pc_type,
            "ksp_rtol": rtol,
        },
        petsc_options_prefix="elasticity_"
    )
    uh = problem.solve()
    
    # Get iteration count
    ksp = problem.solver
    iterations = ksp.getIterationNumber()
    
    # 8. Extract solution on 50x50 uniform grid
    n_eval = 50
    xs = np.linspace(0.0, 1.0, n_eval)
    ys = np.linspace(0.0, 1.0, n_eval)
    XX, YY = np.meshgrid(xs, ys, indexing='ij')
    
    points_2d = np.column_stack([XX.ravel(), YY.ravel()])
    points_3d = np.zeros((points_2d.shape[0], 3))
    points_3d[:, 0] = points_2d[:, 0]
    points_3d[:, 1] = points_2d[:, 1]
    
    # Point evaluation
    bb_tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, points_3d)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points_3d)
    
    u_magnitude = np.full(points_3d.shape[0], np.nan)
    
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
        vals = uh.eval(pts_arr, cells_arr)
        # vals shape: (n_points, 2) for 2D vector
        for idx, global_idx in enumerate(eval_map):
            ux = vals[idx, 0]
            uy = vals[idx, 1]
            u_magnitude[global_idx] = np.sqrt(ux**2 + uy**2)
    
    u_grid = u_magnitude.reshape((n_eval, n_eval))
    
    return {
        "u": u_grid,
        "solver_info": {
            "mesh_resolution": nx,
            "element_degree": degree,
            "ksp_type": ksp_type,
            "pc_type": pc_type,
            "rtol": rtol,
            "iterations": iterations,
        }
    }