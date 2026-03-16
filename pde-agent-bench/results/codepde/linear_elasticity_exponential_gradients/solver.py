import numpy as np
from dolfinx import mesh, fem, default_scalar_type, geometry
from dolfinx.fem import petsc
from mpi4py import MPI
import ufl
from petsc4py import PETSc


def solve(case_spec: dict) -> dict:
    # 1. Parse case_spec
    pde_config = case_spec.get("pde", {})
    
    # Material parameters
    E = 1.0
    nu_poisson = 0.33
    
    # Lamé parameters
    lmbda = E * nu_poisson / ((1.0 + nu_poisson) * (1.0 - 2.0 * nu_poisson))
    mu = E / (2.0 * (1.0 + nu_poisson))
    
    # 2. Create mesh - use high resolution for accuracy
    N = 128
    degree = 2
    domain = mesh.create_unit_square(MPI.COMM_WORLD, N, N, cell_type=mesh.CellType.triangle)
    
    # 3. Vector function space
    V = fem.functionspace(domain, ("Lagrange", degree, (domain.geometry.dim,)))
    
    # 4. Define exact solution and source term using UFL
    x = ufl.SpatialCoordinate(domain)
    pi = ufl.pi
    
    # Exact solution: u = [exp(2*x)*sin(pi*y), -exp(2*y)*sin(pi*x)]
    u_exact = ufl.as_vector([
        ufl.exp(2.0 * x[0]) * ufl.sin(pi * x[1]),
        -ufl.exp(2.0 * x[1]) * ufl.sin(pi * x[0])
    ])
    
    # Strain and stress
    def epsilon(u):
        return ufl.sym(ufl.grad(u))
    
    def sigma(u):
        return 2.0 * mu * epsilon(u) + lmbda * ufl.tr(epsilon(u)) * ufl.Identity(domain.geometry.dim)
    
    # Source term: f = -div(sigma(u_exact))
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
    
    # Build UFL expression for exact solution and interpolate
    u_exact_expr = fem.Expression(
        u_exact,
        V.element.interpolation_points
    )
    u_bc.interpolate(u_exact_expr)
    
    # Locate DOFs on boundary
    dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc = fem.dirichletbc(u_bc, dofs)
    
    # 7. Solve
    ksp_type = "cg"
    pc_type = "hypre"
    rtol = 1e-12
    
    problem = petsc.LinearProblem(
        a, L, bcs=[bc],
        petsc_options={
            "ksp_type": ksp_type,
            "pc_type": pc_type,
            "pc_hypre_type": "boomeramg",
            "ksp_rtol": str(rtol),
            "ksp_atol": "1e-14",
            "ksp_max_it": "2000",
            "ksp_monitor": None,
        },
        petsc_options_prefix="elasticity_"
    )
    uh = problem.solve()
    uh.x.scatter_forward()
    
    # Get iteration count
    ksp = problem.solver
    iterations = ksp.getIterationNumber()
    
    # 8. Extract solution on 50x50 uniform grid
    nx_out, ny_out = 50, 50
    xs = np.linspace(0.0, 1.0, nx_out)
    ys = np.linspace(0.0, 1.0, ny_out)
    XX, YY = np.meshgrid(xs, ys, indexing='ij')
    
    # Points array shape (N, 3) for dolfinx
    points_flat = np.zeros((nx_out * ny_out, 3))
    points_flat[:, 0] = XX.ravel()
    points_flat[:, 1] = YY.ravel()
    
    # Build bounding box tree and find cells
    bb_tree = geometry.bb_tree(domain, tdim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, points_flat)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points_flat)
    
    # Evaluate
    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    
    for i in range(len(points_flat)):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(points_flat[i])
            cells_on_proc.append(links[0])
            eval_map.append(i)
    
    # Displacement magnitude on grid
    u_mag = np.full(nx_out * ny_out, np.nan)
    
    if len(points_on_proc) > 0:
        points_arr = np.array(points_on_proc)
        cells_arr = np.array(cells_on_proc, dtype=np.int32)
        vals = uh.eval(points_arr, cells_arr)
        # vals shape: (n_points, 2) for 2D vector
        mag = np.sqrt(vals[:, 0]**2 + vals[:, 1]**2)
        for idx, global_idx in enumerate(eval_map):
            u_mag[global_idx] = mag[idx]
    
    u_grid = u_mag.reshape((nx_out, ny_out))
    
    return {
        "u": u_grid,
        "solver_info": {
            "mesh_resolution": N,
            "element_degree": degree,
            "ksp_type": ksp_type,
            "pc_type": pc_type,
            "rtol": rtol,
            "iterations": iterations,
        }
    }