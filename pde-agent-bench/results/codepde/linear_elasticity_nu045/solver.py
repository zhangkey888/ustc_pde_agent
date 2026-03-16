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
    nu_val = 0.45
    
    # Lamé parameters
    lam = E * nu_val / ((1.0 + nu_val) * (1.0 - 2.0 * nu_val))
    mu = E / (2.0 * (1.0 + nu_val))
    
    # Mesh resolution and element degree
    N = 80
    degree = 2
    
    # 2. Create mesh
    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
    
    # 3. Vector function space
    V = fem.functionspace(domain, ("Lagrange", degree, (domain.geometry.dim,)))
    
    # 4. Define variational problem
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    
    x = ufl.SpatialCoordinate(domain)
    
    # Exact solution
    u_exact_0 = ufl.sin(ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])
    u_exact_1 = ufl.cos(ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])
    u_exact = ufl.as_vector([u_exact_0, u_exact_1])
    
    # Strain and stress
    def epsilon(w):
        return ufl.sym(ufl.grad(w))
    
    def sigma(w):
        return 2.0 * mu * epsilon(w) + lam * ufl.tr(epsilon(w)) * ufl.Identity(domain.geometry.dim)
    
    # Source term: f = -div(sigma(u_exact))
    f = -ufl.div(sigma(u_exact))
    
    # Bilinear and linear forms
    a = ufl.inner(sigma(u), epsilon(v)) * ufl.dx
    L = ufl.inner(f, v) * ufl.dx
    
    # 5. Boundary conditions - all boundary
    tdim = domain.topology.dim
    fdim = tdim - 1
    
    # Find all boundary facets
    domain.topology.create_connectivity(fdim, tdim)
    boundary_facets = mesh.exterior_facet_indices(domain.topology)
    
    # Create BC function
    u_bc = fem.Function(V)
    u_bc_expr = fem.Expression(u_exact, V.element.interpolation_points)
    u_bc.interpolate(u_bc_expr)
    
    dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc = fem.dirichletbc(u_bc, dofs)
    
    # 6. Solve
    ksp_type = "cg"
    pc_type = "hypre"
    rtol = 1e-10
    
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
    
    # Get iteration count
    ksp = problem.solver
    iterations = ksp.getIterationNumber()
    
    # 7. Extract on 50x50 uniform grid - displacement magnitude
    nx_out, ny_out = 50, 50
    xs = np.linspace(0.0, 1.0, nx_out)
    ys = np.linspace(0.0, 1.0, ny_out)
    XX, YY = np.meshgrid(xs, ys, indexing='ij')
    
    # Points array shape (N, 3) for evaluation
    points_flat = np.zeros((nx_out * ny_out, 3))
    points_flat[:, 0] = XX.ravel()
    points_flat[:, 1] = YY.ravel()
    
    # Build bounding box tree
    bb_tree = geometry.bb_tree(domain, domain.topology.dim)
    
    # Find cells
    cell_candidates = geometry.compute_collisions_points(bb_tree, points_flat)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points_flat)
    
    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    for i in range(len(points_flat)):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(points_flat[i])
            cells_on_proc.append(links[0])
            eval_map.append(i)
    
    u_magnitude = np.full(nx_out * ny_out, np.nan)
    if len(points_on_proc) > 0:
        pts_arr = np.array(points_on_proc)
        cells_arr = np.array(cells_on_proc, dtype=np.int32)
        vals = uh.eval(pts_arr, cells_arr)  # shape (n_points, 2)
        # Compute magnitude
        mag = np.sqrt(vals[:, 0]**2 + vals[:, 1]**2)
        for idx, global_idx in enumerate(eval_map):
            u_magnitude[global_idx] = mag[idx]
    
    u_grid = u_magnitude.reshape((nx_out, ny_out))
    
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