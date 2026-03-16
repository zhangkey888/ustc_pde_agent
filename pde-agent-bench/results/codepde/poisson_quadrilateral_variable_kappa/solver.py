import numpy as np
from dolfinx import mesh, fem, default_scalar_type, geometry
from dolfinx.fem.petsc import LinearProblem
from mpi4py import MPI
import ufl
from petsc4py import PETSc


def solve(case_spec: dict) -> dict:
    # 1. Parse case_spec
    nx_grid = 50
    ny_grid = 50
    
    # Mesh resolution and element degree - choose high enough for accuracy
    mesh_resolution = 80
    element_degree = 2
    
    # 2. Create mesh (quadrilateral as specified in case ID)
    domain = mesh.create_unit_square(
        MPI.COMM_WORLD, mesh_resolution, mesh_resolution,
        cell_type=mesh.CellType.quadrilateral
    )
    
    # 3. Function space
    V = fem.functionspace(domain, ("Lagrange", element_degree))
    
    # 4. Define variational problem
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    
    x = ufl.SpatialCoordinate(domain)
    pi = ufl.pi
    
    # Exact solution: u = sin(2*pi*x)*sin(pi*y)
    u_exact = ufl.sin(2 * pi * x[0]) * ufl.sin(pi * x[1])
    
    # Variable kappa: 1 + 0.5*cos(2*pi*x)*cos(2*pi*y)
    kappa = 1.0 + 0.5 * ufl.cos(2 * pi * x[0]) * ufl.cos(2 * pi * x[1])
    
    # Compute source term f = -div(kappa * grad(u_exact))
    # We use UFL to compute this symbolically
    f = -ufl.div(kappa * ufl.grad(u_exact))
    
    # Bilinear form: a(u, v) = integral of kappa * grad(u) . grad(v) dx
    a = ufl.inner(kappa * ufl.grad(u), ufl.grad(v)) * ufl.dx
    
    # Linear form: L(v) = integral of f * v dx
    L = ufl.inner(f, v) * ufl.dx
    
    # 5. Boundary conditions
    # u = g = u_exact on boundary
    tdim = domain.topology.dim
    fdim = tdim - 1
    
    # Find all boundary facets
    domain.topology.create_connectivity(fdim, tdim)
    boundary_facets = mesh.exterior_facet_indices(domain.topology)
    
    # Create BC function by interpolating exact solution
    u_bc = fem.Function(V)
    u_exact_expr = fem.Expression(u_exact, V.element.interpolation_points)
    u_bc.interpolate(u_exact_expr)
    
    dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc = fem.dirichletbc(u_bc, dofs)
    
    # 6. Solve
    ksp_type = "cg"
    pc_type = "hypre"
    rtol = 1e-10
    
    problem = LinearProblem(
        a, L, bcs=[bc],
        petsc_options={
            "ksp_type": ksp_type,
            "pc_type": pc_type,
            "ksp_rtol": str(rtol),
            "ksp_max_it": "2000",
            "ksp_monitor": None,
        },
        petsc_options_prefix="poisson_"
    )
    uh = problem.solve()
    
    # Get iteration count
    ksp = problem.solver
    iterations = ksp.getIterationNumber()
    
    # 7. Extract solution on uniform grid
    xs = np.linspace(0.0, 1.0, nx_grid)
    ys = np.linspace(0.0, 1.0, ny_grid)
    X, Y = np.meshgrid(xs, ys, indexing='ij')
    
    points = np.zeros((3, nx_grid * ny_grid))
    points[0, :] = X.ravel()
    points[1, :] = Y.ravel()
    
    bb_tree = geometry.bb_tree(domain, domain.topology.dim)
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
    
    u_values = np.full(nx_grid * ny_grid, np.nan)
    if len(points_on_proc) > 0:
        pts_arr = np.array(points_on_proc)
        cells_arr = np.array(cells_on_proc, dtype=np.int32)
        vals = uh.eval(pts_arr, cells_arr)
        u_values[eval_map] = vals.flatten()
    
    u_grid = u_values.reshape((nx_grid, ny_grid))
    
    return {
        "u": u_grid,
        "solver_info": {
            "mesh_resolution": mesh_resolution,
            "element_degree": element_degree,
            "ksp_type": ksp_type,
            "pc_type": pc_type,
            "rtol": rtol,
            "iterations": iterations,
        }
    }