import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc

def solve(case_spec: dict) -> dict:
    """
    Solve the Poisson equation -div(kappa * grad(u)) = f on [0,1]x[0,1]
    with Dirichlet BC u = g on all boundaries.
    
    Case: poisson_no_exact_gaussian_source_nonzero_bc
    f = exp(-180*((x-0.3)^2 + (y-0.7)^2))
    g = sin(2*pi*x) + 0.5*cos(2*pi*y)
    kappa = 1.0
    """
    # Parse case_spec
    grid_spec = case_spec["output"]["grid"]
    nx_grid = grid_spec["nx"]
    ny_grid = grid_spec["ny"]
    bbox = grid_spec["bbox"]  # [xmin, xmax, ymin, ymax]
    
    # Parameters chosen for best accuracy within time budget
    mesh_res = 160
    elem_degree = 3
    ksp_type = "cg"
    pc_type = "hypre"
    rtol = 1e-12
    
    # Create mesh on unit square
    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_square(comm, mesh_res, mesh_res, 
                                      cell_type=mesh.CellType.triangle)
    
    # Function space with P3 elements for high accuracy
    V = fem.functionspace(domain, ("Lagrange", elem_degree))
    
    # Boundary conditions - all Dirichlet
    fdim = domain.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
    )
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    
    # BC value: g = sin(2*pi*x) + 0.5*cos(2*pi*y)
    u_bc = fem.Function(V)
    u_bc.interpolate(lambda x: np.sin(2*np.pi*x[0]) + 0.5*np.cos(2*np.pi*x[1]))
    bc = fem.dirichletbc(u_bc, boundary_dofs)
    
    # Variational problem
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    
    # Spatial coordinates for source term
    x_coord = ufl.SpatialCoordinate(domain)
    
    # Source: f = exp(-180*((x-0.3)^2 + (y-0.7)^2))
    f_expr = ufl.exp(-180*((x_coord[0] - 0.3)**2 + (x_coord[1] - 0.7)**2))
    
    # Weak form: kappa * inner(grad(u), grad(v)) dx = f * v dx
    kappa = 1.0
    a = kappa * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = f_expr * v * ufl.dx
    
    # Solve using CG with Hypre AMG preconditioner
    problem = petsc.LinearProblem(
        a, L, bcs=[bc],
        petsc_options={
            "ksp_type": ksp_type,
            "pc_type": pc_type,
            "ksp_rtol": rtol,
            "ksp_atol": 1e-14,
        },
        petsc_options_prefix="poisson_"
    )
    u_sol = problem.solve()
    u_sol.x.scatter_forward()
    
    # Get solver iteration count
    iterations = problem.solver.getIterationNumber()
    
    # Sample solution on output grid
    xmin, xmax, ymin, ymax = bbox
    xs = np.linspace(xmin, xmax, nx_grid)
    ys = np.linspace(ymin, ymax, ny_grid)
    XX, YY = np.meshgrid(xs, ys)
    
    # Build points array: shape (3, N)
    pts = np.vstack([XX.ravel(), YY.ravel(), np.zeros(nx_grid * ny_grid)])
    
    # Find cells containing each point
    bb_tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, pts.T)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, pts.T)
    
    # Evaluate at points (handle MPI distribution)
    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    for i in range(pts.shape[1]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(pts.T[i])
            cells_on_proc.append(links[0])
            eval_map.append(i)
    
    u_values = np.full((pts.shape[1],), np.nan)
    if len(points_on_proc) > 0:
        vals = u_sol.eval(
            np.array(points_on_proc), 
            np.array(cells_on_proc, dtype=np.int32)
        )
        u_values[eval_map] = vals.flatten()
    
    # Reshape to (ny, nx)
    u_grid = u_values.reshape(ny_grid, nx_grid)
    
    # Build solver_info
    solver_info = {
        "mesh_resolution": mesh_res,
        "element_degree": elem_degree,
        "ksp_type": ksp_type,
        "pc_type": pc_type,
        "rtol": rtol,
        "iterations": int(iterations),
    }
    
    return {
        "u": u_grid,
        "solver_info": solver_info,
    }
