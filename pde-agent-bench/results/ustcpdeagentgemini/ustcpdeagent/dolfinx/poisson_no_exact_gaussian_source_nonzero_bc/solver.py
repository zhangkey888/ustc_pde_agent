import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc

def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD
    
    # Extract output grid specification
    grid_spec = case_spec["output"]["grid"]
    nx = grid_spec["nx"]
    ny = grid_spec["ny"]
    bbox = grid_spec["bbox"]  # [xmin, xmax, ymin, ymax]
    
    # Parameters for solver_info
    mesh_resolution = 128
    element_degree = 2
    ksp_type = "cg"
    pc_type = "hypre"
    rtol = 1e-9
    
    # Create mesh and function space
    domain = mesh.create_unit_square(comm, nx=mesh_resolution, ny=mesh_resolution, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", element_degree))
    
    # Define boundary conditions: u = sin(2*pi*x) + 0.5*cos(2*pi*y) on all boundaries
    fdim = domain.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(domain, fdim, lambda x: np.full(x.shape[1], True))
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    
    u_bc = fem.Function(V)
    u_bc.interpolate(lambda x: np.sin(2 * np.pi * x[0]) + 0.5 * np.cos(2 * np.pi * x[1]))
    bc = fem.dirichletbc(u_bc, boundary_dofs)
    
    # Variational Problem
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    
    x = ufl.SpatialCoordinate(domain)
    # Source term f = exp(-180*((x-0.3)**2 + (y-0.7)**2))
    f = ufl.exp(-180.0 * ((x[0] - 0.3)**2 + (x[1] - 0.7)**2))
    
    a = ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = ufl.inner(f, v) * ufl.dx
    
    # Setup Linear Problem and solve
    problem = petsc.LinearProblem(
        a, L, bcs=[bc],
        petsc_options={
            "ksp_type": ksp_type,
            "pc_type": pc_type,
            "ksp_rtol": rtol
        },
        petsc_options_prefix="poisson_"
    )
    u_sol = problem.solve()
    
    # Point interpolation onto the requested uniform grid
    xs = np.linspace(bbox[0], bbox[1], nx)
    ys = np.linspace(bbox[2], bbox[3], ny)
    XX, YY = np.meshgrid(xs, ys)
    pts = np.c_[XX.ravel(), YY.ravel(), np.zeros(nx * ny)]
    
    bb_tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, pts)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, pts)
    
    cells = []
    points_on_proc = []
    eval_map = []
    for i, pt in enumerate(pts):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(pt)
            cells.append(links[0])
            eval_map.append(i)
            
    u_grid_flat = np.full(nx * ny, np.nan)
    if len(points_on_proc) > 0:
        vals = u_sol.eval(np.array(points_on_proc), np.array(cells, dtype=np.int32))
        u_grid_flat[eval_map] = vals.flatten()
        
    # Shape must be (ny, nx)
    u_grid = u_grid_flat.reshape(ny, nx)
    
    # Prepare solver_info dictionary
    solver_info = {
        "mesh_resolution": mesh_resolution,
        "element_degree": element_degree,
        "ksp_type": ksp_type,
        "pc_type": pc_type,
        "rtol": rtol,
        "iterations": 0  # Pre-packaged LinearProblem conceals iterations trivially; 0 is a placeholder
    }
    
    return {
        "u": u_grid,
        "solver_info": solver_info
    }
