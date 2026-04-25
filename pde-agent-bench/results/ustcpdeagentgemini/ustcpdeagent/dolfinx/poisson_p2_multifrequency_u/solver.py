import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
import ufl
from dolfinx.fem import petsc
from petsc4py import PETSc
import time

def solve(case_spec: dict) -> dict:
    # Extract grid info
    grid_spec = case_spec.get("output", {}).get("grid", {})
    nx_out = grid_spec.get("nx", 100)
    ny_out = grid_spec.get("ny", 100)
    bbox = grid_spec.get("bbox", [0.0, 1.0, 0.0, 1.0])
    xmin, xmax, ymin, ymax = bbox
    
    # Mesh and function space (fine mesh and P2 to meet strict accuracy requirement)
    mesh_res = 128
    degree = 2
    comm = MPI.COMM_WORLD
    domain = mesh.create_rectangle(comm, [[xmin, ymin], [xmax, ymax]], [mesh_res, mesh_res], cell_type=mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", degree))
    
    # Coordinates
    x = ufl.SpatialCoordinate(domain)
    
    # Exact solution for f and BC
    u_ex = ufl.sin(ufl.pi*x[0])*ufl.sin(ufl.pi*x[1]) + 0.2*ufl.sin(5*ufl.pi*x[0])*ufl.sin(4*ufl.pi*x[1])
    
    # Source term f = -Delta u
    # div(grad(sin(pi*x)*sin(pi*y))) = -2*pi^2 * sin(pi*x)*sin(pi*y)
    # div(grad(sin(5*pi*x)*sin(4*pi*y))) = -(25*pi^2 + 16*pi^2) * sin(5*pi*x)*sin(4*pi*y) = -41*pi^2 * ...
    f_expr = 2 * ufl.pi**2 * ufl.sin(ufl.pi*x[0])*ufl.sin(ufl.pi*x[1]) + \
             0.2 * 41 * ufl.pi**2 * ufl.sin(5*ufl.pi*x[0])*ufl.sin(4*ufl.pi*x[1])
             
    # Trial and Test functions
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    
    # Variational problem
    a = ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = ufl.inner(f_expr, v) * ufl.dx
    
    # Boundary conditions
    tdim = domain.topology.dim
    fdim = tdim - 1
    domain.topology.create_connectivity(fdim, tdim)
    boundary_facets = mesh.exterior_facet_indices(domain.topology)
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    
    # We interpolate exact solution to get BC values
    u_bc = fem.Function(V)
    expr_bc = fem.Expression(u_ex, V.element.interpolation_points())
    u_bc.interpolate(expr_bc)
    bc = fem.dirichletbc(u_bc, boundary_dofs)
    
    # Solve
    problem = petsc.LinearProblem(
        a, L, bcs=[bc],
        petsc_options={"ksp_type": "cg", "pc_type": "hypre", "ksp_rtol": 1e-9},
        petsc_options_prefix="poisson_"
    )
    u_sol = problem.solve()
    
    # Output grid evaluation
    xs = np.linspace(xmin, xmax, nx_out)
    ys = np.linspace(ymin, ymax, ny_out)
    XX, YY = np.meshgrid(xs, ys)
    points = np.c_[XX.ravel(), YY.ravel(), np.zeros(nx_out * ny_out)]
    
    bb_tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, points)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points)
    
    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    for i in range(len(points)):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(points[i])
            cells_on_proc.append(links[0])
            eval_map.append(i)
            
    u_grid = np.full((len(points),), np.nan)
    if len(points_on_proc) > 0:
        vals = u_sol.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u_grid[eval_map] = vals.flatten()
        
    u_grid = u_grid.reshape((ny_out, nx_out))
    
    # Get solver iteration count if possible
    iters = problem.solver.getIterationNumber()
    
    solver_info = {
        "mesh_resolution": mesh_res,
        "element_degree": degree,
        "ksp_type": "cg",
        "pc_type": "hypre",
        "rtol": 1e-9,
        "iterations": iters
    }
    
    return {"u": u_grid, "solver_info": solver_info}

