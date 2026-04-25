import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
import ufl
from dolfinx import mesh, fem, geometry
from dolfinx.fem.petsc import LinearProblem
import time

def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD
    
    # Mesh and parameters
    nx_mesh = 128
    ny_mesh = 128
    degree = 2
    domain = mesh.create_unit_square(comm, nx=nx_mesh, ny=ny_mesh, cell_type=mesh.CellType.triangle)
    
    V = fem.functionspace(domain, ("Lagrange", degree))
    
    x = ufl.SpatialCoordinate(domain)
    
    # Exact solution
    u_exact = ufl.sin(ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])
    
    # Kappa coefficient
    kappa = 0.2 + ufl.exp(-120.0 * ((x[0] - 0.55)**2 + (x[1] - 0.45)**2))
    
    # Derive RHS symbolicly
    f = -ufl.div(kappa * ufl.grad(u_exact))
    
    # Boundary condition
    fdim = domain.topology.dim - 1
    
    # Locate boundary facets topologically
    boundary_facets = mesh.locate_entities_boundary(domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool))
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    
    u_bc_func = fem.Function(V)
    u_bc_func.interpolate(fem.Expression(u_exact, V.element.interpolation_points))
    bc = fem.dirichletbc(u_bc_func, boundary_dofs)
    
    # Variational forms
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    
    a = ufl.inner(kappa * ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = ufl.inner(f, v) * ufl.dx
    
    # Solve
    ksp_type = "cg"
    pc_type = "hypre"
    rtol = 1e-8
    
    petsc_options = {
        "ksp_type": ksp_type,
        "pc_type": pc_type,
        "ksp_rtol": rtol
    }
    
    problem = LinearProblem(a, L, bcs=[bc],
                            petsc_options=petsc_options,
                            petsc_options_prefix="poisson_")
    
    u_sol = problem.solve()
    
    # Collect iterations
    solver_ksp = problem.solver
    iterations = solver_ksp.getIterationNumber()
    
    # Interpolate onto grid
    grid_spec = case_spec["output"]["grid"]
    nx_grid = grid_spec["nx"]
    ny_grid = grid_spec["ny"]
    bbox = grid_spec["bbox"]
    
    x_lin = np.linspace(bbox[0], bbox[1], nx_grid)
    y_lin = np.linspace(bbox[2], bbox[3], ny_grid)
    XX, YY = np.meshgrid(x_lin, y_lin)
    pts = np.vstack([XX.ravel(), YY.ravel(), np.zeros_like(XX.ravel())])
    
    # Probe points
    bb_tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, pts.T)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, pts.T)
    
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
        vals = u_sol.eval(np.array(points_on_proc, dtype=np.float64), np.array(cells_on_proc, dtype=np.int32))
        u_values[eval_map] = vals.flatten()
        
    u_grid = u_values.reshape((ny_grid, nx_grid))
    
    solver_info = {
        "mesh_resolution": nx_mesh,
        "element_degree": degree,
        "ksp_type": ksp_type,
        "pc_type": pc_type,
        "rtol": rtol,
        "iterations": iterations
    }
    
    return {
        "u": u_grid,
        "solver_info": solver_info
    }
