import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc

def solve(case_spec: dict) -> dict:
    # Parse parameters
    pde_config = case_spec["oracle_config"]["pde"]
    k = pde_config["wavenumber"]
    
    # Agent-selectable parameters
    mesh_resolution = 128
    element_degree = 2
    ksp_type = "gmres"
    pc_type = "ilu"
    rtol = 1e-8
    
    # Create mesh
    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_square(comm, mesh_resolution, mesh_resolution, 
                                     cell_type=mesh.CellType.triangle)
    
    # Function space
    V = fem.functionspace(domain, ("Lagrange", element_degree))
    
    # Trial and test functions
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    
    # Source term
    x = ufl.SpatialCoordinate(domain)
    f_expr = 10.0 * ufl.exp(-80.0 * ((x[0] - 0.35)**2 + (x[1] - 0.55)**2))
    f = fem.Function(V)
    f.interpolate(fem.Expression(f_expr, V.element.interpolation_points))
    
    # Variational form
    a = (ufl.inner(ufl.grad(u), ufl.grad(v)) - (k**2) * ufl.inner(u, v)) * ufl.dx
    L = ufl.inner(f, v) * ufl.dx
    
    # Boundary conditions (homogeneous Dirichlet)
    tdim = domain.topology.dim
    fdim = tdim - 1
    
    def boundary_marker(x):
        return np.logical_or.reduce([
            np.isclose(x[0], 0.0),
            np.isclose(x[0], 1.0),
            np.isclose(x[1], 0.0),
            np.isclose(x[1], 1.0)
        ])
    
    boundary_facets = mesh.locate_entities_boundary(domain, fdim, boundary_marker)
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    
    u_bc = fem.Function(V)
    u_bc.interpolate(lambda x: np.zeros_like(x[0]))
    bc = fem.dirichletbc(u_bc, boundary_dofs)
    
    # Solve linear problem
    petsc_options = {
        "ksp_type": ksp_type,
        "pc_type": pc_type,
        "ksp_rtol": rtol,
        "ksp_max_it": 1000
    }
    
    problem = petsc.LinearProblem(
        a, L, bcs=[bc],
        petsc_options=petsc_options,
        petsc_options_prefix="helmholtz_"
    )
    
    uh = problem.solve()
    
    # Get solver info
    solver = problem._solver
    ksp = solver.ksp
    its = ksp.getIterationNumber()
    
    solver_info = {
        "mesh_resolution": mesh_resolution,
        "element_degree": element_degree,
        "ksp_type": ksp_type,
        "pc_type": pc_type,
        "rtol": rtol,
        "iterations": its
    }
    
    # Sample solution on 50x50 uniform grid
    nx = ny = 50
    x_vals = np.linspace(0.0, 1.0, nx)
    y_vals = np.linspace(0.0, 1.0, ny)
    X, Y = np.meshgrid(x_vals, y_vals, indexing='ij')
    
    points = np.vstack([X.ravel(), Y.ravel(), np.zeros_like(X.ravel())])
    
    # Evaluate solution at points
    bb_tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, points.T)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points.T)
    
    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    
    for i in range(points.shape[1]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(points.T[i])
            cells_on_proc.append(links[0])
            eval_map.append(i)
    
    u_values = np.full((points.shape[1],), np.nan, dtype=np.float64)
    
    if len(points_on_proc) > 0:
        points_array = np.array(points_on_proc, dtype=np.float64)
        cells_array = np.array(cells_on_proc, dtype=np.int32)
        vals = uh.eval(points_array, cells_array)
        u_values[eval_map] = vals.flatten()
    
    # Gather all values to root process
    if comm.size > 1:
        all_values = np.zeros_like(u_values)
        comm.Allreduce(u_values, all_values, op=MPI.SUM)
        u_values = all_values
    
    u_grid = u_values.reshape((nx, ny))
    
    return {
        "u": u_grid,
        "solver_info": solver_info
    }