import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc

def solve(case_spec: dict) -> dict:
    # 1. Configuration & Mesh Generation
    nx_mesh = 64
    ny_mesh = 64
    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_square(comm, nx_mesh, ny_mesh, cell_type=mesh.CellType.triangle)
    
    # 2. Function Space
    degree = 2
    V = fem.functionspace(domain, ("Lagrange", degree))
    
    # 3. Boundary Conditions
    tdim = domain.topology.dim
    fdim = tdim - 1
    
    def boundary_marker(x):
        # Return all boundaries for the unit square
        return np.logical_or.reduce([
            np.isclose(x[0], 0.0),
            np.isclose(x[0], 1.0),
            np.isclose(x[1], 0.0),
            np.isclose(x[1], 1.0)
        ])
        
    boundary_facets = mesh.locate_entities_boundary(domain, fdim, boundary_marker)
    dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    
    u_bc = fem.Function(V)
    u_bc.interpolate(lambda x: np.cos(2*np.pi*x[0]) * np.cos(3*np.pi*x[1]))
    bc = fem.dirichletbc(u_bc, dofs)
    
    # 4. Variational Problem Setup
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    
    kappa = fem.Constant(domain, PETSc.ScalarType(5.0))
    x = ufl.SpatialCoordinate(domain)
    
    # -div(5.0 grad(u)) = f
    # u = cos(2*pi*x)*cos(3*pi*y)
    # laplacian(u) = -4*pi^2 u - 9*pi^2 u = -13*pi^2 u
    # f = -5.0 * laplacian(u) = 65*pi^2 u
    f_expr = 65.0 * ufl.pi**2 * ufl.cos(2*ufl.pi*x[0]) * ufl.cos(3*ufl.pi*x[1])
    
    a = ufl.inner(kappa * ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = ufl.inner(f_expr, v) * ufl.dx
    
    # 5. Linear Solver
    problem = petsc.LinearProblem(
        a, L, bcs=[bc],
        petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
        petsc_options_prefix="poisson_"
    )
    u_sol = problem.solve()
    
    # 6. Output Extraction & Interpolation
    out_grid = case_spec["output"]["grid"]
    nx_out = out_grid["nx"]
    ny_out = out_grid["ny"]
    bbox = out_grid["bbox"]
    
    xs = np.linspace(bbox[0], bbox[1], nx_out)
    ys = np.linspace(bbox[2], bbox[3], ny_out)
    XX, YY = np.meshgrid(xs, ys)
    points = np.vstack((XX.flatten(), YY.flatten(), np.zeros_like(XX.flatten())))
    
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
            
    u_values = np.full((points.shape[1],), np.nan)
    if len(points_on_proc) > 0:
        vals = u_sol.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u_values[eval_map] = vals.flatten()
        
    u_grid = u_values.reshape((ny_out, nx_out))
    
    # 7. Collect Solver Info
    solver_info = {
        "mesh_resolution": nx_mesh,
        "element_degree": degree,
        "ksp_type": "preonly",
        "pc_type": "lu",
        "rtol": 1e-8,
        "iterations": 1
    }
    
    return {
        "u": u_grid,
        "solver_info": solver_info
    }
