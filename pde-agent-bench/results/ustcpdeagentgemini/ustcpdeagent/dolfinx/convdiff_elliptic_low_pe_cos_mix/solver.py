import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc

def solve(case_spec: dict) -> dict:
    nx = case_spec["output"]["grid"]["nx"]
    ny = case_spec["output"]["grid"]["ny"]
    bbox = case_spec["output"]["grid"]["bbox"]
    
    # Problem Parameters
    epsilon = 0.2
    beta_vec = [0.8, 0.3]
    
    # Mesh
    mesh_res = 64
    degree = 2
    comm = MPI.COMM_WORLD
    domain = mesh.create_rectangle(comm, [[0.0, 0.0], [1.0, 1.0]], [mesh_res, mesh_res], cell_type=mesh.CellType.triangle)
    
    V = fem.functionspace(domain, ("Lagrange", degree))
    
    # Exact solution for BC and f
    x = ufl.SpatialCoordinate(domain)
    u_exact = ufl.cos(ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])
    
    # Forcing term
    # -eps * Laplacian(u) + beta \cdot grad(u)
    # Laplacian(u) = -2 pi^2 u_exact
    f_expr = -epsilon * (-2.0 * ufl.pi**2 * u_exact) \
             + beta_vec[0] * (-ufl.pi * ufl.sin(ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])) \
             + beta_vec[1] * (ufl.pi * ufl.cos(ufl.pi * x[0]) * ufl.cos(ufl.pi * x[1]))
    
    # Boundary Conditions
    u_bc = fem.Function(V)
    u_bc.interpolate(fem.Expression(u_exact, V.element.interpolation_points()))
    
    tdim = domain.topology.dim
    fdim = tdim - 1
    domain.topology.create_connectivity(fdim, tdim)
    boundary_facets = mesh.exterior_facet_indices(domain.topology)
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc = fem.dirichletbc(u_bc, boundary_dofs)
    
    # Variational problem
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    beta = fem.Constant(domain, PETSc.ScalarType(beta_vec))
    eps_const = fem.Constant(domain, PETSc.ScalarType(epsilon))
    
    a = eps_const * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx + ufl.inner(ufl.dot(beta, ufl.grad(u)), v) * ufl.dx
    L = ufl.inner(f_expr, v) * ufl.dx
    
    problem = petsc.LinearProblem(a, L, bcs=[bc], petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
    u_sol = problem.solve()
    
    # Interpolate to output grid
    xs = np.linspace(bbox[0], bbox[1], nx)
    ys = np.linspace(bbox[2], bbox[3], ny)
    XX, YY = np.meshgrid(xs, ys)
    pts = np.c_[XX.ravel(), YY.ravel(), np.zeros(nx * ny)]
    
    bb_tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, pts.T)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, pts.T)
    
    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    for i in range(pts.shape[0]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(pts[i])
            cells_on_proc.append(links[0])
            eval_map.append(i)
            
    u_values = np.full(pts.shape[0], np.nan)
    if len(points_on_proc) > 0:
        vals = u_sol.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u_values[eval_map] = vals.flatten()
        
    u_grid = u_values.reshape((ny, nx))
    
    return {
        "u": u_grid,
        "solver_info": {
            "mesh_resolution": mesh_res,
            "element_degree": degree,
            "ksp_type": "preonly",
            "pc_type": "lu",
            "rtol": 1e-8,
            "iterations": 1
        }
    }
