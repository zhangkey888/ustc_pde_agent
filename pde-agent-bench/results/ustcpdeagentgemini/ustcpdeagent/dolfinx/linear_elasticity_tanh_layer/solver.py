import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc
import time

def solve(case_spec: dict) -> dict:
    start_time = time.time()
    
    # Parameters
    nx = case_spec["output"]["grid"]["nx"]
    ny = case_spec["output"]["grid"]["ny"]
    bbox = case_spec["output"]["grid"]["bbox"]
    
    E = 1.0
    nu = 0.3
    mu = E / (2.0 * (1.0 + nu))
    lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    
    mesh_res = 128
    degree = 2
    
    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_square(comm, mesh_res, mesh_res, cell_type=mesh.CellType.triangle)
    gdim = domain.geometry.dim
    
    V = fem.functionspace(domain, ("Lagrange", degree, (gdim,)))
    
    x = ufl.SpatialCoordinate(domain)
    u_exact_expr = ufl.as_vector([
        ufl.tanh(6*(x[1]-0.5))*ufl.sin(ufl.pi*x[0]),
        0.1*ufl.sin(2*ufl.pi*x[0])*ufl.sin(ufl.pi*x[1])
    ])
    
    def eps(u):
        return ufl.sym(ufl.grad(u))
        
    def sigma(u):
        return 2.0 * mu * eps(u) + lam * ufl.tr(eps(u)) * ufl.Identity(gdim)
        
    f_expr = -ufl.div(sigma(u_exact_expr))
    
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    
    a = ufl.inner(sigma(u), eps(v)) * ufl.dx
    L = ufl.inner(f_expr, v) * ufl.dx
    
    fdim = domain.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
    )
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    
    u_bc = fem.Function(V)
    expr = fem.Expression(u_exact_expr, V.element.interpolation_points())
    u_bc.interpolate(expr)
    bc = fem.dirichletbc(u_bc, boundary_dofs)
    
    problem = petsc.LinearProblem(
        a, L, bcs=[bc],
        petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
        petsc_options_prefix="elasticity_"
    )
    u_sol = problem.solve()
    
    # Evaluate on grid
    xs = np.linspace(bbox[0], bbox[1], nx)
    ys = np.linspace(bbox[2], bbox[3], ny)
    XX, YY = np.meshgrid(xs, ys)
    pts = np.c_[XX.ravel(), YY.ravel(), np.zeros(nx * ny)]
    
    bb_tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, pts)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, pts)
    
    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    for i in range(pts.shape[0]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(pts[i])
            cells_on_proc.append(links[0])
            eval_map.append(i)
            
    u_values = np.zeros((pts.shape[0], gdim))
    if len(points_on_proc) > 0:
        vals = u_sol.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u_values[eval_map] = vals
        
    magnitude = np.linalg.norm(u_values, axis=1).reshape((ny, nx))
    
    solver_info = {
        "mesh_resolution": mesh_res,
        "element_degree": degree,
        "ksp_type": "preonly",
        "pc_type": "lu",
        "rtol": 0.0,
        "iterations": 1
    }
    
    return {"u": magnitude, "solver_info": solver_info}

