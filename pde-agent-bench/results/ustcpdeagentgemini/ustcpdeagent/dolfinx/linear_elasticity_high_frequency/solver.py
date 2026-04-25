import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
import dolfinx.fem.petsc as petsc
import ufl
from petsc4py import PETSc

def solve(case_spec: dict) -> dict:
    nx = case_spec["output"]["grid"]["nx"]
    ny = case_spec["output"]["grid"]["ny"]
    bbox = case_spec["output"]["grid"]["bbox"]
    
    mesh_res = 128
    degree = 2
    
    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_square(comm, mesh_res, mesh_res, cell_type=mesh.CellType.triangle)
    gdim = domain.geometry.dim
    
    V = fem.functionspace(domain, ("Lagrange", degree, (gdim,)))
    
    E = 1.0
    nu = 0.28
    mu = E / (2.0 * (1.0 + nu))
    lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    
    def eps(u):
        return ufl.sym(ufl.grad(u))
        
    def sigma(u):
        return 2.0 * mu * eps(u) + lam * ufl.tr(eps(u)) * ufl.Identity(gdim)
        
    x = ufl.SpatialCoordinate(domain)
    
    # Exact solution
    u_exact = ufl.as_vector([
        ufl.sin(4.0 * ufl.pi * x[0]) * ufl.sin(3.0 * ufl.pi * x[1]),
        ufl.cos(3.0 * ufl.pi * x[0]) * ufl.sin(4.0 * ufl.pi * x[1])
    ])
    
    f = -ufl.div(sigma(u_exact))
    
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    
    a = ufl.inner(sigma(u), eps(v)) * ufl.dx
    L = ufl.inner(f, v) * ufl.dx
    
    fdim = domain.topology.dim - 1
    
    def boundary_marker(x_pts):
        return np.logical_or.reduce([
            np.isclose(x_pts[0], 0.0),
            np.isclose(x_pts[0], 1.0),
            np.isclose(x_pts[1], 0.0),
            np.isclose(x_pts[1], 1.0)
        ])
        
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, boundary_marker
    )
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    
    u_bc = fem.Function(V)
    u_exact_expr = fem.Expression(u_exact, V.element.interpolation_points)
    u_bc.interpolate(u_exact_expr)
    
    bc = fem.dirichletbc(u_bc, boundary_dofs)
    
    problem = petsc.LinearProblem(
        a, L, bcs=[bc],
        petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
        petsc_options_prefix="elast_"
    )
    u_sol = problem.solve()
    
    # Sampling
    xs = np.linspace(bbox[0], bbox[1], nx)
    ys = np.linspace(bbox[2], bbox[3], ny)
    XX, YY = np.meshgrid(xs, ys)
    pts = np.c_[XX.ravel(), YY.ravel(), np.zeros(nx * ny)]
    
    tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(tree, pts)
    colliding = geometry.compute_colliding_cells(domain, cell_candidates, pts)
    
    points_on_proc = []
    cells = []
    eval_map = []
    for i, pt in enumerate(pts):
        links = colliding.links(i)
        if len(links) > 0:
            points_on_proc.append(pt)
            cells.append(links[0])
            eval_map.append(i)
            
    u_values = np.zeros((nx * ny, gdim))
    if len(points_on_proc) > 0:
        vals = u_sol.eval(np.array(points_on_proc), np.array(cells, dtype=np.int32))
        for idx, eval_idx in enumerate(eval_map):
            u_values[eval_idx] = vals[idx]
        
    magnitude = np.linalg.norm(u_values, axis=1).reshape(ny, nx)
    
    solver_info = {
        "mesh_resolution": mesh_res,
        "element_degree": degree,
        "ksp_type": "preonly",
        "pc_type": "lu",
        "rtol": 1e-8,
        "iterations": 1
    }
    
    return {
        "u": magnitude,
        "solver_info": solver_info
    }
