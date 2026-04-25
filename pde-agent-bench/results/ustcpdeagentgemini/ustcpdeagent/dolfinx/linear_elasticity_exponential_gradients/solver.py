import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc
import time

def solve(case_spec: dict) -> dict:
    start_time = time.time()
    
    # Grid info
    grid_spec = case_spec["output"]["grid"]
    nx_out = grid_spec["nx"]
    ny_out = grid_spec["ny"]
    bbox = grid_spec["bbox"]
    
    # Solver parameters
    resolution = 128
    degree = 2
    
    # Mesh
    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_square(comm, resolution, resolution, cell_type=mesh.CellType.triangle)
    gdim = domain.geometry.dim
    
    # Function Space
    V = fem.functionspace(domain, ("Lagrange", degree, (gdim,)))
    
    # Material
    E = 1.0
    nu = 0.33
    mu = E / (2.0 * (1.0 + nu))
    lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    
    def eps(u):
        return ufl.sym(ufl.grad(u))
        
    def sigma(u):
        return 2.0 * mu * eps(u) + lam * ufl.tr(eps(u)) * ufl.Identity(gdim)
        
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    
    # Exact solution
    x = ufl.SpatialCoordinate(domain)
    u_ex = ufl.as_vector([
        ufl.exp(2*x[0]) * ufl.sin(ufl.pi*x[1]),
        -ufl.exp(2*x[1]) * ufl.sin(ufl.pi*x[0])
    ])
    
    # Weak form
    a = ufl.inner(sigma(u), eps(v)) * ufl.dx
    L = ufl.inner(sigma(u_ex), eps(v)) * ufl.dx
    
    # Boundary Conditions
    fdim = domain.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool))
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    
    u_bc = fem.Function(V)
    expr = fem.Expression(u_ex, V.element.interpolation_points())
    u_bc.interpolate(expr)
    
    bc = fem.dirichletbc(u_bc, boundary_dofs)
    
    # Solve
    problem = petsc.LinearProblem(
        a, L, bcs=[bc],
        petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
        petsc_options_prefix="elasticity_"
    )
    u_sol = problem.solve()
    
    # Sampling
    xs = np.linspace(bbox[0], bbox[1], nx_out)
    ys = np.linspace(bbox[2], bbox[3], ny_out)
    XX, YY = np.meshgrid(xs, ys)
    pts = np.c_[XX.ravel(), YY.ravel(), np.zeros(nx_out * ny_out)]
    
    tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(tree, pts)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, pts)
    
    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    for i in range(len(pts)):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(pts[i])
            cells_on_proc.append(links[0])
            eval_map.append(i)
            
    u_values = np.full((len(pts), gdim), np.nan)
    if len(points_on_proc) > 0:
        vals = u_sol.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u_values[eval_map] = vals
        
    magnitude = np.linalg.norm(u_values, axis=1).reshape(ny_out, nx_out)
    
    solver_info = {
        "mesh_resolution": resolution,
        "element_degree": degree,
        "ksp_type": "preonly",
        "pc_type": "lu",
        "rtol": 0.0,
        "iterations": 1
    }
    
    return {
        "u": magnitude,
        "solver_info": solver_info
    }
