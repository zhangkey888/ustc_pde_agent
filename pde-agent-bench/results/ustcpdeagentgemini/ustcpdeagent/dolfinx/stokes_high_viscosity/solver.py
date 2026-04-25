import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import mesh, fem, geometry
from dolfinx.fem.petsc import LinearProblem
from basix.ufl import element, mixed_element
import ufl
import time

def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD
    
    grid_spec = case_spec["output"]["grid"]
    nx_out = grid_spec["nx"]
    ny_out = grid_spec["ny"]
    bbox = grid_spec["bbox"]
    
    nx = 64
    ny = 64
    
    domain = mesh.create_rectangle(comm, [[0.0, 0.0], [1.0, 1.0]], [nx, ny], cell_type=mesh.CellType.triangle)
    gdim = domain.geometry.dim
    
    vel_el = element("Lagrange", domain.topology.cell_name(), 2, shape=(gdim,))
    pres_el = element("Lagrange", domain.topology.cell_name(), 1)
    W = fem.functionspace(domain, mixed_element([vel_el, pres_el]))
    
    V, _ = W.sub(0).collapse()
    Q, _ = W.sub(1).collapse()
    
    (u, p) = ufl.TrialFunctions(W)
    (v, q) = ufl.TestFunctions(W)
    
    nu = 5.0
    x = ufl.SpatialCoordinate(domain)
    
    # Exact solutions
    u_exact = ufl.as_vector([
        ufl.pi * ufl.cos(ufl.pi * x[1]) * ufl.sin(ufl.pi * x[0]),
        -ufl.pi * ufl.cos(ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])
    ])
    p_exact = ufl.cos(ufl.pi * x[0]) * ufl.cos(ufl.pi * x[1])
    
    f = -nu * ufl.div(ufl.grad(u_exact)) + ufl.grad(p_exact)
    
    a = (nu * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
         - p * ufl.div(v) * ufl.dx
         + q * ufl.div(u) * ufl.dx)
    L = ufl.inner(f, v) * ufl.dx
    
    fdim = domain.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool))
    boundary_dofs_u = fem.locate_dofs_topological((W.sub(0), V), fdim, boundary_facets)
    
    u_bc = fem.Function(V)
    expr_u = fem.Expression(u_exact, V.element.interpolation_points)
    u_bc.interpolate(expr_u)
    bc_u = fem.dirichletbc(u_bc, boundary_dofs_u, W.sub(0))
    
    p_dofs = fem.locate_dofs_geometrical((W.sub(1), Q), lambda x: np.isclose(x[0], 0.0) & np.isclose(x[1], 0.0))
    p_bc_val = fem.Function(Q)
    p_bc_val.x.array[:] = 1.0
    bc_p = fem.dirichletbc(p_bc_val, p_dofs, W.sub(1))
    
    bcs = [bc_u, bc_p]
    
    problem = LinearProblem(a, L, bcs=bcs,
                            petsc_options={"ksp_type": "preonly", "pc_type": "lu", "pc_factor_mat_solver_type": "mumps"},
                            petsc_options_prefix="stokes_")
    w_h = problem.solve()
    u_h = w_h.sub(0).collapse()
    
    # Sample on grid
    xs = np.linspace(bbox[0], bbox[1], nx_out)
    ys = np.linspace(bbox[2], bbox[3], ny_out)
    XX, YY = np.meshgrid(xs, ys)
    pts = np.c_[XX.ravel(), YY.ravel(), np.zeros(nx_out * ny_out)]
    
    bb_tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, pts)
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
            
    u_values = np.zeros((len(pts), gdim))
    if len(points_on_proc) > 0:
        vals = u_h.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        for idx, val in zip(eval_map, vals):
            u_values[idx] = val[:gdim]
            
    magnitude = np.linalg.norm(u_values, axis=1).reshape(ny_out, nx_out)
    
    solver_info = {
        "mesh_resolution": nx,
        "element_degree": 2,
        "ksp_type": "preonly",
        "pc_type": "lu",
        "rtol": 1e-8,
        "iterations": 1
    }
    
    return {
        "u": magnitude,
        "solver_info": solver_info
    }

