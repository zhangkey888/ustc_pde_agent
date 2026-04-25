import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from basix.ufl import element, mixed_element
from petsc4py import PETSc
import time

def solve(case_spec: dict) -> dict:
    
    # Mesh resolution
    nx = 128
    ny = 128
    
    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_square(comm, nx, ny, cell_type=mesh.CellType.triangle)
    gdim = domain.geometry.dim
    
    # Elements: Taylor-Hood P2-P1
    P2 = element("Lagrange", domain.topology.cell_name(), 2, shape=(gdim,))
    P1 = element("Lagrange", domain.topology.cell_name(), 1)
    W_elem = mixed_element([P2, P1])
    W = fem.functionspace(domain, W_elem)
    
    V, _ = W.sub(0).collapse()
    Q, _ = W.sub(1).collapse()
    
    (u, p) = ufl.TrialFunctions(W)
    (v, q) = ufl.TestFunctions(W)
    
    nu = 0.05
    f = fem.Constant(domain, PETSc.ScalarType((0.0, 0.0)))
    
    a = (nu * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
         - p * ufl.div(v) * ufl.dx
         + ufl.div(u) * q * ufl.dx)
    L = ufl.inner(f, v) * ufl.dx
    
    fdim = domain.topology.dim - 1
    
    bcs = []
    
    # x = 0: u = [4*y*(1-y), 0]
    facets_x0 = mesh.locate_entities_boundary(domain, fdim, lambda x: np.isclose(x[0], 0.0))
    dofs_x0 = fem.locate_dofs_topological((W.sub(0), V), fdim, facets_x0)
    u_inlet = fem.Function(V)
    u_inlet.interpolate(lambda x: np.vstack((4 * x[1] * (1 - x[1]), np.zeros_like(x[1]))))
    bcs.append(fem.dirichletbc(u_inlet, dofs_x0, W.sub(0)))
    
    # y = 0: u = [0, 0]
    facets_y0 = mesh.locate_entities_boundary(domain, fdim, lambda x: np.isclose(x[1], 0.0))
    dofs_y0 = fem.locate_dofs_topological((W.sub(0), V), fdim, facets_y0)
    u_wall_bottom = fem.Function(V)
    u_wall_bottom.x.array[:] = 0.0
    bcs.append(fem.dirichletbc(u_wall_bottom, dofs_y0, W.sub(0)))
    
    # y = 1: u = [0, 0]
    facets_y1 = mesh.locate_entities_boundary(domain, fdim, lambda x: np.isclose(x[1], 1.0))
    dofs_y1 = fem.locate_dofs_topological((W.sub(0), V), fdim, facets_y1)
    u_wall_top = fem.Function(V)
    u_wall_top.x.array[:] = 0.0
    bcs.append(fem.dirichletbc(u_wall_top, dofs_y1, W.sub(0)))
    
    # Solve
    problem = petsc.LinearProblem(a, L, bcs=bcs,
                                  petsc_options={"ksp_type": "preonly", "pc_type": "lu", "pc_factor_mat_solver_type": "mumps"},
                                  petsc_options_prefix="stokes_")
    w_h = problem.solve()
    u_h = w_h.sub(0).collapse()
    
    # Sample output
    grid_spec = case_spec["output"]["grid"]
    nx_out = grid_spec["nx"]
    ny_out = grid_spec["ny"]
    bbox = grid_spec["bbox"]
    
    xs = np.linspace(bbox[0], bbox[1], nx_out)
    ys = np.linspace(bbox[2], bbox[3], ny_out)
    XX, YY = np.meshgrid(xs, ys)
    pts = np.c_[XX.ravel(), YY.ravel(), np.zeros_like(XX.ravel())]
    
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
        u_values[eval_map] = vals
        
    u_mag = np.linalg.norm(u_values, axis=1).reshape(ny_out, nx_out)
    
    return {
        "u": u_mag,
        "solver_info": {
            "mesh_resolution": nx,
            "element_degree": 2,
            "ksp_type": "preonly",
            "pc_type": "lu",
            "rtol": 1e-8,
            "iterations": 1
        }
    }
