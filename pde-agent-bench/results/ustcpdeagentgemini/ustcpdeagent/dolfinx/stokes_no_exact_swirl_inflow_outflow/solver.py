import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from basix.ufl import element as basix_element, mixed_element as basix_mixed_element
from petsc4py import PETSc
import time

def solve(case_spec: dict) -> dict:
    start_time = time.time()
    
    # Extract grid info
    grid_spec = case_spec["output"]["grid"]
    nx_out = grid_spec["nx"]
    ny_out = grid_spec["ny"]
    bbox = grid_spec["bbox"]
    
    # Domain parameter
    nu = 0.5
    
    # Mesh resolution
    mesh_res = 64
    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_square(comm, mesh_res, mesh_res, cell_type=mesh.CellType.triangle)
    gdim = domain.geometry.dim

    # Function space
    vel_el = basix_element("Lagrange", domain.topology.cell_name(), 2, shape=(gdim,))
    pres_el = basix_element("Lagrange", domain.topology.cell_name(), 1)
    W = fem.functionspace(domain, basix_mixed_element([vel_el, pres_el]))
    
    V, _ = W.sub(0).collapse()
    Q, _ = W.sub(1).collapse()

    # Variational form
    (u, p) = ufl.TrialFunctions(W)
    (v, q) = ufl.TestFunctions(W)
    
    f = fem.Constant(domain, PETSc.ScalarType((0.0, 0.0)))
    
    a = (nu * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
         - p * ufl.div(v) * ufl.dx
         + q * ufl.div(u) * ufl.dx)
    L = ufl.inner(f, v) * ufl.dx

    # Boundary conditions
    fdim = domain.topology.dim - 1
    
    # x0
    facets_x0 = mesh.locate_entities_boundary(domain, fdim, lambda x: np.isclose(x[0], 0.0))
    dofs_x0 = fem.locate_dofs_topological((W.sub(0), V), fdim, facets_x0)
    u_x0 = fem.Function(V)
    u_x0.interpolate(lambda x: np.vstack((np.sin(np.pi * x[1]), 0.2 * np.sin(2 * np.pi * x[1]))))
    bc_x0 = fem.dirichletbc(u_x0, dofs_x0, W.sub(0))
    
    # y0
    facets_y0 = mesh.locate_entities_boundary(domain, fdim, lambda x: np.isclose(x[1], 0.0))
    dofs_y0 = fem.locate_dofs_topological((W.sub(0), V), fdim, facets_y0)
    u_zero = fem.Function(V)
    u_zero.x.array[:] = 0.0
    bc_y0 = fem.dirichletbc(u_zero, dofs_y0, W.sub(0))
    
    # y1
    facets_y1 = mesh.locate_entities_boundary(domain, fdim, lambda x: np.isclose(x[1], 1.0))
    dofs_y1 = fem.locate_dofs_topological((W.sub(0), V), fdim, facets_y1)
    bc_y1 = fem.dirichletbc(u_zero, dofs_y1, W.sub(0))
    
    bcs = [bc_x0, bc_y0, bc_y1]

    # Linear problem
    problem = petsc.LinearProblem(
        a, L, bcs=bcs,
        petsc_options={"ksp_type": "preonly", "pc_type": "lu", "pc_factor_mat_solver_type": "mumps"},
        petsc_options_prefix="stokes_"
    )
    
    w_h = problem.solve()
    u_h, p_h = w_h.sub(0).collapse(), w_h.sub(1).collapse()

    # Interpolate to output grid
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
    for i, pt in enumerate(pts):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(pt)
            cells_on_proc.append(links[0])
            eval_map.append(i)
            
    u_vals = np.zeros((nx_out * ny_out, 2))
    if len(points_on_proc) > 0:
        u_vals_proc = u_h.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u_vals[eval_map] = u_vals_proc
        
    u_mag = np.linalg.norm(u_vals, axis=1).reshape((ny_out, nx_out))
    
    solver_info = {
        "mesh_resolution": mesh_res,
        "element_degree": 2,
        "ksp_type": "preonly",
        "pc_type": "lu",
        "rtol": 1e-8,
        "iterations": 1,
    }
    
    return {
        "u": u_mag,
        "solver_info": solver_info
    }
