import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem.petsc import LinearProblem
import ufl
from petsc4py import PETSc
from basix.ufl import element as basix_element, mixed_element as basix_mixed_element

def solve(case_spec: dict) -> dict:
    nx_out = case_spec["output"]["grid"]["nx"]
    ny_out = case_spec["output"]["grid"]["ny"]
    bbox = case_spec["output"]["grid"]["bbox"]
    
    # Mesh resolution
    mesh_res = 120
    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_square(comm, mesh_res, mesh_res, cell_type=mesh.CellType.triangle)
    gdim = domain.geometry.dim

    # Taylor-Hood P2-P1
    vel_el = basix_element("Lagrange", domain.topology.cell_name(), 2, shape=(gdim,))
    pres_el = basix_element("Lagrange", domain.topology.cell_name(), 1)
    W = fem.functionspace(domain, basix_mixed_element([vel_el, pres_el]))
    V, _ = W.sub(0).collapse()
    Q, _ = W.sub(1).collapse()

    nu = 1.0
    x = ufl.SpatialCoordinate(domain)
    
    # Exact solutions
    pi = ufl.pi
    u_ex_x = pi * ufl.cos(pi*x[1]) * ufl.sin(pi*x[0])
    u_ex_y = -pi * ufl.cos(pi*x[0]) * ufl.sin(pi*x[1])
    
    f_x = 2 * nu * pi**3 * ufl.cos(pi*x[1]) * ufl.sin(pi*x[0]) - pi * ufl.sin(pi*x[0]) * ufl.cos(pi*x[1])
    f_y = -2 * nu * pi**3 * ufl.cos(pi*x[0]) * ufl.sin(pi*x[1]) - pi * ufl.cos(pi*x[0]) * ufl.sin(pi*x[1])
    f = ufl.as_vector((f_x, f_y))

    (u, p) = ufl.TrialFunctions(W)
    (v, q) = ufl.TestFunctions(W)

    a = (2 * nu * ufl.inner(ufl.sym(ufl.grad(u)), ufl.sym(ufl.grad(v))) * ufl.dx
         - p * ufl.div(v) * ufl.dx
         + ufl.div(u) * q * ufl.dx)
    L = ufl.inner(f, v) * ufl.dx

    # Boundary Conditions
    fdim = domain.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool))
    boundary_dofs = fem.locate_dofs_topological((W.sub(0), V), fdim, boundary_facets)
    
    u_bc = fem.Function(V)
    u_exact_expr = ufl.as_vector([u_ex_x, u_ex_y])
    u_bc_expr = fem.Expression(u_exact_expr, V.element.interpolation_points)
    u_bc.interpolate(u_bc_expr)
    bc_u = fem.dirichletbc(u_bc, boundary_dofs, W.sub(0))

    # Pressure pin
    p_dofs = fem.locate_dofs_geometrical((W.sub(1), Q), lambda x: np.isclose(x[0], 0.0) & np.isclose(x[1], 0.0))
    p0 = fem.Function(Q)
    p0.x.array[:] = 1.0
    bcs = [bc_u]
    if len(p_dofs) > 0:
        bc_p = fem.dirichletbc(p0, p_dofs, W.sub(1))
        bcs.append(bc_p)

    problem = LinearProblem(a, L, bcs=bcs,
                            petsc_options={"ksp_type": "preonly", "pc_type": "lu", "pc_factor_mat_solver_type": "mumps"},
                            petsc_options_prefix="stokes_")
    w_h = problem.solve()
    u_sol = w_h.sub(0).collapse()

    # Interpolate onto grid
    xs = np.linspace(bbox[0], bbox[1], nx_out)
    ys = np.linspace(bbox[2], bbox[3], ny_out)
    XX, YY = np.meshgrid(xs, ys)
    pts = np.vstack([XX.ravel(), YY.ravel(), np.zeros_like(XX.ravel())])
    
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
        else:
            # Fallback for boundary points
            pt = pts.T[i].copy()
            pt[0] = np.clip(pt[0], 1e-10, 1.0-1e-10)
            pt[1] = np.clip(pt[1], 1e-10, 1.0-1e-10)
            c_cand = geometry.compute_collisions_points(bb_tree, pt.reshape(1, 3))
            c_coll = geometry.compute_colliding_cells(domain, c_cand, pt.reshape(1, 3))
            links2 = c_coll.links(0)
            if len(links2) > 0:
                points_on_proc.append(pt)
                cells_on_proc.append(links2[0])
                eval_map.append(i)

    u_values = np.zeros((pts.shape[1], 2))
    if len(points_on_proc) > 0:
        vals = u_sol.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        for idx, eval_idx in enumerate(eval_map):
            u_values[eval_idx] = vals[idx][:2]
            
    u_mag = np.linalg.norm(u_values, axis=1).reshape((ny_out, nx_out))

    solver_info = {
        "mesh_resolution": mesh_res,
        "element_degree": 2,
        "ksp_type": "preonly",
        "pc_type": "lu",
        "rtol": 1e-8,
        "iterations": 1
    }

    return {
        "u": u_mag,
        "solver_info": solver_info
    }
