import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem.petsc import LinearProblem
import ufl
from basix.ufl import element as basix_element, mixed_element as basix_mixed_element
from petsc4py import PETSc

import os

def solve(case_spec: dict) -> dict:
    os.environ["OMP_NUM_THREADS"] = "1"
    nx_out = case_spec["output"]["grid"]["nx"]
    ny_out = case_spec["output"]["grid"]["ny"]
    bbox = case_spec["output"]["grid"]["bbox"]

    # Mesh resolution
    N = 96
    comm = MPI.COMM_WORLD
    msh = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
    gdim = msh.geometry.dim

    # Taylor-Hood P2-P1
    vel_el = basix_element("Lagrange", msh.topology.cell_name(), 2, shape=(gdim,))
    pres_el = basix_element("Lagrange", msh.topology.cell_name(), 1)
    W = fem.functionspace(msh, basix_mixed_element([vel_el, pres_el]))
    V, _ = W.sub(0).collapse()
    Q, _ = W.sub(1).collapse()

    nu = 0.1
    x = ufl.SpatialCoordinate(msh)
    
    # Exact solutions
    u_ex = ufl.as_vector([
        ufl.pi * ufl.cos(ufl.pi * x[1]) * ufl.sin(ufl.pi * x[0]),
        -ufl.pi * ufl.cos(ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])
    ])
    p_ex = ufl.cos(ufl.pi * x[0]) * ufl.cos(ufl.pi * x[1])

    # Forcing term
    f_x = (0.2 * ufl.pi**3 - ufl.pi) * ufl.sin(ufl.pi * x[0]) * ufl.cos(ufl.pi * x[1])
    f_y = (-0.2 * ufl.pi**3 - ufl.pi) * ufl.cos(ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])
    f = ufl.as_vector((f_x, f_y))

    (u, p) = ufl.TrialFunctions(W)
    (v, q) = ufl.TestFunctions(W)

    a = (2.0 * nu * ufl.inner(ufl.sym(ufl.grad(u)), ufl.sym(ufl.grad(v))) * ufl.dx
         - p * ufl.div(v) * ufl.dx
         + ufl.div(u) * q * ufl.dx)
    L = ufl.inner(f, v) * ufl.dx

    # BCs
    fdim = msh.topology.dim - 1
    facets = mesh.locate_entities_boundary(msh, fdim, lambda X: np.ones(X.shape[1], dtype=bool))
    dofs_u = fem.locate_dofs_topological((W.sub(0), V), fdim, facets)
    
    u_bc = fem.Function(V)
    u_bc.interpolate(fem.Expression(u_ex, V.element.interpolation_points))
    bc_u = fem.dirichletbc(u_bc, dofs_u, W.sub(0))

    # Pressure pinning
    p_dofs = fem.locate_dofs_geometrical((W.sub(1), Q), lambda x: np.isclose(x[0], 0.0) & np.isclose(x[1], 0.0))
    p_bc = fem.Function(Q)
    p_bc.interpolate(fem.Expression(p_ex, Q.element.interpolation_points))
    bcs = [bc_u]
    if len(p_dofs) > 0:
        bcs.append(fem.dirichletbc(p_bc, p_dofs, W.sub(1)))

    # Solve
    problem = LinearProblem(a, L, bcs=bcs,
                            petsc_options={"ksp_type": "preonly", "pc_type": "lu", "pc_factor_mat_solver_type": "mumps"},
                            petsc_options_prefix="stokes_")
    w_h = problem.solve()
    u_h = w_h.sub(0).collapse()

    # Interpolate to output grid
    xs = np.linspace(bbox[0], bbox[1], nx_out)
    ys = np.linspace(bbox[2], bbox[3], ny_out)
    XX, YY = np.meshgrid(xs, ys)
    pts = np.c_[XX.ravel(), YY.ravel(), np.zeros(nx_out * ny_out)]

    bb_tree = geometry.bb_tree(msh, msh.topology.dim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, pts)
    colliding_cells = geometry.compute_colliding_cells(msh, cell_candidates, pts)

    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    for i in range(len(pts)):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(pts[i])
            cells_on_proc.append(links[0])
            eval_map.append(i)

    u_out = np.full((nx_out * ny_out,), np.nan)
    if len(points_on_proc) > 0:
        u_vals = u_h.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u_mag = np.linalg.norm(u_vals, axis=1)
        u_out[eval_map] = u_mag

    # Replace nan with 0 for simplicity or interpolate (here points are within bbox, so they should be found)
    u_out[np.isnan(u_out)] = 0.0
    u_grid = u_out.reshape((ny_out, nx_out))

    return {
        "u": u_grid,
        "solver_info": {
            "mesh_resolution": N,
            "element_degree": 2,
            "ksp_type": "preonly",
            "pc_type": "lu",
            "rtol": 1e-8,
            "iterations": 1
        }
    }
