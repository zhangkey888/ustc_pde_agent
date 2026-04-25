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
    nx_grid = case_spec["output"]["grid"]["nx"]
    ny_grid = case_spec["output"]["grid"]["ny"]
    bbox = case_spec["output"]["grid"]["bbox"]

    mesh_resolution = 64
    degree_u = 3
    degree_p = 2
    nu = 1.0

    comm = MPI.COMM_WORLD
    msh = mesh.create_unit_square(comm, mesh_resolution, mesh_resolution, cell_type=mesh.CellType.triangle)
    gdim = msh.geometry.dim

    vel_el = basix_element("Lagrange", msh.topology.cell_name(), degree_u, shape=(gdim,))
    pres_el = basix_element("Lagrange", msh.topology.cell_name(), degree_p)
    W = fem.functionspace(msh, basix_mixed_element([vel_el, pres_el]))
    
    V, _ = W.sub(0).collapse()
    Q, _ = W.sub(1).collapse()

    x = ufl.SpatialCoordinate(msh)
    u_exact = ufl.as_vector([
        ufl.pi * ufl.cos(ufl.pi * x[1]) * ufl.sin(ufl.pi * x[0]),
        -ufl.pi * ufl.cos(ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])
    ])
    p_exact = ufl.cos(ufl.pi * x[0]) * ufl.cos(ufl.pi * x[1])
    
    f = -nu * ufl.div(ufl.grad(u_exact)) + ufl.grad(p_exact)

    (u, p) = ufl.TrialFunctions(W)
    (v, q) = ufl.TestFunctions(W)

    a = (nu * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
         - p * ufl.div(v) * ufl.dx
         + ufl.div(u) * q * ufl.dx)
    L = ufl.inner(f, v) * ufl.dx

    fdim = msh.topology.dim - 1
    wall_facets = mesh.locate_entities_boundary(msh, fdim, lambda x_c: np.ones(x_c.shape[1], dtype=bool))
    
    u_bc = fem.Function(V)
    u_bc_expr = fem.Expression(u_exact, V.element.interpolation_points)
    u_bc.interpolate(u_bc_expr)
    bc_u = fem.dirichletbc(u_bc, fem.locate_dofs_topological((W.sub(0), V), fdim, wall_facets), W.sub(0))

    p_dofs = fem.locate_dofs_geometrical((W.sub(1), Q), lambda x_c: np.isclose(x_c[0], 0.0) & np.isclose(x_c[1], 0.0))
    p0 = fem.Function(Q)
    p0.x.array[:] = 1.0  # Since p_exact(0,0) = 1.0
    bc_p = fem.dirichletbc(p0, p_dofs, W.sub(1))

    problem = LinearProblem(a, L, bcs=[bc_u, bc_p],
                            petsc_options={"ksp_type": "preonly", "pc_type": "lu", "pc_factor_mat_solver_type": "mumps"},
                            petsc_options_prefix="stokes_")
    w_h = problem.solve()
    u_h = w_h.sub(0).collapse()

    xs = np.linspace(bbox[0], bbox[1], nx_grid)
    ys = np.linspace(bbox[2], bbox[3], ny_grid)
    XX, YY = np.meshgrid(xs, ys)
    pts = np.c_[XX.ravel(), YY.ravel(), np.zeros(nx_grid * ny_grid)]

    tree = geometry.bb_tree(msh, msh.topology.dim)
    cell_candidates = geometry.compute_collisions_points(tree, pts)
    colliding_cells = geometry.compute_colliding_cells(msh, cell_candidates, pts)

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
        vals = u_h.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u_values[eval_map] = vals

    magnitude = np.linalg.norm(u_values, axis=1).reshape((ny_grid, nx_grid))

    solver_info = {
        "mesh_resolution": mesh_resolution,
        "element_degree": degree_u,
        "ksp_type": "preonly",
        "pc_type": "lu",
        "rtol": 1e-8,
        "iterations": 1
    }

    return {
        "u": magnitude,
        "solver_info": solver_info
    }

if __name__ == "__main__":
    spec = {
        "output": {
            "grid": {
                "nx": 50, "ny": 50, "bbox": [0, 1, 0, 1]
            }
        }
    }
    res = solve(spec)
    print("Shape:", res["u"].shape)
    print("Max magnitude:", np.max(res["u"]))
