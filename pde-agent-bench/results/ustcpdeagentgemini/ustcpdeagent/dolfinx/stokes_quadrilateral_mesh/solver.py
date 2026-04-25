import os
os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem
from dolfinx.fem.petsc import LinearProblem
from basix.ufl import element as basix_element, mixed_element as basix_mixed_element
import ufl
import math

def solve(case_spec: dict) -> dict:
    nx_out = case_spec["output"]["grid"]["nx"]
    ny_out = case_spec["output"]["grid"]["ny"]
    bbox = case_spec["output"]["grid"]["bbox"]
    
    comm = MPI.COMM_WORLD
    N = 100
    msh = mesh.create_rectangle(comm, [np.array([bbox[0], bbox[2]]), np.array([bbox[1], bbox[3]])], [N, N], cell_type=mesh.CellType.quadrilateral)
    gdim = msh.geometry.dim

    vel_el = basix_element("Lagrange", msh.topology.cell_name(), 2, shape=(gdim,))
    pres_el = basix_element("Lagrange", msh.topology.cell_name(), 1)
    W = fem.functionspace(msh, basix_mixed_element([vel_el, pres_el]))
    V, V_to_W = W.sub(0).collapse()
    Q, Q_to_W = W.sub(1).collapse()

    nu = 1.0
    x = ufl.SpatialCoordinate(msh)
    
    u_exact = ufl.as_vector([
        ufl.pi * ufl.cos(ufl.pi * x[1]) * ufl.sin(ufl.pi * x[0]),
        -ufl.pi * ufl.cos(ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])
    ])
    
    fx = ufl.pi * (2*ufl.pi**2 - 1) * ufl.sin(ufl.pi * x[0]) * ufl.cos(ufl.pi * x[1])
    fy = -ufl.pi * (2*ufl.pi**2 + 1) * ufl.cos(ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])
    f = ufl.as_vector([fx, fy])

    (u, p) = ufl.TrialFunctions(W)
    (v, q) = ufl.TestFunctions(W)

    a = (nu * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
         - p * ufl.div(v) * ufl.dx
         + ufl.div(u) * q * ufl.dx)
    L = ufl.inner(f, v) * ufl.dx

    fdim = msh.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(msh, fdim, lambda x_: np.ones(x_.shape[1], dtype=bool))
    boundary_dofs = fem.locate_dofs_topological((W.sub(0), V), fdim, boundary_facets)
    
    u_bc = fem.Function(V)
    u_bc.interpolate(fem.Expression(u_exact, V.element.interpolation_points))
    bc_u = fem.dirichletbc(u_bc, boundary_dofs, W.sub(0))
    
    p_dofs = fem.locate_dofs_geometrical((W.sub(1), Q), lambda x_: np.isclose(x_[0], 0.0) & np.isclose(x_[1], 0.0))
    p0 = fem.Function(Q)
    p0.x.array[:] = 1.0
    bc_p = fem.dirichletbc(p0, p_dofs, W.sub(1))
    
    bcs = [bc_u, bc_p]

    problem = LinearProblem(a, L, bcs=bcs,
                            petsc_options={"ksp_type": "preonly", "pc_type": "lu", "pc_factor_mat_solver_type": "mumps"},
                            petsc_options_prefix="stokes_")
    w_h = problem.solve()
    u_h = w_h.sub(0).collapse()

    xs = np.linspace(bbox[0], bbox[1], nx_out)
    ys = np.linspace(bbox[2], bbox[3], ny_out)
    XX, YY = np.meshgrid(xs, ys)
    pts = np.c_[XX.ravel(), YY.ravel(), np.zeros(nx_out * ny_out)]

    from dolfinx.geometry import bb_tree, compute_collisions_points, compute_colliding_cells
    tree = bb_tree(msh, msh.topology.dim)
    cell_candidates = compute_collisions_points(tree, pts)
    colliding = compute_colliding_cells(msh, cell_candidates, pts)
    
    points_on_proc = []
    cells = []
    eval_map = []
    for i, pt in enumerate(pts):
        links = colliding.links(i)
        if len(links) > 0:
            points_on_proc.append(pt)
            cells.append(links[0])
            eval_map.append(i)

    u_vals = np.zeros((nx_out * ny_out, gdim))
    if len(points_on_proc) > 0:
        vals = u_h.eval(np.array(points_on_proc), np.array(cells, dtype=np.int32))
        u_vals[eval_map] = vals

    magnitude = np.linalg.norm(u_vals, axis=1).reshape(ny_out, nx_out)

    return {
        "u": magnitude,
        "solver_info": {
            "mesh_resolution": N,
            "element_degree": 2,
            "ksp_type": "preonly",
            "pc_type": "lu",
            "rtol": 1e-8,
            "iterations": 1
        }
    }

if __name__ == "__main__":
    case = {
        "output": {
            "grid": {
                "nx": 50,
                "ny": 50,
                "bbox": [0.0, 1.0, 0.0, 1.0]
            }
        }
    }
    sol = solve(case)
    print("Shape:", sol["u"].shape)
    print("Max magnitude:", np.max(sol["u"]))
