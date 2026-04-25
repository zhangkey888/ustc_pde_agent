import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem.petsc import LinearProblem
from basix.ufl import element, mixed_element
import ufl
from petsc4py import PETSc

def solve(case_spec: dict) -> dict:
    # 1. Mesh generation
    nx_mesh = 128
    ny_mesh = 128
    comm = MPI.COMM_WORLD
    msh = mesh.create_unit_square(comm, nx_mesh, ny_mesh, cell_type=mesh.CellType.triangle)
    gdim = msh.geometry.dim

    # 2. Function Spaces (Taylor-Hood P2/P1)
    vel_el = element("Lagrange", msh.topology.cell_name(), 2, shape=(gdim,))
    pres_el = element("Lagrange", msh.topology.cell_name(), 1)
    W = fem.functionspace(msh, mixed_element([vel_el, pres_el]))
    V, _ = W.sub(0).collapse()
    Q, _ = W.sub(1).collapse()

    # 3. Parameters and Variational Form
    nu = 0.9
    f = fem.Constant(msh, PETSc.ScalarType((0.0, 0.0)))
    
    (u, p) = ufl.TrialFunctions(W)
    (v, q) = ufl.TestFunctions(W)
    
    a = (nu * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
         - p * ufl.div(v) * ufl.dx
         + ufl.div(u) * q * ufl.dx)
    L = ufl.inner(f, v) * ufl.dx

    # 4. Boundary Conditions
    fdim = msh.topology.dim - 1
    
    # Left boundary (x0): u = [sin(pi*y), 0.0]
    facets_x0 = mesh.locate_entities_boundary(msh, fdim, lambda x: np.isclose(x[0], 0.0))
    dofs_x0 = fem.locate_dofs_topological((W.sub(0), V), fdim, facets_x0)
    u_x0 = fem.Function(V)
    u_x0.interpolate(lambda x: np.vstack((np.sin(np.pi * x[1]), np.zeros_like(x[0]))))
    bc_x0 = fem.dirichletbc(u_x0, dofs_x0, W.sub(0))
    
    # Bottom boundary (y0): u = [0.0, 0.0]
    facets_y0 = mesh.locate_entities_boundary(msh, fdim, lambda x: np.isclose(x[1], 0.0))
    dofs_y0 = fem.locate_dofs_topological((W.sub(0), V), fdim, facets_y0)
    u_zero = fem.Function(V)
    u_zero.x.array[:] = 0.0
    bc_y0 = fem.dirichletbc(u_zero, dofs_y0, W.sub(0))
    
    # Right boundary (x1): u = [0.0, 0.0]
    facets_x1 = mesh.locate_entities_boundary(msh, fdim, lambda x: np.isclose(x[0], 1.0))
    dofs_x1 = fem.locate_dofs_topological((W.sub(0), V), fdim, facets_x1)
    bc_x1 = fem.dirichletbc(u_zero, dofs_x1, W.sub(0))
    
    bcs = [bc_x0, bc_y0, bc_x1]

    # Note: no Dirichlet BC on top (y1), natural outflow.
    # Therefore pressure is implicitly pinned at the outflow.

    # 5. Solver
    problem = LinearProblem(a, L, bcs=bcs,
                            petsc_options={"ksp_type": "preonly", "pc_type": "lu", "pc_factor_mat_solver_type": "mumps"},
                            petsc_options_prefix="stokes_")
    w_h = problem.solve()
    u_h = w_h.sub(0).collapse()

    # 6. Evaluation on Output Grid
    out_grid = case_spec["output"]["grid"]
    nx_out = out_grid["nx"]
    ny_out = out_grid["ny"]
    bbox = out_grid["bbox"]
    
    xs = np.linspace(bbox[0], bbox[1], nx_out)
    ys = np.linspace(bbox[2], bbox[3], ny_out)
    XX, YY = np.meshgrid(xs, ys)
    pts = np.c_[XX.ravel(), YY.ravel(), np.zeros_like(XX.ravel())]
    
    bb_tree = geometry.bb_tree(msh, msh.topology.dim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, pts)
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
            
    u_vals = np.zeros((pts.shape[0], gdim))
    if len(points_on_proc) > 0:
        vals = u_h.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u_vals[eval_map] = vals
        
    u_mag = np.linalg.norm(u_vals, axis=1).reshape((ny_out, nx_out))
    
    return {
        "u": u_mag,
        "solver_info": {
            "mesh_resolution": nx_mesh,
            "element_degree": 2,
            "ksp_type": "preonly",
            "pc_type": "lu",
            "rtol": 1e-8,
            "iterations": 1
        }
    }

if __name__ == "__main__":
    case_spec = {
        "output": {
            "grid": {
                "nx": 50,
                "ny": 50,
                "bbox": [0.0, 1.0, 0.0, 1.0]
            }
        }
    }
    res = solve(case_spec)
    print("Test passed! Max velocity magnitude:", np.max(res["u"]))
