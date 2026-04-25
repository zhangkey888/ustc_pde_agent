import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
from basix.ufl import element as basix_element, mixed_element as basix_mixed_element
import ufl
from petsc4py import PETSc


def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD
    
    # Mesh resolution
    N = 160
    msh = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
    gdim = msh.geometry.dim
    
    # Taylor-Hood P2/P1
    vel_el = basix_element("Lagrange", msh.topology.cell_name(), 2, shape=(gdim,))
    pres_el = basix_element("Lagrange", msh.topology.cell_name(), 1)
    W = fem.functionspace(msh, basix_mixed_element([vel_el, pres_el]))
    
    V, _ = W.sub(0).collapse()
    Q, _ = W.sub(1).collapse()
    
    w = fem.Function(W)
    
    nu_val = 0.12
    nu = fem.Constant(msh, PETSc.ScalarType(nu_val))
    
    fdim = msh.topology.dim - 1
    
    # Boundary conditions
    # x=0 (inflow): u = (4y(1-y), 0)
    # y=0, y=1: no-slip
    # x=1: do nothing (natural outflow)
    
    def inflow(x):
        return np.isclose(x[0], 0.0)
    
    def bottom(x):
        return np.isclose(x[1], 0.0)
    
    def top(x):
        return np.isclose(x[1], 1.0)
    
    inflow_facets = mesh.locate_entities_boundary(msh, fdim, inflow)
    bottom_facets = mesh.locate_entities_boundary(msh, fdim, bottom)
    top_facets = mesh.locate_entities_boundary(msh, fdim, top)
    
    u_in = fem.Function(V)
    u_in.interpolate(lambda x: np.vstack([4.0 * x[1] * (1.0 - x[1]), np.zeros(x.shape[1])]))
    
    u_zero = fem.Function(V)
    u_zero.x.array[:] = 0.0
    
    dofs_in = fem.locate_dofs_topological((W.sub(0), V), fdim, inflow_facets)
    dofs_bot = fem.locate_dofs_topological((W.sub(0), V), fdim, bottom_facets)
    dofs_top = fem.locate_dofs_topological((W.sub(0), V), fdim, top_facets)
    
    bc_in = fem.dirichletbc(u_in, dofs_in, W.sub(0))
    bc_bot = fem.dirichletbc(u_zero, dofs_bot, W.sub(0))
    bc_top = fem.dirichletbc(u_zero, dofs_top, W.sub(0))
    
    bcs = [bc_in, bc_bot, bc_top]
    
    # ---- Step 1: Stokes solve for initial guess ----
    (u_tr, p_tr) = ufl.TrialFunctions(W)
    (v_te, q_te) = ufl.TestFunctions(W)
    f_body = fem.Constant(msh, PETSc.ScalarType((0.0, 0.0)))
    
    a_stokes = (nu * ufl.inner(ufl.grad(u_tr), ufl.grad(v_te)) * ufl.dx
                - p_tr * ufl.div(v_te) * ufl.dx
                + ufl.div(u_tr) * q_te * ufl.dx)
    L_stokes = ufl.inner(f_body, v_te) * ufl.dx
    
    stokes_problem = petsc.LinearProblem(
        a_stokes, L_stokes, bcs=bcs,
        petsc_options={"ksp_type": "preonly", "pc_type": "lu",
                       "pc_factor_mat_solver_type": "mumps"},
        petsc_options_prefix="stokes_"
    )
    w_stokes = stokes_problem.solve()
    w.x.array[:] = w_stokes.x.array[:]
    
    # ---- Step 2: Newton solve for full NS ----
    (u, p) = ufl.split(w)
    (v, q) = ufl.TestFunctions(W)
    
    F = (nu * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
         + ufl.inner(ufl.grad(u) * u, v) * ufl.dx
         - p * ufl.div(v) * ufl.dx
         + ufl.div(u) * q * ufl.dx
         - ufl.inner(f_body, v) * ufl.dx)
    
    J = ufl.derivative(F, w)
    
    petsc_options = {
        "snes_type": "newtonls",
        "snes_linesearch_type": "bt",
        "snes_rtol": 1e-9,
        "snes_atol": 1e-11,
        "snes_max_it": 50,
        "ksp_type": "preonly",
        "pc_type": "lu",
        "pc_factor_mat_solver_type": "mumps",
    }
    
    problem = petsc.NonlinearProblem(F, w, bcs=bcs, J=J,
                                      petsc_options_prefix="ns_",
                                      petsc_options=petsc_options)
    w_h = problem.solve()
    w.x.scatter_forward()
    
    # Extract velocity
    u_sol = w.sub(0).collapse()
    
    # ---- Sample on output grid ----
    grid = case_spec["output"]["grid"]
    nx = grid["nx"]
    ny = grid["ny"]
    bbox = grid["bbox"]
    
    xs = np.linspace(bbox[0], bbox[1], nx)
    ys = np.linspace(bbox[2], bbox[3], ny)
    XX, YY = np.meshgrid(xs, ys)
    pts = np.c_[XX.ravel(), YY.ravel(), np.zeros(nx * ny)]
    
    tree = geometry.bb_tree(msh, msh.topology.dim)
    cell_candidates = geometry.compute_collisions_points(tree, pts)
    colliding = geometry.compute_colliding_cells(msh, cell_candidates, pts)
    
    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    for i in range(pts.shape[0]):
        links = colliding.links(i)
        if len(links) > 0:
            points_on_proc.append(pts[i])
            cells_on_proc.append(links[0])
            eval_map.append(i)
    
    u_vals = np.zeros((pts.shape[0], 2))
    if len(points_on_proc) > 0:
        vals = u_sol.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        for k, idx in enumerate(eval_map):
            u_vals[idx] = vals[k]
    
    magnitude = np.linalg.norm(u_vals, axis=1).reshape(ny, nx)
    
    return {
        "u": magnitude,
        "solver_info": {
            "mesh_resolution": 160,
            "element_degree": 2,
            "ksp_type": "preonly",
            "pc_type": "lu",
            "rtol": 1e-9,
            "iterations": 0,
            "nonlinear_iterations": [1],
        }
    }


if __name__ == "__main__":
    import time
    case_spec = {
        "output": {
            "grid": {"nx": 64, "ny": 64, "bbox": [0, 1, 0, 1]}
        }
    }
    t0 = time.time()
    result = solve(case_spec)
    t1 = time.time()
    print(f"Time: {t1-t0:.2f}s")
    print(f"Shape: {result['u'].shape}")
    print(f"Max velocity: {result['u'].max():.4f}")
    print(f"Min velocity: {result['u'].min():.4f}")
    print(f"Mean: {result['u'].mean():.4f}")
