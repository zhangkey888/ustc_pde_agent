import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
from basix.ufl import element as basix_element, mixed_element as basix_mixed_element
import ufl
from petsc4py import PETSc


def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD
    
    # Output grid
    grid = case_spec["output"]["grid"]
    nx_out = grid["nx"]
    ny_out = grid["ny"]
    bbox = grid["bbox"]
    
    # Mesh
    N = 64
    msh = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
    gdim = msh.geometry.dim
    
    # Taylor-Hood P2-P1
    deg_u = 2
    deg_p = 1
    vel_el = basix_element("Lagrange", msh.topology.cell_name(), deg_u, shape=(gdim,))
    pres_el = basix_element("Lagrange", msh.topology.cell_name(), deg_p)
    W = fem.functionspace(msh, basix_mixed_element([vel_el, pres_el]))
    V, _ = W.sub(0).collapse()
    Q, _ = W.sub(1).collapse()
    
    nu = 0.2
    
    w = fem.Function(W)
    (u, p) = ufl.split(w)
    (v, q) = ufl.TestFunctions(W)
    
    f = fem.Constant(msh, PETSc.ScalarType((0.0, 0.0)))
    
    # Weak form - standard NS
    F = (
        nu * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
        + ufl.inner(ufl.grad(u) * u, v) * ufl.dx
        - p * ufl.div(v) * ufl.dx
        + q * ufl.div(u) * ufl.dx
        - ufl.inner(f, v) * ufl.dx
    )
    
    # BCs: u=[0.8,0] on y=1, u=[-0.8,0] on y=0
    fdim = msh.topology.dim - 1
    
    top_facets = mesh.locate_entities_boundary(msh, fdim, lambda x: np.isclose(x[1], 1.0))
    bot_facets = mesh.locate_entities_boundary(msh, fdim, lambda x: np.isclose(x[1], 0.0))
    
    u_top = fem.Function(V)
    u_top.interpolate(lambda x: np.vstack([np.full(x.shape[1], 0.8), np.zeros(x.shape[1])]))
    u_bot = fem.Function(V)
    u_bot.interpolate(lambda x: np.vstack([np.full(x.shape[1], -0.8), np.zeros(x.shape[1])]))
    
    dofs_top = fem.locate_dofs_topological((W.sub(0), V), fdim, top_facets)
    dofs_bot = fem.locate_dofs_topological((W.sub(0), V), fdim, bot_facets)
    
    bc_top = fem.dirichletbc(u_top, dofs_top, W.sub(0))
    bc_bot = fem.dirichletbc(u_bot, dofs_bot, W.sub(0))
    
    bcs = [bc_top, bc_bot]
    
    # Pressure pin at corner (0,0)
    p_dofs = fem.locate_dofs_geometrical(
        (W.sub(1), Q),
        lambda x: np.isclose(x[0], 0.0) & np.isclose(x[1], 0.0),
    )
    if len(p_dofs[0]) > 0:
        p0_func = fem.Function(Q)
        p0_func.x.array[:] = 0.0
        bc_p = fem.dirichletbc(p0_func, p_dofs, W.sub(1))
        bcs.append(bc_p)
    
    # Initial guess: linear interpolation
    def init_u(x):
        return np.vstack([0.8 * (2 * x[1] - 1), np.zeros(x.shape[1])])
    
    w_sub0 = w.sub(0)
    # Interpolate initial guess into velocity part
    u_init = fem.Function(V)
    u_init.interpolate(init_u)
    # assign into mixed
    w.x.array[:] = 0.0
    # Map V dofs into W.sub(0) — use a helper: just solve after setting via BCs satisfying initial
    # Simpler: leave w=0 and let Newton with good preconditioning converge
    
    J = ufl.derivative(F, w)
    
    petsc_options = {
        "snes_type": "newtonls",
        "snes_linesearch_type": "bt",
        "snes_rtol": 1e-12,
        "snes_atol": 1e-14,
        "snes_max_it": 50,
        "ksp_type": "preonly",
        "pc_type": "lu",
        "pc_factor_mat_solver_type": "mumps",
    }
    
    problem = petsc.NonlinearProblem(
        F, w, bcs=bcs, J=J,
        petsc_options_prefix="ns_",
        petsc_options=petsc_options,
    )
    
    w_h = problem.solve()
    w.x.scatter_forward()
    
    # Extract number of Newton iterations
    try:
        n_newton = problem.solver.getIterationNumber()
    except Exception:
        n_newton = -1
    
    # Split solution
    u_h = w.sub(0).collapse()
    
    # Sample on output grid
    xs = np.linspace(bbox[0], bbox[1], nx_out)
    ys = np.linspace(bbox[2], bbox[3], ny_out)
    XX, YY = np.meshgrid(xs, ys)
    pts = np.c_[XX.ravel(), YY.ravel(), np.zeros(nx_out * ny_out)]
    
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
    
    u_vals = np.zeros((pts.shape[0], gdim))
    if len(points_on_proc) > 0:
        vals = u_h.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u_vals[eval_map] = vals
    
    mag = np.linalg.norm(u_vals, axis=1).reshape(ny_out, nx_out)
    
    # Accuracy verification against analytical Couette: u_x = 0.8*(2y-1)
    u_exact_mag = np.abs(0.8 * (2 * YY - 1))
    err = np.max(np.abs(mag - u_exact_mag))
    print(f"[verify] max error vs Couette analytic: {err:.3e}, Newton iters: {n_newton}")
    
    return {
        "u": mag,
        "solver_info": {
            "mesh_resolution": N,
            "element_degree": deg_u,
            "ksp_type": "preonly",
            "pc_type": "lu",
            "rtol": 1e-12,
            "iterations": int(n_newton) if n_newton >= 0 else 0,
            "nonlinear_iterations": [int(n_newton) if n_newton >= 0 else 0],
        },
    }


if __name__ == "__main__":
    import time
    case_spec = {
        "output": {
            "grid": {
                "nx": 64,
                "ny": 64,
                "bbox": [0.0, 1.0, 0.0, 1.0],
            }
        }
    }
    t0 = time.time()
    result = solve(case_spec)
    t1 = time.time()
    print(f"Wall time: {t1-t0:.2f}s")
    print(f"Output shape: {result['u'].shape}")
    print(f"Max velocity magnitude: {result['u'].max()}")
    print(f"Min velocity magnitude: {result['u'].min()}")
