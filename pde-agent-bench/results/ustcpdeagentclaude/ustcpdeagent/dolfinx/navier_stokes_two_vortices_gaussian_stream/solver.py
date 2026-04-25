import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
from basix.ufl import element as basix_element, mixed_element as basix_mixed_element
import ufl
from petsc4py import PETSc


def _manufactured(msh):
    x = ufl.SpatialCoordinate(msh)
    nu = 0.14
    u1 = (-60.0 * (x[1] - 0.7) * ufl.exp(-30.0 * ((x[0] - 0.3) ** 2 + (x[1] - 0.7) ** 2))
          + 60.0 * (x[1] - 0.3) * ufl.exp(-30.0 * ((x[0] - 0.7) ** 2 + (x[1] - 0.3) ** 2)))
    u2 = (60.0 * (x[0] - 0.3) * ufl.exp(-30.0 * ((x[0] - 0.3) ** 2 + (x[1] - 0.7) ** 2))
          - 60.0 * (x[0] - 0.7) * ufl.exp(-30.0 * ((x[0] - 0.7) ** 2 + (x[1] - 0.3) ** 2)))
    u_ex = ufl.as_vector([u1, u2])
    p_ex = 0.0 * x[0]
    # f = (u . grad) u - nu * div(grad u) + grad p
    f = ufl.grad(u_ex) * u_ex - nu * ufl.div(ufl.grad(u_ex)) + ufl.as_vector([0.0*ufl.SpatialCoordinate(u_ex.ufl_domain())[0], 0.0*ufl.SpatialCoordinate(u_ex.ufl_domain())[0]])
    return u_ex, p_ex, f, nu


def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD

    N = 96
    msh = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
    gdim = msh.geometry.dim

    deg_u, deg_p = 3, 2
    vel_el = basix_element("Lagrange", msh.topology.cell_name(), deg_u, shape=(gdim,))
    pres_el = basix_element("Lagrange", msh.topology.cell_name(), deg_p)
    W = fem.functionspace(msh, basix_mixed_element([vel_el, pres_el]))
    V, _ = W.sub(0).collapse()
    Q, _ = W.sub(1).collapse()

    u_ex, p_ex, f_src, nu = _manufactured(msh)

    # Dirichlet BC on velocity: u = u_ex on whole boundary
    u_bc = fem.Function(V)
    u_bc.interpolate(fem.Expression(u_ex, V.element.interpolation_points))

    fdim = msh.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(
        msh, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
    )
    dofs_u = fem.locate_dofs_topological((W.sub(0), V), fdim, boundary_facets)
    bc_u = fem.dirichletbc(u_bc, dofs_u, W.sub(0))

    # Pressure pin at (0,0)
    p_dofs = fem.locate_dofs_geometrical(
        (W.sub(1), Q),
        lambda x: np.isclose(x[0], 0.0) & np.isclose(x[1], 0.0),
    )
    p_pin = fem.Function(Q)
    p_pin.x.array[:] = 0.0
    bc_p = fem.dirichletbc(p_pin, p_dofs, W.sub(1))

    bcs = [bc_u, bc_p]

    # Step 1: Stokes solve as initial guess
    w_stokes = fem.Function(W)
    (u_tr, p_tr) = ufl.TrialFunctions(W)
    (v_te, q_te) = ufl.TestFunctions(W)

    a_stokes = (nu * ufl.inner(ufl.grad(u_tr), ufl.grad(v_te)) * ufl.dx
                - p_tr * ufl.div(v_te) * ufl.dx
                + ufl.div(u_tr) * q_te * ufl.dx)
    L_stokes = ufl.inner(f_src, v_te) * ufl.dx

    stokes_problem = petsc.LinearProblem(
        a_stokes, L_stokes, bcs=bcs, u=w_stokes,
        petsc_options={"ksp_type": "preonly", "pc_type": "lu",
                       "pc_factor_mat_solver_type": "mumps"},
        petsc_options_prefix="stokes_init_",
    )
    stokes_problem.solve()

    # Step 2: Newton solve for full NS
    w = fem.Function(W)
    w.x.array[:] = w_stokes.x.array[:]
    (u, p) = ufl.split(w)
    (v, q) = ufl.TestFunctions(W)

    F = (nu * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
         + ufl.inner(ufl.grad(u) * u, v) * ufl.dx
         - p * ufl.div(v) * ufl.dx
         + ufl.div(u) * q * ufl.dx
         - ufl.inner(f_src, v) * ufl.dx)

    J = ufl.derivative(F, w)

    petsc_options = {
        "snes_type": "newtonls",
        "snes_linesearch_type": "bt",
        "snes_rtol": 1e-10,
        "snes_atol": 1e-12,
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
    try:
        snes = problem.solver
        n_newton = int(snes.getIterationNumber())
    except Exception:
        n_newton = -1

    # Extract velocity subfunction
    u_sol = w_h.sub(0).collapse()

    # Sample onto output grid
    grid = case_spec["output"]["grid"]
    nx, ny = int(grid["nx"]), int(grid["ny"])
    bbox = grid["bbox"]
    xs = np.linspace(bbox[0], bbox[1], nx)
    ys = np.linspace(bbox[2], bbox[3], ny)
    XX, YY = np.meshgrid(xs, ys)
    pts = np.c_[XX.ravel(), YY.ravel(), np.zeros(XX.size)]

    tree = geometry.bb_tree(msh, msh.topology.dim)
    cand = geometry.compute_collisions_points(tree, pts)
    coll = geometry.compute_colliding_cells(msh, cand, pts)

    points_on_proc = []
    cells_on_proc = []
    idx_map = []
    for i in range(pts.shape[0]):
        links = coll.links(i)
        if len(links) > 0:
            points_on_proc.append(pts[i])
            cells_on_proc.append(links[0])
            idx_map.append(i)

    vals = np.full((pts.shape[0], gdim), np.nan)
    if len(points_on_proc) > 0:
        v_eval = u_sol.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        vals[idx_map] = v_eval

    mag = np.linalg.norm(vals, axis=1).reshape(ny, nx)

    return {
        "u": mag,
        "solver_info": {
            "mesh_resolution": N,
            "element_degree": deg_u,
            "ksp_type": "preonly",
            "pc_type": "lu",
            "rtol": 1e-10,
            "iterations": n_newton,
            "nonlinear_iterations": [n_newton],
        },
    }


if __name__ == "__main__":
    import time
    spec = {"output": {"grid": {"nx": 128, "ny": 128, "bbox": [0.0, 1.0, 0.0, 1.0]}}}
    t0 = time.time()
    out = solve(spec)
    t1 = time.time()
    print("time:", t1 - t0)
    print("shape:", out["u"].shape)
    print("max mag:", np.nanmax(out["u"]))
    print("info:", out["solver_info"])

    # Exact magnitude on grid for error check
    nx, ny = 128, 128
    xs = np.linspace(0, 1, nx); ys = np.linspace(0, 1, ny)
    X, Y = np.meshgrid(xs, ys)
    u1 = (-60*(Y-0.7)*np.exp(-30*((X-0.3)**2+(Y-0.7)**2))
          + 60*(Y-0.3)*np.exp(-30*((X-0.7)**2+(Y-0.3)**2)))
    u2 = (60*(X-0.3)*np.exp(-30*((X-0.3)**2+(Y-0.7)**2))
          - 60*(X-0.7)*np.exp(-30*((X-0.7)**2+(Y-0.3)**2)))
    mag_ex = np.sqrt(u1**2 + u2**2)
    err = np.sqrt(np.mean((out["u"] - mag_ex)**2))
    print("RMSE vs exact magnitude:", err)
    print("Linf:", np.max(np.abs(out["u"] - mag_ex)))
