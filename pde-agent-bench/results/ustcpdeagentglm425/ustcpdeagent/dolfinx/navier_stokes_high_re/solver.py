import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
from basix.ufl import element as basix_element, mixed_element as basix_mixed_element
import ufl
from petsc4py import PETSc


def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    # ── Parameters ──────────────────────────────────────────────
    nu = 0.02
    N = 64  # mesh resolution (adjustable for accuracy/time tradeoff)

    # ── Mesh ────────────────────────────────────────────────────
    msh = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
    gdim = msh.geometry.dim
    tdim = msh.topology.dim
    fdim = tdim - 1

    # ── Mixed element (Taylor–Hood P2/P1) ───────────────────────
    vel_el  = basix_element("Lagrange", msh.topology.cell_name(), 2, shape=(gdim,))
    pres_el = basix_element("Lagrange", msh.topology.cell_name(), 1)
    W = fem.functionspace(msh, basix_mixed_element([vel_el, pres_el]))
    V, V_to_W = W.sub(0).collapse()   # velocity subspace
    Q, Q_to_W = W.sub(1).collapse()   # pressure subspace

    # ── Unknown and test functions ──────────────────────────────
    w = fem.Function(W)
    (u, p) = ufl.split(w)
    (v, q) = ufl.TestFunctions(W)

    # ── Spatial coordinate & constants ─────────────────────────
    x = ufl.SpatialCoordinate(msh)
    pi = ufl.pi

    # ── Exact velocity (for BCs and source derivation) ─────────
    u_exact = ufl.as_vector([
        pi * ufl.cos(pi * x[1]) * ufl.sin(pi * x[0]),
       -pi * ufl.cos(pi * x[0]) * ufl.sin(pi * x[1])
    ])

    # ── Source term f (analytical) ─────────────────────────────
    # f = u_ex·∇u_ex  -  ν ∇²u_ex  +  ∇p_ex
    # where p_ex = 0, and we computed:
    #   (u·∇u)_1 = π³ sin(πx)cos(πx)
    #   (u·∇u)_2 = π³ sin(πy)cos(πy)
    #   -ν∇²u_1  = 2νπ³ cos(πy)sin(πx)
    #   -ν∇²u_2  = -2νπ³ cos(πx)sin(πy)
    f_body = ufl.as_vector([
        pi**3 * ufl.sin(pi*x[0]) * ufl.cos(pi*x[0])
        + 2.0*nu*pi**3 * ufl.cos(pi*x[1]) * ufl.sin(pi*x[0]),
        pi**3 * ufl.sin(pi*x[1]) * ufl.cos(pi*x[1])
        - 2.0*nu*pi**3 * ufl.cos(pi*x[0]) * ufl.sin(pi*x[1])
    ])

    # ── Residual F ─────────────────────────────────────────────
    def eps(uu):
        return ufl.sym(ufl.grad(uu))

    def sigma(uu, pp):
        return 2.0 * nu * eps(uu) - pp * ufl.Identity(gdim)

    F = (
        ufl.inner(sigma(u, p), eps(v)) * ufl.dx
        + ufl.inner(ufl.grad(u) * u, v) * ufl.dx
        - ufl.inner(f_body, v) * ufl.dx
        + ufl.inner(ufl.div(u), q) * ufl.dx
    )

    # ── Boundary conditions ────────────────────────────────────
    # Velocity BC: u = u_exact on entire boundary
    boundary_facets = mesh.locate_entities_boundary(
        msh, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
    )
    u_bc_func = fem.Function(V)
    u_bc_func.interpolate(lambda X: np.array([
        np.pi * np.cos(np.pi * X[1]) * np.sin(np.pi * X[0]),
       -np.pi * np.cos(np.pi * X[0]) * np.sin(np.pi * X[1])
    ]))
    dofs_u = fem.locate_dofs_topological((W.sub(0), V), fdim, boundary_facets)
    bc_u = fem.dirichletbc(u_bc_func, dofs_u, W.sub(0))

    # Pressure pin: p = 0 at origin
    p_dofs = fem.locate_dofs_geometrical(
        (W.sub(1), Q),
        lambda x: np.isclose(x[0], 0.0) & np.isclose(x[1], 0.0)
    )
    p0_func = fem.Function(Q)
    p0_func.x.array[:] = 0.0
    bc_p = fem.dirichletbc(p0_func, p_dofs, W.sub(1))

    bcs = [bc_u, bc_p]

    # ── Step 1: Stokes solve (initial guess) ───────────────────
    (u_t, p_t) = ufl.TrialFunctions(W)
    a_stokes = (
        2.0*nu * ufl.inner(ufl.sym(ufl.grad(u_t)), ufl.sym(ufl.grad(v))) * ufl.dx
        - p_t * ufl.div(v) * ufl.dx
        + ufl.div(u_t) * q * ufl.dx
    )
    L_stokes = ufl.inner(f_body, v) * ufl.dx

    stokes_problem = petsc.LinearProblem(
        a_stokes, L_stokes, bcs=bcs,
        petsc_options={
            "ksp_type": "preonly",
            "pc_type": "lu",
            "pc_factor_mat_solver_type": "mumps",
        },
        petsc_options_prefix="stokes_",
    )
    w_stokes = stokes_problem.solve()
    w.x.array[:] = w_stokes.x.array[:]
    w.x.scatter_forward()

    # ── Step 2: Newton solve for Navier–Stokes ─────────────────
    J_form = ufl.derivative(F, w)

    ns_petsc_opts = {
        "snes_type": "newtonls",
        "snes_linesearch_type": "bt",
        "snes_rtol": 1e-10,
        "snes_atol": 1e-12,
        "snes_max_it": 50,
        "ksp_type": "preonly",
        "pc_type": "lu",
        "pc_factor_mat_solver_type": "mumps",
    }

    ns_problem = petsc.NonlinearProblem(
        F, w, bcs=bcs, J=J_form,
        petsc_options_prefix="ns_",
        petsc_options=ns_petsc_opts,
    )

    w_h = ns_problem.solve()
    w.x.scatter_forward()

    # ── Solver info ────────────────────────────────────────────
    snes = ns_problem.solver
    newton_its = int(snes.getIterationNumber())
    linear_its = int(snes.getLinearSolveIterations())

    # ── Extract velocity sub-function ──────────────────────────
    u_h = w.sub(0).collapse()

    # ── Sample velocity magnitude on output grid ──────────────
    grid_spec = case_spec["output"]["grid"]
    nx_out = grid_spec["nx"]
    ny_out = grid_spec["ny"]
    bbox  = grid_spec["bbox"]  # [xmin, xmax, ymin, ymax]

    xs = np.linspace(bbox[0], bbox[1], nx_out)
    ys = np.linspace(bbox[2], bbox[3], ny_out)
    XX, YY = np.meshgrid(xs, ys)
    pts = np.column_stack([XX.ravel(), YY.ravel(), np.zeros(nx_out * ny_out)])

    bb_tree = geometry.bb_tree(msh, tdim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, pts)
    colliding_cells = geometry.compute_colliding_cells(msh, cell_candidates, pts)

    points_on_proc = []
    cells_on_proc  = []
    eval_map       = []
    for i in range(pts.shape[0]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(pts[i])
            cells_on_proc.append(links[0])
            eval_map.append(i)

    u_values = np.full((pts.shape[0], gdim), np.nan)
    if len(points_on_proc) > 0:
        pts_arr  = np.array(points_on_proc)
        cell_arr = np.array(cells_on_proc, dtype=np.int32)
        vals = u_h.eval(pts_arr, cell_arr)
        u_values[eval_map] = vals

    # Handle MPI: gather from all ranks
    if comm.size > 1:
        all_u = np.zeros_like(u_values)
        comm.Allreduce(u_values, all_u, op=MPI.SUM)
        # Fix NaN: if any rank had non-NaN, use that
        nan_mask = np.isnan(u_values)
        u_values = np.where(nan_mask, all_u, u_values)
        # If still NaN (point not on any proc), use 0
        u_values = np.nan_to_num(u_values, nan=0.0)

    u_mag = np.sqrt(u_values[:, 0]**2 + u_values[:, 1]**2)
    u_grid = u_mag.reshape(ny_out, nx_out)

    # ── Accuracy verification: L2 error ───────────────────────
    u_exact_expr = ufl.as_vector([
        pi * ufl.cos(pi * x[1]) * ufl.sin(pi * x[0]),
       -pi * ufl.cos(pi * x[0]) * ufl.sin(pi * x[1])
    ])
    # Interpolate exact solution for comparison
    u_ex_func = fem.Function(V)
    u_ex_func.interpolate(lambda X: np.array([
        np.pi * np.cos(np.pi * X[1]) * np.sin(np.pi * X[0]),
       -np.pi * np.cos(np.pi * X[0]) * np.sin(np.pi * X[1])
    ]))

    # Compute L2 error via UFL
    L2_err_ufl = fem.assemble_scalar(
        fem.form(ufl.inner(u_h - u_ex_func, u_h - u_ex_func) * ufl.dx)
    )
    L2_err_ufl = np.sqrt(comm.allreduce(L2_err_ufl, op=MPI.SUM))
    if rank == 0:
        print(f"[solver] L2 velocity error = {L2_err_ufl:.6e}")
        print(f"[solver] Newton its = {newton_its}, Linear its = {linear_its}")

    # ── Return ─────────────────────────────────────────────────
    return {
        "u": u_grid,
        "solver_info": {
            "mesh_resolution": N,
            "element_degree": 2,
            "ksp_type": "preonly",
            "pc_type": "lu",
            "rtol": 1e-10,
            "iterations": linear_its,
            "nonlinear_iterations": [newton_its],
        },
    }


if __name__ == "__main__":
    # Quick self-test with a dummy case_spec
    case_spec = {
        "output": {
            "grid": {"nx": 33, "ny": 33, "bbox": [0.0, 1.0, 0.0, 1.0]}
        },
        "pde": {"time": None},
    }
    import time
    t0 = time.perf_counter()
    result = solve(case_spec)
    t1 = time.perf_counter()
    print(f"Wall time: {t1-t0:.2f} s")
    print(f"Output shape: {result['u'].shape}")
    print(f"Max velocity magnitude: {np.nanmax(result['u']):.6f}")
    print(f"Solver info: {result['solver_info']}")
