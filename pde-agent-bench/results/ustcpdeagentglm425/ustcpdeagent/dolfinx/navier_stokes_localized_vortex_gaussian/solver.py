import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc
from basix.ufl import element as basix_element, mixed_element as basix_mixed_element


def solve(case_spec: dict) -> dict:
    nu = 0.12
    nx_out = case_spec["output"]["grid"]["nx"]
    ny_out = case_spec["output"]["grid"]["ny"]
    bbox = case_spec["output"]["grid"]["bbox"]

    N = 300
    comm = MPI.COMM_WORLD

    msh = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
    gdim = msh.geometry.dim
    tdim = msh.topology.dim
    fdim = tdim - 1

    # Taylor-Hood P2/P1
    vel_el = basix_element("Lagrange", msh.topology.cell_name(), 2, shape=(gdim,))
    pres_el = basix_element("Lagrange", msh.topology.cell_name(), 1)
    W = fem.functionspace(msh, basix_mixed_element([vel_el, pres_el]))
    V, _ = W.sub(0).collapse()
    Q, _ = W.sub(1).collapse()

    # Exact solution (UFL) for MMS
    x = ufl.SpatialCoordinate(msh)
    r2 = (x[0] - 0.5)**2 + (x[1] - 0.5)**2
    exp_factor = ufl.exp(-20 * r2)
    u_ex_ufl = ufl.as_vector((-40*(x[1]-0.5)*exp_factor, 40*(x[0]-0.5)*exp_factor))
    zero_vec = ufl.as_vector([0.0 * x[0], 0.0 * x[1]])
    f_ufl = -nu * ufl.div(ufl.grad(u_ex_ufl)) + ufl.grad(u_ex_ufl) * u_ex_ufl + zero_vec

    # Interpolate f for numerical stability
    V_vec = fem.functionspace(msh, ("Lagrange", 1, (gdim,)))
    f_func = fem.Function(V_vec)
    f_expr_interp = fem.Expression(f_ufl, V_vec.element.interpolation_points)
    f_func.interpolate(f_expr_interp)

    # Exact solution callable for BCs and error
    def u_exact_callable(x_pts):
        r2_val = (x_pts[0] - 0.5)**2 + (x_pts[1] - 0.5)**2
        exp_f = np.exp(-20 * r2_val)
        return np.vstack([-40*(x_pts[1]-0.5)*exp_f, 40*(x_pts[0]-0.5)*exp_f])

    # --- BCs ---
    boundary_facets = mesh.locate_entities_boundary(msh, fdim, lambda x: np.ones(x.shape[1], dtype=bool))
    u_bc_func = fem.Function(V)
    u_bc_func.interpolate(u_exact_callable)
    dofs_u = fem.locate_dofs_topological((W.sub(0), V), fdim, boundary_facets)
    bc_u = fem.dirichletbc(u_bc_func, dofs_u, W.sub(0))

    p_dofs = fem.locate_dofs_geometrical(
        (W.sub(1), Q),
        lambda x: np.isclose(x[0], 0.0) & np.isclose(x[1], 0.0)
    )
    p0_func = fem.Function(Q)
    p0_func.x.array[:] = 0.0
    bc_p = fem.dirichletbc(p0_func, p_dofs, W.sub(1))
    bcs = [bc_u, bc_p]

    # --- Stokes solve for initial guess ---
    w = fem.Function(W)
    (u_t, p_t) = ufl.TrialFunctions(W)
    (v, q) = ufl.TestFunctions(W)

    a_stokes = (
        nu * ufl.inner(ufl.grad(u_t), ufl.grad(v)) * ufl.dx
        - p_t * ufl.div(v) * ufl.dx
        + ufl.div(u_t) * q * ufl.dx
    )
    L_stokes = ufl.inner(f_func, v) * ufl.dx

    stokes_problem = petsc.LinearProblem(
        a_stokes, L_stokes, u=w, bcs=bcs,
        petsc_options={"ksp_type": "preonly", "pc_type": "lu", "pc_factor_mat_solver_type": "mumps"},
        petsc_options_prefix="stokes_"
    )
    stokes_problem.solve()
    w.x.scatter_forward()

    # --- Newton solve for full NS ---
    (u, p) = ufl.split(w)
    F = (
        nu * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
        + ufl.inner(ufl.grad(u) * u, v) * ufl.dx
        - p * ufl.div(v) * ufl.dx
        + ufl.div(u) * q * ufl.dx
        - ufl.inner(f_func, v) * ufl.dx
    )
    J = ufl.derivative(F, w)

    ns_options = {
        "snes_type": "newtonls",
        "snes_linesearch_type": "bt",
        "snes_rtol": 1e-8,
        "snes_atol": 1e-10,
        "snes_max_it": 50,
        "ksp_type": "gmres",
        "pc_type": "lu",
        "pc_factor_mat_solver_type": "mumps",
        "ksp_rtol": 1e-10,
    }

    problem = petsc.NonlinearProblem(
        F, w, bcs=bcs, J=J,
        petsc_options_prefix="ns_",
        petsc_options=ns_options
    )

    w_h = problem.solve()
    w.x.scatter_forward()

    # Iteration info
    newton_its = 0
    ksp_its_total = 0
    try:
        snes = problem._snes
        newton_its = int(snes.getIterationNumber())
        ksp_its_total = int(snes.getLinearSolveIterations())
    except Exception:
        pass

    # --- Evaluate velocity magnitude on output grid ---
    u_h = w.sub(0).collapse()

    xmin, xmax, ymin, ymax = bbox
    xs = np.linspace(xmin, xmax, nx_out)
    ys = np.linspace(ymin, ymax, ny_out)
    XX, YY = np.meshgrid(xs, ys)
    pts = np.vstack([XX.ravel(), YY.ravel(), np.zeros(nx_out * ny_out)])

    bb_tree = geometry.bb_tree(msh, tdim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, pts.T)
    colliding_cells = geometry.compute_colliding_cells(msh, cell_candidates, pts.T)

    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    for i in range(pts.shape[1]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(pts.T[i])
            cells_on_proc.append(links[0])
            eval_map.append(i)

    u_values = np.full((pts.shape[1], gdim), np.nan)
    if len(points_on_proc) > 0:
        vals = u_h.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u_values[eval_map] = vals

    magnitude = np.sqrt(np.nansum(u_values**2, axis=1))
    u_grid = magnitude.reshape(ny_out, nx_out)

    # --- L2 error verification ---
    u_ex_func_V = fem.Function(V)
    u_ex_func_V.interpolate(u_exact_callable)
    error_sq = fem.assemble_scalar(
        fem.form(ufl.inner(u_h - u_ex_func_V, u_h - u_ex_func_V) * ufl.dx)
    )
    error_sq = msh.comm.allreduce(error_sq, op=MPI.SUM)
    l2_error = np.sqrt(error_sq)

    if comm.rank == 0:
        print(f"Mesh: {N}x{N}, L2 velocity error: {l2_error:.6e}")
        print(f"Newton iterations: {newton_its}, KSP iterations: {ksp_its_total}")

    return {
        "u": u_grid,
        "solver_info": {
            "mesh_resolution": N,
            "element_degree": 2,
            "ksp_type": "gmres",
            "pc_type": "lu",
            "rtol": 1e-8,
            "iterations": ksp_its_total,
            "nonlinear_iterations": [newton_its],
        }
    }
