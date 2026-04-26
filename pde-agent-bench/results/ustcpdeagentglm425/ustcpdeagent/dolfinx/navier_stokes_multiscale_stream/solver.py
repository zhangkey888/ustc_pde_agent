import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc
from basix.ufl import element as basix_element, mixed_element as basix_mixed_element

def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD
    
    # ── PDE parameters ──
    pde_params = case_spec.get("pde", {}).get("params", {})
    nu_val = float(pde_params.get("viscosity", 0.12))
    
    # ── Output grid ──
    nx_out = case_spec["output"]["grid"]["nx"]
    ny_out = case_spec["output"]["grid"]["ny"]
    bbox   = case_spec["output"]["grid"]["bbox"]
    
    # ── Mesh resolution ──
    N = 64
    
    msh = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
    gdim = msh.geometry.dim
    fdim = msh.topology.dim - 1
    
    # ── Taylor-Hood P2/P1 mixed element ──
    vel_el  = basix_element("Lagrange", msh.topology.cell_name(), 2, shape=(gdim,))
    pres_el = basix_element("Lagrange", msh.topology.cell_name(), 1)
    W = fem.functionspace(msh, basix_mixed_element([vel_el, pres_el]))
    
    V, _ = W.sub(0).collapse()
    Q, _ = W.sub(1).collapse()
    
    # ── Compute forcing and exact solution using sympy ──
    import sympy as sp
    sx, sy = sp.symbols('x y')
    spi = sp.pi
    snu = sp.Rational(str(nu_val))
    
    su1 = spi*sp.cos(spi*sy)*sp.sin(spi*sx) + sp.Rational(3,5)*spi*sp.cos(2*spi*sy)*sp.sin(3*spi*sx)
    su2 = -spi*sp.cos(spi*sx)*sp.sin(spi*sy) - sp.Rational(9,10)*spi*sp.cos(3*spi*sx)*sp.sin(2*spi*sy)
    sp_ex = sp.cos(2*spi*sx)*sp.cos(spi*sy)
    
    # Convective terms
    sconv1 = su1*sp.diff(su1, sx) + su2*sp.diff(su1, sy)
    sconv2 = su1*sp.diff(su2, sx) + su2*sp.diff(su2, sy)
    
    # Laplacian
    slap1 = sp.diff(su1, sx, 2) + sp.diff(su1, sy, 2)
    slap2 = sp.diff(su2, sx, 2) + sp.diff(su2, sy, 2)
    
    # Pressure gradient
    sdp_dx = sp.diff(sp_ex, sx)
    sdp_dy = sp.diff(sp_ex, sy)
    
    # Forcing f = (u·∇)u - ν Δu + ∇p
    sf1 = sp.simplify(sconv1 - snu*slap1 + sdp_dx)
    sf2 = sp.simplify(sconv2 - snu*slap2 + sdp_dy)
    
    # Lambdify
    f1_func = sp.lambdify((sx, sy), sf1, 'numpy')
    f2_func = sp.lambdify((sx, sy), sf2, 'numpy')
    u1_np = sp.lambdify((sx, sy), su1, 'numpy')
    u2_np = sp.lambdify((sx, sy), su2, 'numpy')
    p_np = sp.lambdify((sx, sy), sp_ex, 'numpy')
    
    # ── Create dolfinx functions for forcing and BCs ──
    f_force = fem.Function(V)
    f_force.interpolate(lambda x: np.vstack([f1_func(x[0], x[1]), f2_func(x[0], x[1])]))
    
    u_bc_func = fem.Function(V)
    u_bc_func.interpolate(lambda x: np.vstack([u1_np(x[0], x[1]), u2_np(x[0], x[1])]))
    
    # ── UFL unknowns and test functions ──
    w = fem.Function(W)
    (u, p) = ufl.split(w)
    (v, q) = ufl.TestFunctions(W)
    
    nu = PETSc.ScalarType(nu_val)
    
    def eps_f(u_):
        return ufl.sym(ufl.grad(u_))
    
    # ── NS residual F = 0 ──
    F = (
        ufl.inner(2.0 * nu * eps_f(u), eps_f(v)) * ufl.dx
        + ufl.inner(ufl.grad(u) * u, v) * ufl.dx
        - ufl.inner(f_force, v) * ufl.dx
        - ufl.inner(p, ufl.div(v)) * ufl.dx
        + ufl.inner(ufl.div(u), q) * ufl.dx
    )
    
    J_form = ufl.derivative(F, w)
    
    # ── Boundary conditions ──
    boundary_facets = mesh.locate_entities_boundary(
        msh, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
    )
    
    bc_u = fem.dirichletbc(u_bc_func,
                           fem.locate_dofs_topological((W.sub(0), V), fdim, boundary_facets),
                           W.sub(0))
    
    bcs = [bc_u]
    
    # ── Pressure pinning at (0,0) ──
    p_dofs = fem.locate_dofs_geometrical(
        (W.sub(1), Q),
        lambda x: np.isclose(x[0], 0.0) & np.isclose(x[1], 0.0)
    )
    if len(p_dofs) > 0:
        p0_func = fem.Function(Q)
        p0_func.x.array[:] = 0.0
        bc_p = fem.dirichletbc(p0_func, p_dofs, W.sub(1))
        bcs.append(bc_p)
    
    # ── Step 1: Stokes solve for initial guess ──
    w0 = ufl.TrialFunction(W)
    (u0, p0_t) = ufl.split(w0)
    
    a_stokes = (
        ufl.inner(2.0 * nu * eps_f(u0), eps_f(v)) * ufl.dx
        - ufl.inner(p0_t, ufl.div(v)) * ufl.dx
        + ufl.inner(ufl.div(u0), q) * ufl.dx
    )
    L_stokes = ufl.inner(f_force, v) * ufl.dx
    
    w_stokes = petsc.LinearProblem(
        a_stokes, L_stokes, bcs=bcs,
        petsc_options={
            "ksp_type": "preonly",
            "pc_type": "lu",
            "pc_factor_mat_solver_type": "mumps",
        },
        petsc_options_prefix="stokes_"
    ).solve()
    w.x.array[:] = w_stokes.x.array[:]
    w.x.scatter_forward()
    
    # ── Step 2: Newton solve for NS ──
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
    
    ns_problem = petsc.NonlinearProblem(
        F, w, bcs=bcs, J=J_form,
        petsc_options_prefix="ns_",
        petsc_options=petsc_options
    )
    
    w_h = ns_problem.solve()
    w.x.scatter_forward()
    
    # ── Extract velocity ──
    u_h = w_h.sub(0).collapse()
    
    # ── Compute L2 error by sampling ──
    n_err = 50
    xs_err = np.linspace(0, 1, n_err)
    ys_err = np.linspace(0, 1, n_err)
    XXe, YYe = np.meshgrid(xs_err, ys_err)
    pts_err = np.column_stack([XXe.ravel(), YYe.ravel(), np.zeros(n_err**2)])
    
    bb_tree = geometry.bb_tree(msh, msh.topology.dim)
    cell_cand = geometry.compute_collisions_points(bb_tree, pts_err)
    coll_cells = geometry.compute_colliding_cells(msh, cell_cand, pts_err)
    
    pts_proc = []
    cls_proc = []
    emap = []
    for i in range(pts_err.shape[0]):
        lnks = coll_cells.links(i)
        if len(lnks) > 0:
            pts_proc.append(pts_err[i])
            cls_proc.append(lnks[0])
            emap.append(i)
    
    err_vals = np.full((n_err**2,), np.nan)
    if len(pts_proc) > 0:
        ev = u_h.eval(np.array(pts_proc), np.array(cls_proc, dtype=np.int32))
        err_vals[emap] = np.linalg.norm(ev, axis=1)
    
    err_global = np.zeros_like(err_vals) if comm.rank == 0 else None
    comm.Reduce(err_vals, err_global, op=MPI.SUM, root=0)
    
    if comm.rank == 0:
        u_exact_mag = np.sqrt(u1_np(XXe, YYe)**2 + u2_np(XXe, YYe)**2)
        err_diff = np.abs(err_global.reshape(n_err, n_err) - u_exact_mag)
        L2_approx = np.sqrt(np.mean(err_diff**2))
        print(f"Approx L2 velocity error: {L2_approx:.6e}")
    
    # ── Sample on output grid ──
    xs = np.linspace(bbox[0], bbox[1], nx_out)
    ys = np.linspace(bbox[2], bbox[3], ny_out)
    XX, YY = np.meshgrid(xs, ys)
    pts_flat = np.column_stack([XX.ravel(), YY.ravel(), np.zeros(nx_out * ny_out)])
    
    bb_tree2 = geometry.bb_tree(msh, msh.topology.dim)
    cell_cand2 = geometry.compute_collisions_points(bb_tree2, pts_flat)
    coll_cells2 = geometry.compute_colliding_cells(msh, cell_cand2, pts_flat)
    
    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    for i in range(pts_flat.shape[0]):
        links = coll_cells2.links(i)
        if len(links) > 0:
            points_on_proc.append(pts_flat[i])
            cells_on_proc.append(links[0])
            eval_map.append(i)
    
    u_mag_values = np.full((nx_out * ny_out,), np.nan)
    if len(points_on_proc) > 0:
        u_vals = u_h.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u_mag_values[eval_map] = np.linalg.norm(u_vals, axis=1)
    
    u_mag_global = np.zeros_like(u_mag_values) if comm.rank == 0 else None
    comm.Reduce(u_mag_values, u_mag_global, op=MPI.SUM, root=0)
    
    if comm.rank == 0:
        u_grid = u_mag_global.reshape(ny_out, nx_out)
    else:
        u_grid = np.zeros((ny_out, nx_out))
    
    u_grid = comm.bcast(u_grid, root=0)
    
    # ── Solver info ──
    snes = ns_problem._snes
    newton_its = snes.getIterationNumber()
    ksp_its = snes.getLinearSolveIterations()
    
    solver_info = {
        "mesh_resolution": N,
        "element_degree": 2,
        "ksp_type": "preonly",
        "pc_type": "lu",
        "rtol": 1e-10,
        "iterations": int(ksp_its),
        "nonlinear_iterations": [int(newton_its)],
    }
    
    if comm.rank == 0:
        print(f"Newton iterations: {newton_its}, KSP iterations: {ksp_its}")
    
    return {
        "u": u_grid,
        "solver_info": solver_info,
    }
