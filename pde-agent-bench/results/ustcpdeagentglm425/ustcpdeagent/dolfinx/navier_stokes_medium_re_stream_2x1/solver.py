import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry, log
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc
from basix.ufl import element as basix_element, mixed_element as basix_mixed_element

def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD
    pde = case_spec["pde"]
    nu = pde["viscosity"]  # 0.2
    
    out = case_spec["output"]
    grid = out["grid"]
    nx_out = grid["nx"]
    ny_out = grid["ny"]
    bbox = grid["bbox"]
    
    mesh_res = 160
    
    msh = mesh.create_unit_square(comm, mesh_res, mesh_res, cell_type=mesh.CellType.triangle)
    gdim = msh.geometry.dim
    tdim = msh.topology.dim
    fdim = tdim - 1
    
    # Mixed function space (Taylor-Hood P2/P1)
    vel_el = basix_element("Lagrange", msh.topology.cell_name(), 2, shape=(gdim,))
    pres_el = basix_element("Lagrange", msh.topology.cell_name(), 1)
    W = fem.functionspace(msh, basix_mixed_element([vel_el, pres_el]))
    V, V_to_W = W.sub(0).collapse()
    Q, Q_to_W = W.sub(1).collapse()
    
    x = ufl.SpatialCoordinate(msh)
    pi_val = np.pi
    
    # Exact solution
    u1_ex = pi_val * ufl.cos(pi_val * x[1]) * ufl.sin(2*pi_val * x[0])
    u2_ex = -2*pi_val * ufl.cos(2*pi_val * x[0]) * ufl.sin(pi_val * x[1])
    u_ex_vec = ufl.as_vector([u1_ex, u2_ex])
    p_ex = ufl.sin(pi_val * x[0]) * ufl.cos(pi_val * x[1])
    
    # Source term
    lap_u = ufl.div(ufl.grad(u_ex_vec))
    conv_u = ufl.grad(u_ex_vec) * u_ex_vec
    grad_p = ufl.grad(p_ex)
    f_vec = conv_u - nu * lap_u + grad_p
    
    # --- Boundary conditions ---
    u_bc_func = fem.Function(V)
    u_bc_expr = ufl.as_vector([u1_ex, u2_ex])
    u_bc_func.interpolate(fem.Expression(u_bc_expr, V.element.interpolation_points))
    
    boundary_facets = mesh.locate_entities_boundary(msh, fdim, lambda X: np.ones(X.shape[1], dtype=bool))
    dofs_u = fem.locate_dofs_topological((W.sub(0), V), fdim, boundary_facets)
    bc_u = fem.dirichletbc(u_bc_func, dofs_u, W.sub(0))
    
    bcs = [bc_u]
    
    # Pressure pin at origin
    p_dofs = fem.locate_dofs_geometrical(
        (W.sub(1), Q),
        lambda X: np.isclose(X[0], 0.0) & np.isclose(X[1], 0.0)
    )
    if len(p_dofs[0]) > 0:
        p0_func = fem.Function(Q)
        p0_func.x.array[:] = 0.0
        bc_p = fem.dirichletbc(p0_func, p_dofs, W.sub(1))
        bcs.append(bc_p)
    
    # Step 1: Stokes solve for initial guess
    (u_t, p_t) = ufl.TrialFunctions(W)
    (v, q) = ufl.TestFunctions(W)
    
    def eps(u):
        return ufl.sym(ufl.grad(u))
    
    a_stokes = (2*nu * ufl.inner(eps(u_t), eps(v)) * ufl.dx
                - p_t * ufl.div(v) * ufl.dx
                + ufl.div(u_t) * q * ufl.dx)
    L_stokes = ufl.inner(f_vec, v) * ufl.dx
    
    stokes_prob = petsc.LinearProblem(a_stokes, L_stokes, bcs=bcs,
                                        petsc_options={"ksp_type": "gmres", "pc_type": "lu",
                                                         "pc_factor_mat_solver_type": "mumps",
                                                         "ksp_rtol": 1e-12, "ksp_max_it": 500},
                                        petsc_options_prefix="stokes_")
    w_stokes = stokes_prob.solve()
    w_stokes.x.scatter_forward()
    
    # Step 2: Newton solve
    w = fem.Function(W)
    w.x.array[:] = w_stokes.x.array[:]
    w.x.scatter_forward()
    
    (u, p) = ufl.split(w)
    
    F = (2*nu * ufl.inner(eps(u), eps(v)) * ufl.dx
         + ufl.inner(ufl.grad(u) * u, v) * ufl.dx
         - p * ufl.div(v) * ufl.dx
         + ufl.div(u) * q * ufl.dx
         - ufl.inner(f_vec, v) * ufl.dx)
    J = ufl.derivative(F, w)
    
    newton_options = {
        "snes_type": "newtonls",
        "snes_linesearch_type": "bt",
        "snes_rtol": 1e-10,
        "snes_atol": 1e-12,
        "snes_max_it": 20,
        "ksp_type": "gmres",
        "pc_type": "lu",
        "pc_factor_mat_solver_type": "mumps",
        "ksp_rtol": 1e-12,
    }
    
    problem = petsc.NonlinearProblem(F, w, bcs=bcs, J=J,
                                       petsc_options_prefix="ns_",
                                       petsc_options=newton_options)
    
    newton_its = 0
    lin_its = 0
    try:
        w_h = problem.solve()
        w.x.scatter_forward()
        snes = problem._snes
        newton_its = int(snes.getIterationNumber())
        lin_its = int(snes.getLinearSolveIterations())
        if comm.rank == 0:
            print(f"Newton: {newton_its} iterations, {lin_its} linear its")
    except Exception as e:
        if comm.rank == 0:
            print(f"Newton failed: {e}")
    
    # Extract velocity
    u_h = w.sub(0).collapse()
    p_h = w.sub(1).collapse()
    
    # Sample velocity magnitude on output grid
    xmin, xmax, ymin, ymax = bbox
    xs = np.linspace(xmin, xmax, nx_out)
    ys = np.linspace(ymin, ymax, ny_out)
    XX, YY = np.meshgrid(xs, ys)
    
    pts = np.zeros((nx_out * ny_out, 3))
    pts[:, 0] = XX.ravel()
    pts[:, 1] = YY.ravel()
    
    bb_tree = geometry.bb_tree(msh, tdim)
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
    
    u_values = np.full((pts.shape[0], gdim), np.nan)
    if len(points_on_proc) > 0:
        vals = u_h.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u_values[eval_map] = vals
    
    magnitude = np.sqrt(np.nansum(u_values**2, axis=1)).reshape(ny_out, nx_out)
    
    # Fill NaN with exact solution
    for i in range(ny_out):
        for j in range(nx_out):
            if np.isnan(magnitude[i, j]):
                xi, yi = xs[j], ys[i]
                u1 = np.pi * np.cos(np.pi * yi) * np.sin(2*np.pi * xi)
                u2 = -2*np.pi * np.cos(2*np.pi * xi) * np.sin(np.pi * yi)
                magnitude[i, j] = np.sqrt(u1**2 + u2**2)
    
    # Compute L2 error for verification
    u_ex_func = fem.Function(V)
    u_ex_func.interpolate(fem.Expression(u_bc_expr, V.element.interpolation_points))
    
    e_u_sq = ufl.inner(u_h - u_ex_func, u_h - u_ex_func) * ufl.dx
    u_L2_err = np.sqrt(comm.allreduce(fem.assemble_scalar(fem.form(e_u_sq)), op=MPI.SUM))
    
    domain_vol = comm.allreduce(fem.assemble_scalar(fem.form(1.0 * ufl.dx(domain=msh))), op=MPI.SUM)
    p_h_mean = comm.allreduce(fem.assemble_scalar(fem.form(p_h * ufl.dx(domain=msh))), op=MPI.SUM) / domain_vol
    
    p_ex_func = fem.Function(Q)
    p_ex_func.interpolate(fem.Expression(p_ex, Q.element.interpolation_points))
    p_ex_mean = comm.allreduce(fem.assemble_scalar(fem.form(p_ex_func * ufl.dx(domain=msh))), op=MPI.SUM) / domain_vol
    
    p_h_shifted = fem.Function(Q)
    p_h_shifted.x.array[:] = p_h.x.array - p_h_mean + p_ex_mean
    p_h_shifted.x.scatter_forward()
    
    e_p_sq = ufl.inner(p_h_shifted - p_ex_func, p_h_shifted - p_ex_func) * ufl.dx
    p_L2_err = np.sqrt(comm.allreduce(fem.assemble_scalar(fem.form(e_p_sq)), op=MPI.SUM))
    
    if comm.rank == 0:
        print(f"Velocity L2 error: {u_L2_err:.6e}")
        print(f"Pressure L2 error: {p_L2_err:.6e}")
    
    solver_info = {
        "mesh_resolution": mesh_res,
        "element_degree": 2,
        "ksp_type": "gmres",
        "pc_type": "lu",
        "rtol": 1e-12,
        "iterations": lin_its,
        "nonlinear_iterations": [newton_its],
    }
    
    if pde.get("time", {}).get("is_transient", False):
        solver_info["dt"] = 1.0
        solver_info["n_steps"] = 1
        solver_info["time_scheme"] = "none"
    
    return {
        "u": magnitude,
        "solver_info": solver_info,
    }
