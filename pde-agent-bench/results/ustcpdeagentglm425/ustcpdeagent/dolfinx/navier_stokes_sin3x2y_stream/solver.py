import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc as fem_petsc
from basix.ufl import element as basix_element, mixed_element as basix_mixed_element
import ufl
from petsc4py import PETSc

def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD
    
    # Parse case_spec
    pde = case_spec["pde"]
    nu = float(pde["viscosity"])
    output = case_spec["output"]
    grid_spec = output["grid"]
    nx_out = grid_spec["nx"]
    ny_out = grid_spec["ny"]
    bbox = grid_spec["bbox"]
    xmin, xmax, ymin, ymax = bbox[0], bbox[1], bbox[2], bbox[3]
    
    # Mesh resolution
    N = 160
    
    # Create mesh
    msh = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
    gdim = msh.geometry.dim
    tdim = msh.topology.dim
    fdim = tdim - 1
    
    # Taylor-Hood P2/P1 mixed element
    vel_el = basix_element("Lagrange", msh.topology.cell_name(), 2, shape=(gdim,))
    pres_el = basix_element("Lagrange", msh.topology.cell_name(), 1)
    W = fem.functionspace(msh, basix_mixed_element([vel_el, pres_el]))
    
    V, _ = W.sub(0).collapse()
    Q, _ = W.sub(1).collapse()
    
    # Define exact solution using UFL
    x = ufl.SpatialCoordinate(msh)
    pi = ufl.pi
    
    u_ex_1 = 2*pi*ufl.cos(2*pi*x[1])*ufl.sin(3*pi*x[0])
    u_ex_2 = -3*pi*ufl.cos(3*pi*x[0])*ufl.sin(2*pi*x[1])
    p_ex = ufl.cos(pi*x[0])*ufl.cos(2*pi*x[1])
    
    # Source terms
    conv_term = ufl.grad(ufl.as_vector([u_ex_1, u_ex_2])) * ufl.as_vector([u_ex_1, u_ex_2])
    lap_u_ex = ufl.div(ufl.grad(ufl.as_vector([u_ex_1, u_ex_2])))
    grad_p_ex = ufl.grad(p_ex)
    
    f_full = conv_term - nu * lap_u_ex + grad_p_ex
    f_stokes = -nu * lap_u_ex + grad_p_ex
    
    # Velocity BC on all boundaries
    boundary_facets = mesh.locate_entities_boundary(
        msh, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
    )
    
    u_bc_func = fem.Function(V)
    u_bc_func.interpolate(
        fem.Expression(ufl.as_vector([u_ex_1, u_ex_2]), V.element.interpolation_points)
    )
    
    dofs_u = fem.locate_dofs_topological((W.sub(0), V), fdim, boundary_facets)
    bc_u = fem.dirichletbc(u_bc_func, dofs_u, W.sub(0))
    
    # Pressure pin at origin corner - pin to exact value p(0,0) = 1
    p_dofs = fem.locate_dofs_geometrical(
        (W.sub(1), Q),
        lambda x: np.isclose(x[0], 0.0) & np.isclose(x[1], 0.0),
    )
    bcs = [bc_u]
    if len(p_dofs) > 0:
        p0_func = fem.Function(Q)
        p0_func.interpolate(fem.Expression(p_ex, Q.element.interpolation_points))
        bc_p = fem.dirichletbc(p0_func, p_dofs, W.sub(1))
        bcs.append(bc_p)
    
    # ============ Step 1: Solve Stokes for initial guess ============
    (u_s, p_s) = ufl.TrialFunctions(W)
    (v, q) = ufl.TestFunctions(W)
    
    a_stokes = (
        nu * ufl.inner(ufl.grad(u_s), ufl.grad(v)) * ufl.dx
        - p_s * ufl.div(v) * ufl.dx
        + ufl.div(u_s) * q * ufl.dx
    )
    L_stokes = ufl.inner(f_stokes, v) * ufl.dx
    
    stokes_problem = fem_petsc.LinearProblem(
        a_stokes, L_stokes, bcs=bcs,
        petsc_options_prefix="stokes_",
        petsc_options={
            "ksp_type": "preonly",
            "pc_type": "lu",
            "pc_factor_mat_solver_type": "mumps",
        }
    )
    
    w_stokes = stokes_problem.solve()
    w_stokes.x.scatter_forward()
    
    # ============ Step 2: Solve Navier-Stokes with Newton ============
    w = fem.Function(W)
    w.x.array[:] = w_stokes.x.array[:]
    w.x.scatter_forward()
    
    (u, p) = ufl.split(w)
    (v, q) = ufl.TestFunctions(W)
    
    # Weak form using Laplacian formulation for viscous term
    F = (
        nu * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
        + ufl.inner(ufl.grad(u) * u, v) * ufl.dx
        - p * ufl.div(v) * ufl.dx
        - ufl.inner(f_full, v) * ufl.dx
        + ufl.div(u) * q * ufl.dx
    )
    
    J = ufl.derivative(F, w)
    
    newton_rtol = 1e-8
    newton_max_it = 30
    
    petsc_options_ns = {
        "snes_type": "newtonls",
        "snes_linesearch_type": "bt",
        "snes_rtol": str(newton_rtol),
        "snes_atol": "1e-10",
        "snes_max_it": str(newton_max_it),
        "snes_monitor": "",
        "ksp_type": "gmres",
        "pc_type": "lu",
        "pc_factor_mat_solver_type": "mumps",
        "ksp_rtol": "1e-8",
        "ksp_atol": "1e-12",
    }
    
    ns_problem = fem_petsc.NonlinearProblem(
        F, w, bcs=bcs, J=J,
        petsc_options_prefix="ns_",
        petsc_options=petsc_options_ns
    )
    
    w_h = ns_problem.solve()
    w.x.scatter_forward()
    
    # Get iteration info
    snes = ns_problem._snes
    newton_iterations = snes.getIterationNumber()
    total_ksp_iterations = snes.getLinearSolveIterations()
    
    # Extract velocity
    u_h = w.sub(0).collapse()
    
    # Compute L2 error for verification
    u_l2_sq = fem.assemble_scalar(
        fem.form(ufl.inner(u_h - ufl.as_vector([u_ex_1, u_ex_2]), u_h - ufl.as_vector([u_ex_1, u_ex_2])) * ufl.dx)
    )
    u_l2 = np.sqrt(comm.allreduce(float(u_l2_sq), op=MPI.SUM))
    
    if comm.rank == 0:
        print(f"Velocity L2 error: {u_l2:.6e}")
        print(f"Newton iterations: {newton_iterations}")
        print(f"Total KSP iterations: {total_ksp_iterations}")
    
    # Compute velocity magnitude on FEM space
    V_mag = fem.functionspace(msh, ("Lagrange", 2))
    u_mag = fem.Function(V_mag)
    u_mag.interpolate(
        fem.Expression(ufl.sqrt(ufl.inner(u_h, u_h)), V_mag.element.interpolation_points)
    )
    
    # Sample on output grid
    xs = np.linspace(xmin, xmax, nx_out)
    ys = np.linspace(ymin, ymax, ny_out)
    XX, YY = np.meshgrid(xs, ys)
    pts = np.zeros((nx_out * ny_out, 3))
    pts[:, 0] = XX.ravel()
    pts[:, 1] = YY.ravel()
    
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
    
    u_values = np.full((pts.shape[0],), np.nan)
    if len(points_on_proc) > 0:
        vals = u_mag.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u_values[eval_map] = vals.flatten()
    
    # Gather on all ranks
    if comm.size > 1:
        all_values = np.zeros_like(u_values)
        comm.Allreduce(u_values, all_values, op=MPI.SUM)
        u_values = all_values
    
    u_grid = u_values.reshape(ny_out, nx_out)
    
    # Build solver_info
    solver_info = {
        "mesh_resolution": N,
        "element_degree": 2,
        "ksp_type": "gmres",
        "pc_type": "lu",
        "rtol": float(newton_rtol),
        "iterations": int(total_ksp_iterations),
        "nonlinear_iterations": [int(newton_iterations)],
        "dt": 0.0,
        "n_steps": 0,
        "time_scheme": "steady",
    }
    
    result = {
        "u": u_grid,
        "solver_info": solver_info,
    }
    
    if comm.rank == 0:
        print(f"Output grid shape: {u_grid.shape}")
        print(f"Max velocity magnitude: {np.nanmax(u_grid):.6e}")
        print(f"Accuracy check - Velocity L2 error: {u_l2:.6e} (threshold: 6.77)")
    
    return result
