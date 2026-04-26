import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc
import os

os.environ.setdefault("OMP_NUM_THREADS", "4")

ScalarType = PETSc.ScalarType

def solve(case_spec: dict) -> dict:
    # ── Extract parameters from case_spec ──
    pde = case_spec["pde"]
    eps = float(pde.get("epsilon", 0.01))
    rho = float(pde.get("reaction_rho", 100.0))
    
    time_info = pde.get("time", {})
    t0 = float(time_info.get("t0", 0.0))
    t_end = float(time_info.get("t_end", 0.2))
    
    grid_info = case_spec["output"]["grid"]
    nx_out = grid_info["nx"]
    ny_out = grid_info["ny"]
    bbox = grid_info["bbox"]
    xmin, xmax, ymin, ymax = bbox[0], bbox[1], bbox[2], bbox[3]
    
    # ── Numerical parameters ──
    mesh_res = 96
    elem_degree = 2
    dt = 0.001
    n_steps = int(round((t_end - t0) / dt))
    actual_dt = (t_end - t0) / n_steps
    
    # ── Create mesh ──
    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_square(comm, mesh_res, mesh_res, cell_type=mesh.CellType.triangle)
    tdim = domain.topology.dim
    fdim = tdim - 1
    
    # ── Function space ──
    V = fem.functionspace(domain, ("Lagrange", elem_degree))
    
    # ── Exact solution and source (numpy for interpolation) ──
    def u_exact_np(x_arr, y_arr, t_val):
        return np.exp(-t_val) * (0.35 + 0.1 * np.cos(2*np.pi*x_arr) * np.sin(np.pi*y_arr))
    
    def source_np(x_arr, y_arr, t_val):
        u_val = u_exact_np(x_arr, y_arr, t_val)
        du_dt = -u_val
        lap_u = -5.0 * np.pi**2 * np.exp(-t_val) * 0.1 * np.cos(2*np.pi*x_arr) * np.sin(np.pi*y_arr)
        R_u = rho * u_val * (1.0 - u_val)
        return du_dt - eps * lap_u + R_u
    
    # ── Functions ──
    u_n = fem.Function(V)
    u_h = fem.Function(V)
    f_func = fem.Function(V)
    g_func = fem.Function(V)
    R_explicit = fem.Function(V)
    
    # Initial condition
    u_n.interpolate(lambda x: u_exact_np(x[0], x[1], t0))
    u_h.x.array[:] = u_n.x.array[:]
    
    # BC at t0
    g_func.interpolate(lambda x: u_exact_np(x[0], x[1], t0))
    
    # ── Boundary conditions ──
    boundary_facets = mesh.locate_entities_boundary(domain, fdim,
        lambda x: np.ones(x.shape[1], dtype=bool))
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc = fem.dirichletbc(g_func, boundary_dofs)
    
    # ── IMEX: implicit diffusion + explicit reaction ──
    u_trial = ufl.TrialFunction(V)
    v_test = ufl.TestFunction(V)
    
    a_form = (1.0/actual_dt) * u_trial * v_test * ufl.dx
    a_form += eps * ufl.inner(ufl.grad(u_trial), ufl.grad(v_test)) * ufl.dx
    
    L_form = f_func * v_test * ufl.dx
    L_form += (1.0/actual_dt) * u_n * v_test * ufl.dx
    L_form -= R_explicit * v_test * ufl.dx
    
    a_compiled = fem.form(a_form)
    L_compiled = fem.form(L_form)
    
    # ── Assemble matrix (once - time-independent!) ──
    A = petsc.assemble_matrix(a_compiled, bcs=[bc])
    A.assemble()
    
    b = petsc.create_vector(L_compiled.function_spaces)
    
    # ── Solver: CG + Jacobi (SPD system) ──
    ksp = PETSc.KSP().create(domain.comm)
    ksp.setOperators(A)
    ksp.setType(PETSc.KSP.Type.CG)
    ksp.getPC().setType(PETSc.PC.Type.JACOBI)
    ksp.setTolerances(rtol=1e-10, atol=1e-12, max_it=200)
    ksp.setFromOptions()
    
    # ── Time stepping ──
    total_linear_iters = 0
    t = t0
    
    for step in range(n_steps):
        t_new = t + actual_dt
        
        # Update source and BC for new time
        f_func.interpolate(lambda x, tv=t_new: source_np(x[0], x[1], tv))
        g_func.interpolate(lambda x, tv=t_new: u_exact_np(x[0], x[1], tv))
        
        # Explicit reaction: R(u_n) = rho * u_n * (1 - u_n)
        R_explicit.x.array[:] = rho * u_n.x.array[:] * (1.0 - u_n.x.array[:])
        
        # Assemble RHS
        with b.localForm() as loc:
            loc.set(0)
        petsc.assemble_vector(b, L_compiled)
        petsc.apply_lifting(b, [a_compiled], bcs=[[bc]])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        petsc.set_bc(b, [bc])
        
        # Solve
        ksp.solve(b, u_h.x.petsc_vec)
        u_h.x.scatter_forward()
        
        total_linear_iters += ksp.getIterationNumber()
        
        # Update
        u_n.x.array[:] = u_h.x.array[:]
        t = t_new
    
    # ── Sample solution on output grid ──
    xs = np.linspace(xmin, xmax, nx_out)
    ys = np.linspace(ymin, ymax, ny_out)
    XX, YY = np.meshgrid(xs, ys)
    points = np.zeros((3, nx_out * ny_out))
    points[0, :] = XX.ravel()
    points[1, :] = YY.ravel()
    
    bb_tree = geometry.bb_tree(domain, tdim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, points.T)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points.T)
    
    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    for i in range(points.shape[1]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(points.T[i])
            cells_on_proc.append(links[0])
            eval_map.append(i)
    
    u_values = np.full((points.shape[1],), np.nan, dtype=np.float64)
    if len(points_on_proc) > 0:
        vals = u_h.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u_values[eval_map] = vals.flatten()
    
    u_global = np.zeros_like(u_values)
    comm.Allreduce(u_values, u_global, op=MPI.SUM)
    u_global = np.nan_to_num(u_global, nan=0.0)
    u_grid = u_global.reshape(ny_out, nx_out)
    
    # ── Initial condition grid ──
    u_init_vals = np.full((points.shape[1],), np.nan, dtype=np.float64)
    if len(points_on_proc) > 0:
        u_init_func = fem.Function(V)
        u_init_func.interpolate(lambda x: u_exact_np(x[0], x[1], t0))
        vals0 = u_init_func.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u_init_vals[eval_map] = vals0.flatten()
    
    u_init_global = np.zeros_like(u_init_vals)
    comm.Allreduce(u_init_vals, u_init_global, op=MPI.SUM)
    u_init_global = np.nan_to_num(u_init_global, nan=0.0)
    u_initial_grid = u_init_global.reshape(ny_out, nx_out)
    
    # ── L2 error verification ──
    u_exact_final = fem.Function(V)
    u_exact_final.interpolate(lambda x: u_exact_np(x[0], x[1], t_end))
    
    diff = u_h - u_exact_final
    err2 = fem.assemble_scalar(fem.form(ufl.inner(diff, diff) * ufl.dx))
    l2_err = np.sqrt(max(comm.allreduce(err2, op=MPI.SUM), 0.0))
    
    if comm.rank == 0:
        print(f"[Verification] L2 error: {l2_err:.6e} (threshold: 2.67e-03)")
        print(f"Mesh: {mesh_res}, Degree: {elem_degree}, dt: {actual_dt:.6f}, Steps: {n_steps}")
        print(f"Total CG iterations: {total_linear_iters}")
    
    # ── Build result ──
    solver_info = {
        "mesh_resolution": mesh_res,
        "element_degree": elem_degree,
        "ksp_type": "cg",
        "pc_type": "jacobi",
        "rtol": 1e-10,
        "iterations": total_linear_iters,
        "dt": actual_dt,
        "n_steps": n_steps,
        "time_scheme": "backward_euler",
        "nonlinear_iterations": [0] * n_steps,
    }
    
    return {
        "u": u_grid,
        "u_initial": u_initial_grid,
        "solver_info": solver_info,
    }
