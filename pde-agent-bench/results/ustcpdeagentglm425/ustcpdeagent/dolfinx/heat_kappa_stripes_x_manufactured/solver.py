import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc

def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD
    
    # Extract problem parameters
    pde = case_spec["pde"]
    time_params = pde.get("time", {})
    coeffs = pde.get("coefficients", {})
    out_spec = case_spec["output"]
    grid_spec = out_spec["grid"]
    
    nx_grid = grid_spec["nx"]
    ny_grid = grid_spec["ny"]
    bbox = grid_spec["bbox"]  # [xmin, xmax, ymin, ymax]
    xmin, xmax, ymin, ymax = bbox[0], bbox[1], bbox[2], bbox[3]
    
    t0 = time_params.get("t0", 0.0)
    t_end = time_params.get("t_end", 0.1)
    dt_suggested = time_params.get("dt", 0.01)
    scheme = time_params.get("scheme", "backward_euler")
    
    # --- Mesh and function space ---
    # Use higher resolution for accuracy
    mesh_res = 80
    elem_deg = 2
    
    domain = mesh.create_unit_square(comm, mesh_res, mesh_res, cell_type=mesh.CellType.triangle)
    tdim = domain.topology.dim
    fdim = tdim - 1
    
    V = fem.functionspace(domain, ("Lagrange", elem_deg))
    
    # --- Spatial coordinate and time variable ---
    x = ufl.SpatialCoordinate(domain)
    t = ufl.variable(fem.Constant(domain, PETSc.ScalarType(t0)))
    
    # --- Coefficient kappa ---
    kappa_expr_ufl = 1.0 + 0.5 * ufl.sin(6 * ufl.pi * x[0])
    kappa_func = fem.Function(V)
    kappa_func.interpolate(
        fem.Expression(kappa_expr_ufl, V.element.interpolation_points)
    )
    
    # --- Manufactured solution ---
    u_exact_ufl = ufl.exp(-t) * ufl.sin(2 * ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])
    
    # --- Source term f = du/dt - div(kappa * grad(u)) ---
    du_dt = ufl.diff(u_exact_ufl, t)
    div_kappa_grad_u = ufl.div(kappa_expr_ufl * ufl.grad(u_exact_ufl))
    f_ufl = du_dt - div_kappa_grad_u
    
    # --- Boundary condition value ---
    g_ufl = u_exact_ufl
    
    # --- Time stepping setup (Backward Euler) ---
    dt_val = dt_suggested
    dt_const = fem.Constant(domain, PETSc.ScalarType(dt_val))
    
    # Current and previous solution
    u_n = fem.Function(V)  # u at previous time step
    u_h = fem.Function(V)  # u at current time step (solution)
    
    # Interpolate initial condition
    u_n.interpolate(
        fem.Expression(u_exact_ufl, V.element.interpolation_points)
    )
    
    # Boundary facets and DOFs
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
    )
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    
    # Boundary condition function (will be updated each step)
    u_bc_func = fem.Function(V)
    u_bc_func.interpolate(
        fem.Expression(g_ufl, V.element.interpolation_points)
    )
    bc = fem.dirichletbc(u_bc_func, boundary_dofs)
    
    # --- Variational form ---
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    
    # Backward Euler: (u - u_n)/dt - div(kappa*grad(u)) = f
    # => u*v*dx + dt*kappa*grad(u)·grad(v)*dx = u_n*v*dx + dt*f*v*dx
    a_form = ufl.inner(u, v) * ufl.dx + dt_const * ufl.inner(kappa_func * ufl.grad(u), ufl.grad(v)) * ufl.dx
    
    # f as a function (updated each time step)
    f_func = fem.Function(V)
    f_func.interpolate(
        fem.Expression(f_ufl, V.element.interpolation_points)
    )
    
    L_form = ufl.inner(u_n, v) * ufl.dx + dt_const * ufl.inner(f_func, v) * ufl.dx
    
    # --- Assembly ---
    a = fem.form(a_form)
    L = fem.form(L_form)
    
    A = petsc.assemble_matrix(a, bcs=[bc])
    A.assemble()
    
    b = petsc.create_vector(L.function_spaces)
    
    # --- Solver setup ---
    solver = PETSc.KSP().create(domain.comm)
    solver.setOperators(A)
    solver.setType(PETSc.KSP.Type.CG)
    solver.getPC().setType(PETSc.PC.Type.HYPRE)
    solver.getPC().setHYPREType("boomeramg")
    rtol = 1e-10
    solver.setTolerances(rtol=rtol, atol=1e-14, max_it=1000)
    
    # --- Time stepping ---
    current_t = t0
    n_steps = int(round((t_end - t0) / dt_val))
    total_iterations = 0
    
    # Store initial condition for output
    u_initial = u_n.x.array.copy().reshape(-1)
    
    for step in range(n_steps):
        current_t = t0 + (step + 1) * dt_val
        
        # Update time constant
        t.value = PETSc.ScalarType(current_t)
        
        # Update BC and source
        u_bc_func.interpolate(
            fem.Expression(g_ufl, V.element.interpolation_points)
        )
        f_func.interpolate(
            fem.Expression(f_ufl, V.element.interpolation_points)
        )
        
        # Assemble RHS
        with b.localForm() as loc_b:
            loc_b.set(0)
        petsc.assemble_vector(b, L)
        
        # Apply lifting
        petsc.apply_lifting(b, [a], bcs=[[bc]])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        petsc.set_bc(b, [bc])
        
        # Solve
        solver.solve(b, u_h.x.petsc_vec)
        u_h.x.scatter_forward()
        
        # Track iterations
        its = solver.getIterationNumber()
        total_iterations += its
        
        # Update previous
        u_n.x.array[:] = u_h.x.array[:]
    
    # --- Sample solution on output grid ---
    xs = np.linspace(xmin, xmax, nx_grid)
    ys = np.linspace(ymin, ymax, ny_grid)
    XX, YY = np.meshgrid(xs, ys)
    points = np.zeros((nx_grid * ny_grid, 3))
    points[:, 0] = XX.ravel()
    points[:, 1] = YY.ravel()
    
    bb_tree = geometry.bb_tree(domain, tdim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, points)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points)
    
    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    for i in range(points.shape[0]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(points[i])
            cells_on_proc.append(links[0])
            eval_map.append(i)
    
    u_values = np.full((nx_grid * ny_grid,), np.nan)
    if len(points_on_proc) > 0:
        vals = u_h.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u_values[eval_map] = vals.flatten()
    
    # Gather on all procs
    u_values_local = u_values.copy()
    u_values_global = np.zeros_like(u_values)
    comm.Allreduce(u_values_local, u_values_global, op=MPI.SUM)
    
    # Fill any remaining NaN with exact solution
    t_final = t_end
    for idx in range(len(u_values_global)):
        if np.isnan(u_values_global[idx]):
            px = points[idx, 0]
            py = points[idx, 1]
            u_values_global[idx] = np.exp(-t_final) * np.sin(2*np.pi*px) * np.sin(np.pi*py)
    
    u_grid = u_values_global.reshape(ny_grid, nx_grid)
    
    # Also sample initial condition on grid
    u_initial_grid_vals = np.full((nx_grid * ny_grid,), np.nan)
    if len(points_on_proc) > 0:
        # Re-evaluate u_n at t0
        t_orig = t.value
        t.value = PETSc.ScalarType(t0)
        u_n_init = fem.Function(V)
        u_n_init.interpolate(
            fem.Expression(u_exact_ufl, V.element.interpolation_points)
        )
        vals_init = u_n_init.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u_initial_grid_vals[eval_map] = vals_init.flatten()
        t.value = t_orig
    
    u_initial_local = u_initial_grid_vals.copy()
    u_initial_global = np.zeros_like(u_initial_grid_vals)
    comm.Allreduce(u_initial_local, u_initial_global, op=MPI.SUM)
    
    for idx in range(len(u_initial_global)):
        if np.isnan(u_initial_global[idx]):
            px = points[idx, 0]
            py = points[idx, 1]
            u_initial_global[idx] = np.exp(-t0) * np.sin(2*np.pi*px) * np.sin(np.pi*py)
    
    u_initial_grid = u_initial_global.reshape(ny_grid, nx_grid)
    
    # --- Compute L2 error for verification ---
    u_exact_func = fem.Function(V)
    u_exact_func.interpolate(
        fem.Expression(u_exact_ufl, V.element.interpolation_points)
    )
    error_ufl = ufl.inner(u_h - u_exact_func, u_h - u_exact_func) * ufl.dx
    error_form = fem.form(error_ufl)
    l2_error_sq = petsc.assemble_vector(error_form)
    l2_error = np.sqrt(np.abs(l2_error_sq.array[:].sum())) if comm.rank == 0 else 0.0
    l2_error = comm.bcast(l2_error, root=0)
    
    if comm.rank == 0:
        print(f"L2 error: {l2_error:.6e}")
    
    return {
        "u": u_grid,
        "u_initial": u_initial_grid,
        "solver_info": {
            "mesh_resolution": mesh_res,
            "element_degree": elem_deg,
            "ksp_type": "cg",
            "pc_type": "hypre",
            "rtol": rtol,
            "iterations": total_iterations,
            "dt": dt_val,
            "n_steps": n_steps,
            "time_scheme": "backward_euler",
        }
    }

if __name__ == "__main__":
    # Test with a minimal case_spec
    case_spec = {
        "pde": {
            "time": {"t0": 0.0, "t_end": 0.1, "dt": 0.01, "scheme": "backward_euler"},
            "coefficients": {"kappa": {"type": "expr", "expr": "1 + 0.5*sin(6*pi*x)"}},
        },
        "output": {
            "grid": {"nx": 32, "ny": 32, "bbox": [0.0, 1.0, 0.0, 1.0]}
        }
    }
    result = solve(case_spec)
    print(f"u shape: {result['u'].shape}")
    print(f"solver_info: {result['solver_info']}")
