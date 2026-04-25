import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc as petsc_fem
import ufl
from petsc4py import PETSc

ScalarType = PETSc.ScalarType

def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD
    
    # Extract PDE parameters
    pde = case_spec["pde"]
    t0 = pde["time"]["t0"]
    t_end = pde["time"]["t_end"]
    dt_suggested = pde["time"]["dt"]
    kappa_expr = pde["coefficients"]["kappa"]["expr"]
    
    # Extract output grid
    out = case_spec["output"]
    grid = out["grid"]
    nx_out = grid["nx"]
    ny_out = grid["ny"]
    bbox = grid["bbox"]  # [xmin, xmax, ymin, ymax]
    
    # --- Select solver parameters ---
    mesh_res = 96
    element_degree = 2
    dt_use = 0.0025  # smaller than suggested for better accuracy
    ksp_type = "cg"
    pc_type = "hypre"
    rtol = 1e-10
    time_scheme = "backward_euler"
    
    n_steps = int(round((t_end - t0) / dt_use))
    dt_actual = (t_end - t0) / n_steps
    
    # --- Create mesh ---
    domain = mesh.create_unit_square(comm, mesh_res, mesh_res, cell_type=mesh.CellType.triangle)
    
    # --- Function space ---
    V = fem.functionspace(domain, ("Lagrange", element_degree))
    
    # --- Spatial coordinate for UFL expressions ---
    x = ufl.SpatialCoordinate(domain)
    
    # --- Variable coefficient kappa ---
    kappa_ufl = 1.0 + 30.0 * ufl.exp(-150.0 * ((x[0] - 0.35)**2 + (x[1] - 0.65)**2))
    
    # --- Manufactured solution: u = exp(-t)*sin(pi*x)*sin(2*pi*y) ---
    # Derive source term f = du/dt - div(kappa * grad(u))
    # We'll build this symbolically in UFL
    
    # Time constant (will be updated each step)
    t_const = fem.Constant(domain, ScalarType(t0))
    
    # Exact solution as UFL expression
    u_exact_ufl = ufl.exp(-t_const) * ufl.sin(ufl.pi * x[0]) * ufl.sin(2 * ufl.pi * x[1])
    
    # grad(u) 
    grad_u_exact = ufl.grad(u_exact_ufl)
    
    # div(kappa * grad(u))
    div_kappa_grad_u = ufl.div(kappa_ufl * grad_u_exact)
    
    # du/dt
    du_dt = -u_exact_ufl  # derivative of exp(-t) is -exp(-t)
    
    # Source term: f = du/dt - div(kappa * grad(u))
    f_ufl = du_dt - div_kappa_grad_u
    
    # --- Boundary condition: u = g on boundary ---
    # g = u_exact on boundary
    g_ufl = u_exact_ufl
    
    # --- Locate boundary DOFs ---
    tdim = domain.topology.dim
    fdim = tdim - 1
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
    )
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    
    # BC function
    u_bc_func = fem.Function(V)
    u_bc_func.interpolate(
        fem.Expression(g_ufl, V.element.interpolation_points)
    )
    bc = fem.dirichletbc(u_bc_func, boundary_dofs)
    
    # --- Initial condition ---
    u_n = fem.Function(V)
    u_n.interpolate(
        fem.Expression(u_exact_ufl, V.element.interpolation_points)
    )
    
    # Store initial condition for output
    u_initial_grid = None
    
    # --- Variational form (backward Euler) ---
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    
    f_expr = fem.Expression(f_ufl, V.element.interpolation_points)
    f_func = fem.Function(V)
    
    # a(u,v) = <u,v> + dt * <kappa * grad(u), grad(v)>
    a_form = ufl.inner(u, v) * ufl.dx + dt_actual * ufl.inner(kappa_ufl * ufl.grad(u), ufl.grad(v)) * ufl.dx
    
    # L(v) = <u_n, v> + dt * <f, v>
    L_form = ufl.inner(u_n, v) * ufl.dx + dt_actual * ufl.inner(f_func, v) * ufl.dx
    
    # Compile forms
    a_compiled = fem.form(a_form)
    L_compiled = fem.form(L_form)
    
    # --- Assemble matrix A (constant across time steps) ---
    A = petsc_fem.assemble_matrix(a_compiled, bcs=[bc])
    A.assemble()
    
    # --- Create solver ---
    solver = PETSc.KSP().create(comm)
    solver.setOperators(A)
    solver.setType(ksp_type)
    solver.getPC().setType(pc_type)
    solver.setTolerances(rtol=rtol, atol=1e-14, max_it=1000)
    solver.getPC().setHYPREType("boomeramg")
    
    # --- Solution function ---
    u_sol = fem.Function(V)
    
    # --- RHS vector ---
    b = petsc_fem.create_vector(L_compiled.function_spaces)
    
    total_iterations = 0
    
    # --- Sample initial condition on output grid ---
    u_initial_grid = _sample_function_on_grid(domain, u_n, nx_out, ny_out, bbox)
    
    # --- Time stepping ---
    t = t0
    for step in range(n_steps):
        t = t0 + (step + 1) * dt_actual
        t_const.value = ScalarType(t)
        
        # Update source term
        f_func.interpolate(f_expr)
        
        # Update BC
        u_bc_func.interpolate(
            fem.Expression(g_ufl, V.element.interpolation_points)
        )
        
        # Assemble RHS
        with b.localForm() as loc:
            loc.set(0)
        petsc_fem.assemble_vector(b, L_compiled)
        
        # Apply lifting
        petsc_fem.apply_lifting(b, [a_compiled], bcs=[[bc]])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        
        # Set BC on RHS
        petsc_fem.set_bc(b, [bc])
        
        # Solve
        solver.solve(b, u_sol.x.petsc_vec)
        u_sol.x.scatter_forward()
        
        # Count iterations
        total_iterations += solver.getIterationNumber()
        
        # Update previous solution
        u_n.x.array[:] = u_sol.x.array[:]
    
    # --- Sample solution on output grid ---
    u_grid = _sample_function_on_grid(domain, u_sol, nx_out, ny_out, bbox)
    
    # --- Verify accuracy ---
    # Build exact solution at final time for verification
    t_final = t_end
    exact_vals = np.zeros((ny_out, nx_out))
    xs = np.linspace(bbox[0], bbox[1], nx_out)
    ys = np.linspace(bbox[2], bbox[3], ny_out)
    for j in range(ny_out):
        for i in range(nx_out):
            exact_vals[j, i] = np.exp(-t_final) * np.sin(np.pi * xs[i]) * np.sin(2 * np.pi * ys[j])
    
    error = np.sqrt(np.mean((u_grid - exact_vals)**2))
    max_error = np.max(np.abs(u_grid - exact_vals))
    if comm.rank == 0:
        print(f"L2 error on grid: {error:.6e}, Max error: {max_error:.6e}")
        print(f"Mesh res: {mesh_res}, Element deg: {element_degree}, dt: {dt_actual}, Steps: {n_steps}")
        print(f"Total KSP iterations: {total_iterations}")
    
    return {
        "u": u_grid,
        "u_initial": u_initial_grid,
        "solver_info": {
            "mesh_resolution": mesh_res,
            "element_degree": element_degree,
            "ksp_type": ksp_type,
            "pc_type": pc_type,
            "rtol": rtol,
            "iterations": total_iterations,
            "dt": dt_actual,
            "n_steps": n_steps,
            "time_scheme": time_scheme,
        }
    }


def _sample_function_on_grid(domain, u_func, nx, ny, bbox):
    """Sample a dolfinx Function on a uniform grid, returning shape (ny, nx)."""
    comm = domain.comm
    xmin, xmax, ymin, ymax = bbox
    xs = np.linspace(xmin, xmax, nx)
    ys = np.linspace(ymin, ymax, ny)
    XX, YY = np.meshgrid(xs, ys)
    points = np.zeros((nx * ny, 3))
    points[:, 0] = XX.ravel()
    points[:, 1] = YY.ravel()
    
    bb_tree = geometry.bb_tree(domain, domain.topology.dim)
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
    
    u_values = np.full((points.shape[0],), np.nan)
    if len(points_on_proc) > 0:
        pts_arr = np.array(points_on_proc)
        cells_arr = np.array(cells_on_proc, dtype=np.int32)
        vals = u_func.eval(pts_arr, cells_arr)
        u_values[eval_map] = vals.flatten()
    
    # Gather across processes
    if comm.size > 1:
        from mpi4py import MPI as MPI4PY
        gathered = comm.allgather(u_values)
        u_global = np.full_like(u_values, np.nan)
        for arr in gathered:
            mask = ~np.isnan(arr)
            u_global[mask] = arr[mask]
        u_values = u_global
    
    return u_values.reshape((ny, nx))
