import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc
import time as time_module

ScalarType = PETSc.ScalarType


def solve(case_spec: dict) -> dict:
    t_start = time_module.time()
    
    comm = MPI.COMM_WORLD
    
    # Parse case_spec
    pde = case_spec.get("pde", {})
    time_params = pde.get("time", {})
    output_spec = case_spec.get("output", {})
    grid_spec = output_spec.get("grid", {})
    
    nx_out = grid_spec.get("nx", 50)
    ny_out = grid_spec.get("ny", 50)
    bbox = grid_spec.get("bbox", [0, 1, 0, 1])
    xmin, xmax, ymin, ymax = bbox
    
    # Time parameters
    is_transient = time_params is not None and len(time_params) > 0
    t0 = time_params.get("t0", 0.0) if is_transient else 0.0
    t_end = time_params.get("t_end", 0.3) if is_transient else 0.3
    dt_suggested = time_params.get("dt", 0.005) if is_transient else 0.005
    scheme = time_params.get("scheme", "crank_nicolson") if is_transient else "crank_nicolson"
    
    if not is_transient:
        is_transient = True
    
    # Diffusion coefficient
    coefficients = pde.get("coefficients", {})
    epsilon = coefficients.get("epsilon", 1.0)
    if isinstance(epsilon, dict):
        epsilon = epsilon.get("value", 1.0)
    
    # Reaction coefficient
    reaction_coeff = 1.0
    reaction_type = "linear"
    
    reaction = pde.get("reaction", {})
    if isinstance(reaction, dict):
        reaction_type = reaction.get("type", "linear")
        reaction_coeff = reaction.get("coefficient", 1.0)
    elif reaction is not None:
        try:
            reaction_coeff = float(reaction)
        except (TypeError, ValueError):
            pass
    
    if "reaction" in coefficients:
        r_val = coefficients["reaction"]
        if isinstance(r_val, dict):
            reaction_coeff = r_val.get("value", reaction_coeff)
        else:
            try:
                reaction_coeff = float(r_val)
            except (TypeError, ValueError):
                pass
    
    # Solver parameters
    mesh_res = 120
    elem_degree = 2
    dt = dt_suggested
    
    # Theta for time scheme
    if scheme == "crank_nicolson":
        theta = 0.5
    elif scheme == "backward_euler":
        theta = 1.0
    else:
        theta = 0.5
    
    # Create mesh
    domain = mesh.create_rectangle(
        comm,
        [np.array([xmin, ymin]), np.array([xmax, ymax])],
        [mesh_res, mesh_res],
        cell_type=mesh.CellType.triangle
    )
    
    # Function space
    V = fem.functionspace(domain, ("Lagrange", elem_degree))
    
    # Trial and test functions
    u_trial = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    
    # Solution functions
    u_n = fem.Function(V, name="u_n")
    u_h = fem.Function(V, name="u_h")
    
    # Source term functions (interpolated each time step)
    f_new_func = fem.Function(V, name="f_new")
    f_old_func = fem.Function(V, name="f_old")
    
    # Constants
    dt_const = fem.Constant(domain, ScalarType(dt))
    theta_c = ScalarType(theta)
    one_m_theta = ScalarType(1.0 - theta)
    
    # Source term coefficient:
    # u = exp(-t)*sin(4*pi*x)*sin(3*pi*y)
    # du/dt = -u
    # nabla^2(u) = -25*pi^2*u
    # -eps*nabla^2(u) = 25*eps*pi^2*u
    # R(u) = reaction_coeff*u
    # f = du/dt - eps*nabla^2(u) + R(u) = (-1 + 25*eps*pi^2 + reaction_coeff)*u
    coeff_f_val = -1.0 + 25.0 * epsilon * np.pi**2 + reaction_coeff
    
    # Bilinear form (LHS)
    a_form = (u_trial * v / dt_const) * ufl.dx \
           + theta_c * epsilon * ufl.inner(ufl.grad(u_trial), ufl.grad(v)) * ufl.dx \
           + theta_c * reaction_coeff * u_trial * v * ufl.dx
    
    # Linear form (RHS)
    L_form = (u_n * v / dt_const) * ufl.dx \
           - one_m_theta * epsilon * ufl.inner(ufl.grad(u_n), ufl.grad(v)) * ufl.dx \
           - one_m_theta * reaction_coeff * u_n * v * ufl.dx \
           + theta_c * f_new_func * v * ufl.dx \
           + one_m_theta * f_old_func * v * ufl.dx
    
    # Boundary conditions (homogeneous Dirichlet)
    tdim = domain.topology.dim
    fdim = tdim - 1
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x_arr: np.ones(x_arr.shape[1], dtype=bool)
    )
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    u_bc = fem.Function(V)
    u_bc.x.array[:] = 0.0
    bc = fem.dirichletbc(u_bc, boundary_dofs)
    bcs = [bc]
    
    # Initial condition
    u_n.interpolate(lambda x: np.exp(-t0) * np.sin(4*np.pi*x[0]) * np.sin(3*np.pi*x[1]))
    u_h.x.array[:] = u_n.x.array[:]
    
    # Compile forms
    a_compiled = fem.form(a_form)
    L_compiled = fem.form(L_form)
    
    # Assemble matrix (constant for linear problem)
    A = petsc.assemble_matrix(a_compiled, bcs=bcs)
    A.assemble()
    
    # Create RHS vector
    b = petsc.create_vector(V)
    
    # Setup KSP solver - use direct LU for robustness
    ksp_type = "preonly"
    pc_type = "lu"
    rtol = 1e-10
    
    ksp = PETSc.KSP().create(domain.comm)
    ksp.setOperators(A)
    ksp.setType(PETSc.KSP.Type.PREONLY)
    pc = ksp.getPC()
    pc.setType(PETSc.PC.Type.LU)
    ksp.setUp()
    
    # Time stepping
    n_steps = int(round((t_end - t0) / dt))
    actual_dt = (t_end - t0) / n_steps
    dt_const.value = actual_dt
    
    total_iterations = 0
    current_t = t0
    
    for step in range(n_steps):
        ct_old = current_t
        current_t += actual_dt
        ct_new = current_t
        
        # Interpolate source terms
        f_new_func.interpolate(lambda x, t=ct_new: coeff_f_val * np.exp(-t) * np.sin(4*np.pi*x[0]) * np.sin(3*np.pi*x[1]))
        f_old_func.interpolate(lambda x, t=ct_old: coeff_f_val * np.exp(-t) * np.sin(4*np.pi*x[0]) * np.sin(3*np.pi*x[1]))
        
        # Assemble RHS
        with b.localForm() as loc:
            loc.set(0.0)
        petsc.assemble_vector(b, L_compiled)
        petsc.apply_lifting(b, [a_compiled], bcs=[bcs])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        petsc.set_bc(b, bcs)
        
        # Solve
        ksp.solve(b, u_h.x.petsc_vec)
        u_h.x.scatter_forward()
        
        total_iterations += 1  # Direct solver = 1 iteration per step
        
        # Update
        u_n.x.array[:] = u_h.x.array[:]
    
    # Sample solution onto output grid
    xs = np.linspace(xmin, xmax, nx_out)
    ys = np.linspace(ymin, ymax, ny_out)
    XX, YY = np.meshgrid(xs, ys)
    pts_3d = np.zeros((nx_out * ny_out, 3))
    pts_3d[:, 0] = XX.ravel()
    pts_3d[:, 1] = YY.ravel()
    
    bb_tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, pts_3d)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, pts_3d)
    
    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    for i in range(len(pts_3d)):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(pts_3d[i])
            cells_on_proc.append(links[0])
            eval_map.append(i)
    
    pts_arr = np.array(points_on_proc) if points_on_proc else np.empty((0, 3))
    cells_arr = np.array(cells_on_proc, dtype=np.int32) if cells_on_proc else np.empty(0, dtype=np.int32)
    
    def sample_on_grid(func):
        vals_out = np.full(nx_out * ny_out, np.nan)
        if len(pts_arr) > 0:
            vals = func.eval(pts_arr, cells_arr)
            vals_out[eval_map] = vals.flatten()
        return vals_out.reshape(ny_out, nx_out)
    
    u_grid = sample_on_grid(u_h)
    
    # Sample initial condition
    u_init_func = fem.Function(V)
    u_init_func.interpolate(lambda x: np.exp(-t0) * np.sin(4*np.pi*x[0]) * np.sin(3*np.pi*x[1]))
    u_initial_grid = sample_on_grid(u_init_func)
    
    # Accuracy verification
    exact_grid = np.exp(-current_t) * np.sin(4 * np.pi * XX) * np.sin(3 * np.pi * YY)
    grid_rms = np.sqrt(np.nanmean((u_grid - exact_grid)**2))
    grid_max = np.nanmax(np.abs(u_grid - exact_grid))
    
    elapsed = time_module.time() - t_start
    print(f"[solver] mesh_res={mesh_res}, degree={elem_degree}, dt={actual_dt}, n_steps={n_steps}")
    print(f"[solver] Grid RMS error = {grid_rms:.6e}, Max error = {grid_max:.6e}")
    print(f"[solver] Total iterations = {total_iterations}")
    print(f"[solver] Wall time = {elapsed:.2f}s")
    
    # Cleanup
    ksp.destroy()
    A.destroy()
    b.destroy()
    
    result = {
        "u": u_grid,
        "u_initial": u_initial_grid,
        "solver_info": {
            "mesh_resolution": mesh_res,
            "element_degree": elem_degree,
            "ksp_type": ksp_type,
            "pc_type": pc_type,
            "rtol": rtol,
            "iterations": total_iterations,
            "dt": actual_dt,
            "n_steps": n_steps,
            "time_scheme": scheme,
        }
    }
    
    return result
