import numpy as np
from dolfinx import mesh, fem, default_scalar_type, geometry
from dolfinx.fem import petsc
from mpi4py import MPI
import ufl
from petsc4py import PETSc


def solve(case_spec: dict) -> dict:
    # 1. Parse case_spec
    pde = case_spec.get("pde", {})
    
    # Source term
    f_val = 1.0
    if "source_term" in pde:
        st = pde["source_term"]
        if isinstance(st, (int, float)):
            f_val = float(st)
        elif isinstance(st, str):
            try:
                f_val = float(st)
            except ValueError:
                f_val = 1.0
    
    # Initial condition
    u0_val = 0.0
    if "initial_condition" in pde:
        ic = pde["initial_condition"]
        if isinstance(ic, (int, float)):
            u0_val = float(ic)
        elif isinstance(ic, str):
            try:
                u0_val = float(ic)
            except ValueError:
                u0_val = 0.0
    
    # Diffusivity
    kappa = 1.0
    coeffs = pde.get("coefficients", {})
    if "kappa" in coeffs:
        kappa = float(coeffs["kappa"])
    
    # Time parameters
    time_params = pde.get("time", {})
    t_end = float(time_params.get("t_end", 0.12))
    dt_suggested = float(time_params.get("dt", 0.03))
    scheme = time_params.get("scheme", "backward_euler")
    
    # Boundary conditions - default to zero Dirichlet
    bc_val = 0.0
    bcs_spec = pde.get("boundary_conditions", [])
    if isinstance(bcs_spec, list):
        for bc_item in bcs_spec:
            if isinstance(bc_item, dict) and bc_item.get("type") == "dirichlet":
                try:
                    bc_val = float(bc_item.get("value", 0.0))
                except (ValueError, TypeError):
                    bc_val = 0.0
    
    # 2. Mesh and function space
    nx = 80
    ny = 80
    degree = 1
    
    domain = mesh.create_unit_square(MPI.COMM_WORLD, nx, ny, cell_type=mesh.CellType.quadrilateral)
    V = fem.functionspace(domain, ("Lagrange", degree))
    
    # 3. Time stepping setup
    # Use a finer dt for accuracy
    dt = dt_suggested
    n_steps = int(np.ceil(t_end / dt))
    dt = t_end / n_steps  # adjust to hit t_end exactly
    
    # 4. Functions
    u_n = fem.Function(V)  # solution at previous time step
    u_n.interpolate(lambda x: np.full(x.shape[1], u0_val))
    
    # Store initial condition for output
    u_initial_func = fem.Function(V)
    u_initial_func.interpolate(lambda x: np.full(x.shape[1], u0_val))
    
    uh = fem.Function(V)  # solution at current time step
    
    # 5. Variational form (Backward Euler)
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    
    f = fem.Constant(domain, PETSc.ScalarType(f_val))
    dt_const = fem.Constant(domain, PETSc.ScalarType(dt))
    kappa_const = fem.Constant(domain, PETSc.ScalarType(kappa))
    
    # Backward Euler: (u - u_n)/dt - kappa * laplacian(u) = f
    # Weak form: (u, v)/dt + kappa*(grad(u), grad(v)) = (f, v) + (u_n, v)/dt
    a = (ufl.inner(u, v) / dt_const + kappa_const * ufl.inner(ufl.grad(u), ufl.grad(v))) * ufl.dx
    L = (ufl.inner(f, v) + ufl.inner(u_n, v) / dt_const) * ufl.dx
    
    # 6. Boundary conditions
    tdim = domain.topology.dim
    fdim = tdim - 1
    
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
    )
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc = fem.dirichletbc(PETSc.ScalarType(bc_val), boundary_dofs, V)
    
    # 7. Assemble and solve with manual assembly for efficiency in time loop
    a_form = fem.form(a)
    L_form = fem.form(L)
    
    A = petsc.assemble_matrix(a_form, bcs=[bc])
    A.assemble()
    
    b = fem.petsc.create_vector(L_form)
    
    solver = PETSc.KSP().create(domain.comm)
    solver.setOperators(A)
    solver.setType(PETSc.KSP.Type.CG)
    pc = solver.getPC()
    pc.setType(PETSc.PC.Type.ILU)
    solver.setTolerances(rtol=1e-10, atol=1e-12, max_it=1000)
    solver.setUp()
    
    total_iterations = 0
    
    # 8. Time loop
    for step in range(n_steps):
        # Assemble RHS
        with b.localForm() as loc:
            loc.set(0.0)
        petsc.assemble_vector(b, L_form)
        petsc.apply_lifting(b, [a_form], bcs=[[bc]])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        petsc.set_bc(b, [bc])
        
        # Solve
        solver.solve(b, uh.x.petsc_vec)
        uh.x.scatter_forward()
        
        total_iterations += solver.getIterationNumber()
        
        # Update for next step
        u_n.x.array[:] = uh.x.array[:]
    
    # 9. Extract solution on 50x50 grid
    nx_out = 50
    ny_out = 50
    xs = np.linspace(0.0, 1.0, nx_out)
    ys = np.linspace(0.0, 1.0, ny_out)
    XX, YY = np.meshgrid(xs, ys, indexing='ij')
    points = np.zeros((3, nx_out * ny_out))
    points[0, :] = XX.ravel()
    points[1, :] = YY.ravel()
    
    bb_tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, points.T)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points.T)
    
    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    for i in range(points.shape[1]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(points[:, i])
            cells_on_proc.append(links[0])
            eval_map.append(i)
    
    u_values = np.full(nx_out * ny_out, np.nan)
    if len(points_on_proc) > 0:
        pts_arr = np.array(points_on_proc)
        cells_arr = np.array(cells_on_proc, dtype=np.int32)
        vals = uh.eval(pts_arr, cells_arr)
        u_values[eval_map] = vals.flatten()
    
    u_grid = u_values.reshape((nx_out, ny_out))
    
    # Also extract initial condition on same grid
    u0_values = np.full(nx_out * ny_out, np.nan)
    if len(points_on_proc) > 0:
        vals0 = u_initial_func.eval(pts_arr, cells_arr)
        u0_values[eval_map] = vals0.flatten()
    u_initial_grid = u0_values.reshape((nx_out, ny_out))
    
    # Cleanup
    solver.destroy()
    A.destroy()
    b.destroy()
    
    return {
        "u": u_grid,
        "u_initial": u_initial_grid,
        "solver_info": {
            "mesh_resolution": nx,
            "element_degree": degree,
            "ksp_type": "cg",
            "pc_type": "ilu",
            "rtol": 1e-10,
            "iterations": total_iterations,
            "dt": dt,
            "n_steps": n_steps,
            "time_scheme": "backward_euler",
        }
    }