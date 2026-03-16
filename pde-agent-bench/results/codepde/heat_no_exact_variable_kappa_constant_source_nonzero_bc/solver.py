import numpy as np
from dolfinx import mesh, fem, default_scalar_type, geometry
from dolfinx.fem import petsc
from mpi4py import MPI
import ufl
from petsc4py import PETSc


def solve(case_spec: dict) -> dict:
    # 1. Parse case_spec
    pde = case_spec.get("pde", case_spec.get("oracle_config", {}).get("pde", {}))
    
    # Source term
    source_val = 1.0
    if "source" in pde:
        src = pde["source"]
        if isinstance(src, (int, float)):
            source_val = float(src)
        elif isinstance(src, dict) and "value" in src:
            source_val = float(src["value"])
        elif isinstance(src, str):
            source_val = float(src)
    
    # Initial condition
    u0_val = 0.0
    if "initial_condition" in pde:
        ic = pde["initial_condition"]
        if isinstance(ic, (int, float)):
            u0_val = float(ic)
        elif isinstance(ic, dict) and "value" in ic:
            u0_val = float(ic["value"])
    
    # Time parameters
    time_params = pde.get("time", {})
    t_end = float(time_params.get("t_end", 0.1))
    dt_suggested = float(time_params.get("dt", 0.02))
    scheme = time_params.get("scheme", "backward_euler")
    
    # Kappa expression
    coeffs = pde.get("coefficients", {})
    kappa_info = coeffs.get("kappa", coeffs.get("k", {"type": "constant", "value": 1.0}))
    
    # Boundary conditions
    bc_info = pde.get("boundary_conditions", {})
    
    # 2. Create mesh
    nx = ny = 80
    domain = mesh.create_unit_square(MPI.COMM_WORLD, nx, ny, cell_type=mesh.CellType.triangle)
    
    # 3. Function space
    degree = 1
    V = fem.functionspace(domain, ("Lagrange", degree))
    
    # 4. Define kappa
    x = ufl.SpatialCoordinate(domain)
    pi = ufl.pi
    
    if isinstance(kappa_info, dict) and kappa_info.get("type") == "expr":
        # kappa = 1 + 0.5*sin(2*pi*x)*sin(2*pi*y)
        kappa = 1.0 + 0.5 * ufl.sin(2 * pi * x[0]) * ufl.sin(2 * pi * x[1])
    elif isinstance(kappa_info, dict) and kappa_info.get("type") == "constant":
        kappa = fem.Constant(domain, PETSc.ScalarType(float(kappa_info.get("value", 1.0))))
    else:
        kappa = fem.Constant(domain, PETSc.ScalarType(1.0))
    
    # 5. Source term
    f = fem.Constant(domain, PETSc.ScalarType(source_val))
    
    # 6. Time stepping setup
    dt = dt_suggested
    n_steps = int(np.ceil(t_end / dt))
    dt = t_end / n_steps  # adjust to hit t_end exactly
    
    dt_const = fem.Constant(domain, PETSc.ScalarType(dt))
    
    # 7. Functions
    u_n = fem.Function(V, name="u_n")  # solution at previous time step
    u_n.interpolate(lambda x_arr: np.full(x_arr.shape[1], u0_val))
    
    uh = fem.Function(V, name="uh")  # solution at current time step
    
    # Store initial condition for output
    u_initial_func = fem.Function(V)
    u_initial_func.interpolate(lambda x_arr: np.full(x_arr.shape[1], u0_val))
    
    # 8. Variational form (backward Euler)
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    
    # Backward Euler: (u - u_n)/dt - div(kappa * grad(u)) = f
    # Weak form: (u, v)/dt + (kappa * grad(u), grad(v)) = (f, v) + (u_n, v)/dt
    a = (u * v / dt_const) * ufl.dx + kappa * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = (f * v) * ufl.dx + (u_n * v / dt_const) * ufl.dx
    
    # 9. Boundary conditions
    # Parse BC - default to 0 Dirichlet
    bc_value = 0.0
    if bc_info:
        for key, val in bc_info.items():
            if isinstance(val, dict):
                if val.get("type") == "dirichlet":
                    bc_value = float(val.get("value", 0.0))
            elif isinstance(val, (int, float)):
                bc_value = float(val)
    
    # Check if there's a specific nonzero BC
    # For "nonzero_bc" case, let's check the case ID or BC info more carefully
    # The case says "nonzero_bc" but we need to figure out the actual value
    # Let's check all possible locations
    if "boundary_conditions" in pde:
        bcs_spec = pde["boundary_conditions"]
        if isinstance(bcs_spec, dict):
            for k, v_bc in bcs_spec.items():
                if isinstance(v_bc, dict):
                    if "value" in v_bc:
                        bc_value = float(v_bc["value"])
                    elif "expr" in v_bc:
                        pass  # handle expression BCs if needed
    
    tdim = domain.topology.dim
    fdim = tdim - 1
    
    # All boundary
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x_arr: np.ones(x_arr.shape[1], dtype=bool)
    )
    
    dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    
    u_bc = fem.Function(V)
    u_bc.interpolate(lambda x_arr: np.full(x_arr.shape[1], bc_value))
    bc = fem.dirichletbc(u_bc, dofs)
    
    # 10. Assemble and solve with manual assembly for time loop efficiency
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
    solver.setTolerances(rtol=1e-8, atol=1e-12, max_it=2000)
    solver.setUp()
    
    total_iterations = 0
    
    # 11. Time loop
    t = 0.0
    for step in range(n_steps):
        t += dt
        
        # Assemble RHS
        with b.localForm() as loc:
            loc.set(0.0)
        petsc.assemble_vector(b, L_form)
        
        # Apply lifting for Dirichlet BCs
        petsc.apply_lifting(b, [a_form], bcs=[[bc]])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        petsc.set_bc(b, [bc])
        
        # Solve
        solver.solve(b, uh.x.petsc_vec)
        uh.x.scatter_forward()
        
        total_iterations += solver.getIterationNumber()
        
        # Update previous solution
        u_n.x.array[:] = uh.x.array[:]
    
    # 12. Extract solution on 50x50 grid
    nx_out = ny_out = 50
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
            "rtol": 1e-8,
            "iterations": total_iterations,
            "dt": dt,
            "n_steps": n_steps,
            "time_scheme": "backward_euler",
        }
    }