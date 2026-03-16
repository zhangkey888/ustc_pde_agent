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
    if "source" in pde:
        src = pde["source"]
        if isinstance(src, (int, float)):
            f_val = float(src)
        elif isinstance(src, str):
            try:
                f_val = float(src)
            except ValueError:
                f_val = 1.0
        elif isinstance(src, dict):
            f_val = float(src.get("value", 1.0))
    
    # Diffusivity
    kappa_val = 1.0
    if "coefficients" in pde:
        coeffs = pde["coefficients"]
        if isinstance(coeffs, dict):
            kappa_val = float(coeffs.get("kappa", coeffs.get("k", 1.0)))
    
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
        elif isinstance(ic, dict):
            u0_val = float(ic.get("value", 0.0))
    
    # Time parameters
    time_params = pde.get("time", {})
    t_end = float(time_params.get("t_end", 0.1))
    dt_suggest = float(time_params.get("dt", 0.02))
    scheme = time_params.get("scheme", "backward_euler")
    
    # Boundary conditions (homogeneous Dirichlet by default)
    bc_val = 0.0
    if "boundary_conditions" in pde:
        bcs_spec = pde["boundary_conditions"]
        if isinstance(bcs_spec, dict):
            bc_val = float(bcs_spec.get("value", 0.0))
        elif isinstance(bcs_spec, list):
            for b in bcs_spec:
                if isinstance(b, dict) and b.get("type", "dirichlet") == "dirichlet":
                    bc_val = float(b.get("value", 0.0))
    
    # 2. Mesh and function space
    nx = ny = 64
    degree = 1
    domain = mesh.create_unit_square(MPI.COMM_WORLD, nx, ny, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", degree))
    
    # 3. Time stepping setup
    dt = dt_suggest
    n_steps = int(np.ceil(t_end / dt))
    dt = t_end / n_steps  # adjust to hit t_end exactly
    
    # 4. Functions
    u_n = fem.Function(V)  # solution at previous time step
    u_n.interpolate(lambda x: np.full(x.shape[1], u0_val))
    
    # Store initial condition for output
    u0_grid = None  # will compute later
    
    # 5. Variational form (backward Euler)
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    
    f = fem.Constant(domain, PETSc.ScalarType(f_val))
    kappa = fem.Constant(domain, PETSc.ScalarType(kappa_val))
    dt_const = fem.Constant(domain, PETSc.ScalarType(dt))
    
    # Backward Euler: (u - u_n)/dt - kappa * laplacian(u) = f
    # Weak form: (u, v)/dt + kappa*(grad(u), grad(v)) = (f, v) + (u_n, v)/dt
    a = (ufl.inner(u, v) / dt_const + kappa * ufl.inner(ufl.grad(u), ufl.grad(v))) * ufl.dx
    L = (ufl.inner(f, v) + ufl.inner(u_n, v) / dt_const) * ufl.dx
    
    # 6. Boundary conditions
    tdim = domain.topology.dim
    fdim = tdim - 1
    
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
    )
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc = fem.dirichletbc(PETSc.ScalarType(bc_val), boundary_dofs, V)
    
    # 7. Assemble and solve using manual assembly for efficiency in time loop
    a_form = fem.form(a)
    L_form = fem.form(L)
    
    A = petsc.assemble_matrix(a_form, bcs=[bc])
    A.assemble()
    
    b = fem.petsc.create_vector(L_form)
    
    # Set up KSP solver
    solver = PETSc.KSP().create(domain.comm)
    solver.setOperators(A)
    solver.setType(PETSc.KSP.Type.CG)
    pc = solver.getPC()
    pc.setType(PETSc.PC.Type.HYPRE)
    solver.setTolerances(rtol=1e-8, atol=1e-12, max_it=1000)
    solver.setUp()
    
    uh = fem.Function(V)
    
    total_iterations = 0
    
    # 8. Time stepping loop
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
        
        # Update u_n
        u_n.x.array[:] = uh.x.array[:]
    
    # 9. Extract solution on 50x50 grid
    n_eval = 50
    xs = np.linspace(0.0, 1.0, n_eval)
    ys = np.linspace(0.0, 1.0, n_eval)
    XX, YY = np.meshgrid(xs, ys, indexing='ij')
    points_2d = np.vstack([XX.ravel(), YY.ravel()])
    points_3d = np.vstack([points_2d, np.zeros(points_2d.shape[1])])
    
    bb_tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, points_3d.T)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points_3d.T)
    
    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    for i in range(points_3d.shape[1]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(points_3d.T[i])
            cells_on_proc.append(links[0])
            eval_map.append(i)
    
    u_values = np.full(points_3d.shape[1], np.nan)
    if len(points_on_proc) > 0:
        pts_arr = np.array(points_on_proc)
        cells_arr = np.array(cells_on_proc, dtype=np.int32)
        vals = uh.eval(pts_arr, cells_arr)
        u_values[eval_map] = vals.flatten()
    
    u_grid = u_values.reshape((n_eval, n_eval))
    
    # Also extract initial condition on same grid
    u_init_func = fem.Function(V)
    u_init_func.interpolate(lambda x: np.full(x.shape[1], u0_val))
    
    u0_values = np.full(points_3d.shape[1], np.nan)
    if len(points_on_proc) > 0:
        vals0 = u_init_func.eval(pts_arr, cells_arr)
        u0_values[eval_map] = vals0.flatten()
    u0_grid = u0_values.reshape((n_eval, n_eval))
    
    # Cleanup
    solver.destroy()
    A.destroy()
    b.destroy()
    
    return {
        "u": u_grid,
        "u_initial": u0_grid,
        "solver_info": {
            "mesh_resolution": nx,
            "element_degree": degree,
            "ksp_type": "cg",
            "pc_type": "hypre",
            "rtol": 1e-8,
            "iterations": total_iterations,
            "dt": dt,
            "n_steps": n_steps,
            "time_scheme": "backward_euler",
        },
    }