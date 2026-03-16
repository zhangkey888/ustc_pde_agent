import numpy as np
from dolfinx import mesh, fem, default_scalar_type, geometry
from dolfinx.fem import petsc
from mpi4py import MPI
import ufl
from petsc4py import PETSc


def solve(case_spec: dict) -> dict:
    # 1. Parse case_spec
    pde = case_spec.get("pde", {})
    
    # Extract parameters
    kappa = 1.0
    t_end = 0.1
    dt_suggested = 0.02
    scheme = "backward_euler"
    
    # Try to extract from case_spec
    if "coefficients" in pde:
        kappa = float(pde["coefficients"].get("kappa", kappa))
    if "time" in pde:
        t_end = float(pde["time"].get("t_end", t_end))
        dt_suggested = float(pde["time"].get("dt", dt_suggested))
        scheme = pde["time"].get("scheme", scheme)
    
    # Agent-selected parameters
    nx = 80
    ny = 80
    degree = 1
    dt = dt_suggested
    n_steps = int(np.round(t_end / dt))
    dt = t_end / n_steps  # adjust to hit t_end exactly
    
    # 2. Create mesh
    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_square(comm, nx, ny, cell_type=mesh.CellType.triangle)
    
    # 3. Function space
    V = fem.functionspace(domain, ("Lagrange", degree))
    
    # 4. Define functions
    u_n = fem.Function(V)  # solution at previous time step
    u_n.name = "u_n"
    
    # Initial condition: u0 = 0.0
    u_n.x.array[:] = 0.0
    
    # Trial and test functions
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    
    # Source term
    x = ufl.SpatialCoordinate(domain)
    f = (ufl.exp(-220.0 * ((x[0] - 0.25)**2 + (x[1] - 0.25)**2)) +
         ufl.exp(-220.0 * ((x[0] - 0.75)**2 + (x[1] - 0.7)**2)))
    
    # Diffusivity
    kappa_c = fem.Constant(domain, PETSc.ScalarType(kappa))
    dt_c = fem.Constant(domain, PETSc.ScalarType(dt))
    
    # Backward Euler variational form:
    # (u - u_n)/dt - kappa * laplacian(u) = f
    # Weak form: (u, v)/dt + kappa*(grad(u), grad(v)) = (u_n, v)/dt + (f, v)
    a = (ufl.inner(u, v) / dt_c * ufl.dx +
         kappa_c * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx)
    L = (ufl.inner(u_n, v) / dt_c * ufl.dx +
         ufl.inner(f, v) * ufl.dx)
    
    # 5. Boundary conditions: u = g on boundary
    # g = 0 (homogeneous Dirichlet, consistent with initial condition)
    tdim = domain.topology.dim
    fdim = tdim - 1
    
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool))
    dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc = fem.dirichletbc(PETSc.ScalarType(0.0), dofs, V)
    
    # 6. Assemble and set up solver manually for time stepping
    a_form = fem.form(a)
    L_form = fem.form(L)
    
    A = petsc.assemble_matrix(a_form, bcs=[bc])
    A.assemble()
    
    b = fem.Function(V)
    b_vec = b.x.petsc_vec
    
    # Create solution function
    u_sol = fem.Function(V)
    u_sol.name = "u_sol"
    
    # Set up KSP solver
    solver = PETSc.KSP().create(domain.comm)
    solver.setOperators(A)
    solver.setType(PETSc.KSP.Type.CG)
    pc = solver.getPC()
    pc.setType(PETSc.PC.Type.HYPRE)
    solver.setTolerances(rtol=1e-10, atol=1e-12, max_it=1000)
    solver.setUp()
    
    total_iterations = 0
    
    # Store initial condition for output
    # Evaluate on grid before time stepping
    nx_out, ny_out = 50, 50
    xs = np.linspace(0.0, 1.0, nx_out)
    ys = np.linspace(0.0, 1.0, ny_out)
    XX, YY = np.meshgrid(xs, ys, indexing='ij')
    points_2d = np.column_stack([XX.ravel(), YY.ravel()])
    points_3d = np.zeros((points_2d.shape[0], 3))
    points_3d[:, :2] = points_2d
    
    # Build evaluation structures
    bb_tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, points_3d)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points_3d)
    
    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    for i in range(points_3d.shape[0]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(points_3d[i])
            cells_on_proc.append(links[0])
            eval_map.append(i)
    
    points_on_proc = np.array(points_on_proc) if len(points_on_proc) > 0 else np.zeros((0, 3))
    cells_on_proc = np.array(cells_on_proc, dtype=np.int32) if len(cells_on_proc) > 0 else np.zeros(0, dtype=np.int32)
    
    # Evaluate initial condition
    u_initial_values = np.full(points_3d.shape[0], np.nan)
    if len(points_on_proc) > 0:
        vals = u_n.eval(points_on_proc, cells_on_proc)
        u_initial_values[eval_map] = vals.flatten()
    u_initial_grid = u_initial_values.reshape((nx_out, ny_out))
    
    # 7. Time stepping
    t = 0.0
    for step in range(n_steps):
        t += dt
        
        # Assemble RHS
        with b_vec.localForm() as loc:
            loc.set(0.0)
        petsc.assemble_vector(b_vec, L_form)
        petsc.apply_lifting(b_vec, [a_form], bcs=[[bc]])
        b_vec.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        petsc.set_bc(b_vec, [bc])
        
        # Solve
        solver.solve(b_vec, u_sol.x.petsc_vec)
        u_sol.x.scatter_forward()
        
        total_iterations += solver.getIterationNumber()
        
        # Update previous solution
        u_n.x.array[:] = u_sol.x.array[:]
    
    # 8. Extract solution on grid
    u_values = np.full(points_3d.shape[0], np.nan)
    if len(points_on_proc) > 0:
        vals = u_sol.eval(points_on_proc, cells_on_proc)
        u_values[eval_map] = vals.flatten()
    u_grid = u_values.reshape((nx_out, ny_out))
    
    # Clean up
    solver.destroy()
    A.destroy()
    
    return {
        "u": u_grid,
        "u_initial": u_initial_grid,
        "solver_info": {
            "mesh_resolution": nx,
            "element_degree": degree,
            "ksp_type": "cg",
            "pc_type": "hypre",
            "rtol": 1e-10,
            "iterations": total_iterations,
            "dt": dt,
            "n_steps": n_steps,
            "time_scheme": "backward_euler",
        }
    }