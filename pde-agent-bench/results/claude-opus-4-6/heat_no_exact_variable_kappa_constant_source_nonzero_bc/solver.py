import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc

ScalarType = PETSc.ScalarType


def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD
    
    # Parameters
    nx = ny = 64
    degree = 1
    dt = 0.02
    t_end = 0.1
    n_steps = int(round(t_end / dt))
    
    # Create mesh
    domain = mesh.create_unit_square(comm, nx, ny, cell_type=mesh.CellType.triangle)
    tdim = domain.topology.dim
    fdim = tdim - 1
    
    # Function space
    V = fem.functionspace(domain, ("Lagrange", degree))
    
    # Functions
    u_n = fem.Function(V)  # solution at previous time step
    u_n.x.array[:] = 0.0  # initial condition u0 = 0
    
    u_sol = fem.Function(V)  # solution at current time step
    
    # Trial and test functions
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    
    # Spatial coordinate for kappa
    x = ufl.SpatialCoordinate(domain)
    kappa = 1.0 + 0.5 * ufl.sin(2 * ufl.pi * x[0]) * ufl.sin(2 * ufl.pi * x[1])
    
    # Source term
    f = fem.Constant(domain, ScalarType(1.0))
    
    # Time step constant
    dt_c = fem.Constant(domain, ScalarType(dt))
    
    # Backward Euler weak form:
    # (u - u_n)/dt * v + kappa * grad(u) . grad(v) = f * v
    a = (u * v / dt_c) * ufl.dx + ufl.inner(kappa * ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = (u_n / dt_c) * v * ufl.dx + f * v * ufl.dx
    
    # Boundary conditions: u = g on boundary
    # g is not specified explicitly in the case, but the BC says u = g
    # From the problem: nonzero_bc. Let's check case_spec if available, otherwise use 0
    # The case says "nonzero_bc" but doesn't specify g explicitly. 
    # Looking at the case ID: "nonzero_bc" - let me check if there's BC info in case_spec
    # For safety, let's try to extract from case_spec
    g_val = 0.0
    if case_spec and 'pde' in case_spec:
        bc_info = case_spec['pde'].get('bc', {})
        if isinstance(bc_info, dict):
            g_val_raw = bc_info.get('value', 0.0)
            if isinstance(g_val_raw, (int, float)):
                g_val = float(g_val_raw)
    
    # If nonzero_bc but no explicit value, try extracting from case_spec more carefully
    # Default: use what's given. The name says nonzero_bc so there should be a value.
    # Let's also check for 'g' key
    if case_spec and 'pde' in case_spec:
        bc_info = case_spec['pde'].get('boundary_conditions', case_spec['pde'].get('bc', {}))
        if isinstance(bc_info, dict):
            g_expr = bc_info.get('expr', bc_info.get('value', None))
            if g_expr is not None:
                if isinstance(g_expr, (int, float)):
                    g_val = float(g_expr)
                elif isinstance(g_expr, str):
                    # Try to parse simple expressions
                    try:
                        g_val = float(g_expr)
                    except ValueError:
                        g_val = 0.0
        elif isinstance(bc_info, list):
            for bc_item in bc_info:
                if isinstance(bc_item, dict):
                    val = bc_item.get('value', bc_item.get('expr', None))
                    if val is not None:
                        try:
                            g_val = float(val)
                        except (ValueError, TypeError):
                            g_val = 0.0

    # Locate all boundary facets
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
    )
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    
    u_bc = fem.Function(V)
    u_bc.interpolate(lambda x: np.full(x.shape[1], g_val))
    bc = fem.dirichletbc(u_bc, boundary_dofs)
    bcs = [bc]
    
    # Compile forms
    a_form = fem.form(a)
    L_form = fem.form(L)
    
    # Assemble matrix (constant in time for this problem since kappa doesn't depend on t)
    A = petsc.assemble_matrix(a_form, bcs=bcs)
    A.assemble()
    
    # Create RHS vector
    b = petsc.create_vector(L_form)
    
    # Setup solver
    solver = PETSc.KSP().create(domain.comm)
    solver.setOperators(A)
    solver.setType(PETSc.KSP.Type.CG)
    pc = solver.getPC()
    pc.setType(PETSc.PC.Type.ILU)
    solver.setTolerances(rtol=1e-8, atol=1e-12, max_it=1000)
    solver.setUp()
    
    # Store initial condition for output
    u_initial_local = u_n.x.array.copy()
    
    # Time stepping
    total_iterations = 0
    for step in range(n_steps):
        # Assemble RHS
        with b.localForm() as loc:
            loc.set(0.0)
        petsc.assemble_vector(b, L_form)
        petsc.apply_lifting(b, [a_form], bcs=[bcs])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        petsc.set_bc(b, bcs)
        
        # Solve
        solver.solve(b, u_sol.x.petsc_vec)
        u_sol.x.scatter_forward()
        
        total_iterations += solver.getIterationNumber()
        
        # Update previous solution
        u_n.x.array[:] = u_sol.x.array[:]
    
    # Sample solution on 50x50 grid
    nx_out, ny_out = 50, 50
    xs = np.linspace(0, 1, nx_out)
    ys = np.linspace(0, 1, ny_out)
    XX, YY = np.meshgrid(xs, ys, indexing='ij')
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
            points_on_proc.append(points[:, i])
            cells_on_proc.append(links[0])
            eval_map.append(i)
    
    u_values = np.full(nx_out * ny_out, np.nan)
    if len(points_on_proc) > 0:
        pts_arr = np.array(points_on_proc)
        cells_arr = np.array(cells_on_proc, dtype=np.int32)
        vals = u_sol.eval(pts_arr, cells_arr)
        u_values[eval_map] = vals.flatten()
    
    u_grid = u_values.reshape((nx_out, ny_out))
    
    # Also sample initial condition
    u_n_init = fem.Function(V)
    u_n_init.x.array[:] = 0.0  # u0 = 0
    u_init_values = np.full(nx_out * ny_out, np.nan)
    if len(points_on_proc) > 0:
        vals_init = u_n_init.eval(pts_arr, cells_arr)
        u_init_values[eval_map] = vals_init.flatten()
    u_initial_grid = u_init_values.reshape((nx_out, ny_out))
    
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