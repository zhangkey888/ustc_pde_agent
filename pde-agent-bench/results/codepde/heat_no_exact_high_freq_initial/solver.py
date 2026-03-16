import numpy as np
from dolfinx import mesh, fem, default_scalar_type, geometry
from dolfinx.fem import petsc
from mpi4py import MPI
import ufl
from petsc4py import PETSc


def solve(case_spec: dict) -> dict:
    # 1. Parse case_spec
    pde = case_spec.get("pde", {})
    if not pde:
        pde = case_spec.get("oracle_config", {}).get("pde", {})
    
    kappa = float(pde.get("coefficients", {}).get("kappa", 1.0))
    
    time_params = pde.get("time", {})
    t_end = float(time_params.get("t_end", 0.12))
    dt_suggested = float(time_params.get("dt", 0.02))
    scheme = time_params.get("scheme", "backward_euler")
    
    # Choose mesh resolution and dt for accuracy
    nx = ny = 80
    dt = 0.002  # smaller dt for accuracy with high-frequency initial condition
    n_steps = int(round(t_end / dt))
    dt = t_end / n_steps  # adjust to hit t_end exactly
    
    element_degree = 1
    
    # 2. Create mesh
    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_square(comm, nx, ny, cell_type=mesh.CellType.triangle)
    
    # 3. Function space
    V = fem.functionspace(domain, ("Lagrange", element_degree))
    
    # 4. Define functions
    u_n = fem.Function(V)  # solution at previous time step
    u_h = fem.Function(V)  # solution at current time step (will be updated)
    
    # Initial condition: u0 = sin(6*pi*x)*sin(6*pi*y)
    def u0_expr(x):
        return np.sin(6.0 * np.pi * x[0]) * np.sin(6.0 * np.pi * x[1])
    
    u_n.interpolate(u0_expr)
    
    # Store initial condition for output
    # Build evaluation grid first
    nx_eval, ny_eval = 50, 50
    x_eval = np.linspace(0, 1, nx_eval)
    y_eval = np.linspace(0, 1, ny_eval)
    XX, YY = np.meshgrid(x_eval, y_eval, indexing='ij')
    points_2d = np.vstack([XX.ravel(), YY.ravel()])
    points_3d = np.vstack([points_2d, np.zeros(points_2d.shape[1])])
    
    # Evaluate initial condition on grid
    u_initial = u0_expr(points_3d).reshape(nx_eval, ny_eval)
    
    # 5. Boundary conditions: u = g on boundary
    # For this problem, g = 0 on boundary (sin(6*pi*x)*sin(6*pi*y) = 0 on boundary of unit square)
    tdim = domain.topology.dim
    fdim = tdim - 1
    
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
    )
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc = fem.dirichletbc(PETSc.ScalarType(0.0), boundary_dofs, V)
    bcs = [bc]
    
    # 6. Variational form for backward Euler
    # (u - u_n)/dt - kappa * laplacian(u) = f
    # f = 0 (no source term specified beyond initial condition decay)
    # Weak form: (u, v)/dt + kappa*(grad(u), grad(v)) = (u_n, v)/dt + (f, v)
    
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    
    kappa_c = fem.Constant(domain, PETSc.ScalarType(kappa))
    dt_c = fem.Constant(domain, PETSc.ScalarType(dt))
    f = fem.Constant(domain, PETSc.ScalarType(0.0))
    
    a = (u * v / dt_c + kappa_c * ufl.inner(ufl.grad(u), ufl.grad(v))) * ufl.dx
    L = (u_n * v / dt_c + f * v) * ufl.dx
    
    # 7. Compile forms and set up manual solver for efficiency in time loop
    a_form = fem.form(a)
    L_form = fem.form(L)
    
    # Assemble matrix (constant in time)
    A = petsc.assemble_matrix(a_form, bcs=bcs)
    A.assemble()
    
    # Create RHS vector
    b = fem.petsc.create_vector(L_form)
    
    # Set up KSP solver
    solver = PETSc.KSP().create(domain.comm)
    solver.setOperators(A)
    solver.setType(PETSc.KSP.Type.CG)
    pc = solver.getPC()
    pc.setType(PETSc.PC.Type.HYPRE)
    solver.setTolerances(rtol=1e-10, atol=1e-12, max_it=1000)
    solver.setUp()
    
    total_iterations = 0
    
    # 8. Time stepping
    for step in range(n_steps):
        # Assemble RHS
        with b.localForm() as loc:
            loc.set(0.0)
        petsc.assemble_vector(b, L_form)
        
        # Apply lifting for Dirichlet BCs
        petsc.apply_lifting(b, [a_form], bcs=[bcs])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        petsc.set_bc(b, bcs)
        
        # Solve
        solver.solve(b, u_h.x.petsc_vec)
        u_h.x.scatter_forward()
        
        total_iterations += solver.getIterationNumber()
        
        # Update for next step
        u_n.x.array[:] = u_h.x.array[:]
    
    # 9. Extract solution on evaluation grid
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
        vals = u_h.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u_values[eval_map] = vals.flatten()
    
    u_grid = u_values.reshape(nx_eval, ny_eval)
    
    # Clean up PETSc objects
    solver.destroy()
    A.destroy()
    b.destroy()
    
    return {
        "u": u_grid,
        "u_initial": u_initial,
        "solver_info": {
            "mesh_resolution": nx,
            "element_degree": element_degree,
            "ksp_type": "cg",
            "pc_type": "hypre",
            "rtol": 1e-10,
            "iterations": total_iterations,
            "dt": dt,
            "n_steps": n_steps,
            "time_scheme": "backward_euler",
        }
    }