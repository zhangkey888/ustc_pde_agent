import numpy as np
from dolfinx import mesh, fem, default_scalar_type, geometry
from dolfinx.fem import petsc
from mpi4py import MPI
import ufl
from petsc4py import PETSc


def solve(case_spec: dict) -> dict:
    # 1. Parse case_spec
    pde_config = case_spec.get("pde", {})
    coeffs = pde_config.get("coefficients", {})
    kappa = coeffs.get("kappa", 1.0)
    
    time_params = pde_config.get("time", {})
    t_end = time_params.get("t_end", 0.1)
    dt_suggested = time_params.get("dt", 0.005)
    scheme = time_params.get("scheme", "backward_euler")
    
    # For high-frequency solution, use finer mesh and smaller dt
    nx = 100
    ny = 100
    degree = 1
    dt = 0.002  # smaller than suggested for accuracy with high-frequency solution
    
    n_steps = int(round(t_end / dt))
    dt = t_end / n_steps  # adjust to hit t_end exactly
    
    # 2. Create mesh
    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_square(comm, nx, ny, cell_type=mesh.CellType.triangle)
    
    # 3. Function space
    V = fem.functionspace(domain, ("Lagrange", degree))
    
    # Spatial coordinates
    x = ufl.SpatialCoordinate(domain)
    
    # Time parameter
    t = fem.Constant(domain, default_scalar_type(0.0))
    
    # Manufactured solution: u = exp(-t)*sin(4*pi*x)*sin(4*pi*y)
    pi = np.pi
    u_exact_ufl = ufl.exp(-t) * ufl.sin(4 * ufl.pi * x[0]) * ufl.sin(4 * ufl.pi * x[1])
    
    # Source term: f = du/dt - kappa * laplacian(u)
    # du/dt = -exp(-t)*sin(4*pi*x)*sin(4*pi*y)
    # laplacian(u) = exp(-t)*(-16*pi^2*sin(4*pi*x)*sin(4*pi*y) - 16*pi^2*sin(4*pi*x)*sin(4*pi*y))
    #              = -32*pi^2 * exp(-t)*sin(4*pi*x)*sin(4*pi*y)
    # f = du/dt - kappa * laplacian(u)
    #   = -exp(-t)*sin(...) - kappa*(-32*pi^2)*exp(-t)*sin(...)
    #   = exp(-t)*sin(4*pi*x)*sin(4*pi*y) * (-1 + 32*kappa*pi^2)
    f_ufl = ufl.exp(-t) * ufl.sin(4 * ufl.pi * x[0]) * ufl.sin(4 * ufl.pi * x[1]) * (-1.0 + 32.0 * kappa * pi**2)
    
    # 4. Define variational forms for backward Euler
    # (u^{n+1} - u^n)/dt - kappa * laplacian(u^{n+1}) = f^{n+1}
    # Weak form: (u^{n+1}, v)/dt + kappa*(grad(u^{n+1}), grad(v)) = (u^n, v)/dt + (f^{n+1}, v)
    
    u_trial = ufl.TrialFunction(V)
    v_test = ufl.TestFunction(V)
    
    u_n = fem.Function(V)  # solution at previous time step
    
    dt_const = fem.Constant(domain, default_scalar_type(dt))
    kappa_const = fem.Constant(domain, default_scalar_type(kappa))
    
    a_form = (u_trial * v_test / dt_const + kappa_const * ufl.inner(ufl.grad(u_trial), ufl.grad(v_test))) * ufl.dx
    L_form = (u_n * v_test / dt_const + f_ufl * v_test) * ufl.dx
    
    # 5. Boundary conditions - all boundary
    tdim = domain.topology.dim
    fdim = tdim - 1
    
    # Create boundary condition function
    u_bc_func = fem.Function(V)
    
    # Locate all boundary facets
    domain.topology.create_connectivity(fdim, tdim)
    boundary_facets = mesh.exterior_facet_indices(domain.topology)
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    
    # Create expression for exact solution on boundary
    u_exact_expr = fem.Expression(u_exact_ufl, V.element.interpolation_points)
    
    # 6. Initial condition
    t.value = 0.0
    u_n.interpolate(u_exact_expr)
    
    # Store initial condition for output
    # Evaluate on grid
    nx_out, ny_out = 50, 50
    xs = np.linspace(0, 1, nx_out)
    ys = np.linspace(0, 1, ny_out)
    X, Y = np.meshgrid(xs, ys, indexing='ij')
    points_2d = np.vstack([X.ravel(), Y.ravel()])
    points_3d = np.vstack([points_2d, np.zeros(points_2d.shape[1])])
    
    # Build evaluation structures
    bb_tree = geometry.bb_tree(domain, tdim)
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
    
    points_on_proc_arr = np.array(points_on_proc) if len(points_on_proc) > 0 else np.empty((0, 3))
    cells_on_proc_arr = np.array(cells_on_proc, dtype=np.int32) if len(cells_on_proc) > 0 else np.empty(0, dtype=np.int32)
    
    # Evaluate initial condition
    u_initial_vals = np.full(points_3d.shape[1], np.nan)
    if len(points_on_proc) > 0:
        vals = u_n.eval(points_on_proc_arr, cells_on_proc_arr)
        u_initial_vals[eval_map] = vals.flatten()
    u_initial_grid = u_initial_vals.reshape(nx_out, ny_out)
    
    # 7. Time stepping with manual assembly for efficiency
    a_compiled = fem.form(a_form)
    L_compiled = fem.form(L_form)
    
    # Assemble matrix (constant in time for backward Euler with constant kappa)
    A = petsc.assemble_matrix(a_compiled, bcs=[])
    A.assemble()
    
    b = fem.petsc.create_vector(L_compiled)
    
    u_sol = fem.Function(V)
    
    # Setup solver
    solver = PETSc.KSP().create(domain.comm)
    solver.setOperators(A)
    solver.setType(PETSc.KSP.Type.CG)
    pc = solver.getPC()
    pc.setType(PETSc.PC.Type.HYPRE)
    solver.setTolerances(rtol=1e-10, atol=1e-12, max_it=1000)
    solver.setUp()
    
    total_iterations = 0
    
    # We need to reassemble A with BCs applied
    # Actually, let's reassemble with BCs since BC values change each step
    for step in range(n_steps):
        t.value = (step + 1) * dt
        
        # Update boundary condition
        u_bc_func.interpolate(u_exact_expr)
        bc = fem.dirichletbc(u_bc_func, boundary_dofs)
        
        # Reassemble matrix with BCs (matrix is constant but BCs change values)
        A.zeroEntries()
        petsc.assemble_matrix(A, a_compiled, bcs=[bc])
        A.assemble()
        
        # Assemble RHS
        with b.localForm() as loc:
            loc.set(0)
        petsc.assemble_vector(b, L_compiled)
        petsc.apply_lifting(b, [a_compiled], bcs=[[bc]])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        petsc.set_bc(b, [bc])
        
        # Solve
        solver.solve(b, u_sol.x.petsc_vec)
        u_sol.x.scatter_forward()
        
        total_iterations += solver.getIterationNumber()
        
        # Update u_n
        u_n.x.array[:] = u_sol.x.array[:]
    
    # 8. Extract solution on grid
    u_final_vals = np.full(points_3d.shape[1], np.nan)
    if len(points_on_proc) > 0:
        vals = u_sol.eval(points_on_proc_arr, cells_on_proc_arr)
        u_final_vals[eval_map] = vals.flatten()
    u_grid = u_final_vals.reshape(nx_out, ny_out)
    
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
            "pc_type": "hypre",
            "rtol": 1e-10,
            "iterations": total_iterations,
            "dt": dt,
            "n_steps": n_steps,
            "time_scheme": "backward_euler",
        }
    }