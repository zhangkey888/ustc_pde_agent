import numpy as np
from dolfinx import mesh, fem, default_scalar_type, geometry
from dolfinx.fem import petsc
from mpi4py import MPI
import ufl
from petsc4py import PETSc


def solve(case_spec: dict) -> dict:
    # 1. Parse case_spec
    pde = case_spec.get("pde", case_spec.get("oracle_config", {}).get("pde", {}))
    
    # Time parameters
    time_params = pde.get("time", {})
    t_end = time_params.get("t_end", 0.1)
    dt_suggested = time_params.get("dt", 0.02)
    scheme = time_params.get("scheme", "backward_euler")
    
    # Use a smaller dt for better accuracy
    dt = 0.005
    n_steps = int(round(t_end / dt))
    dt = t_end / n_steps  # adjust to hit t_end exactly
    
    # Mesh resolution
    nx = ny = 80
    
    # 2. Create mesh
    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_square(comm, nx, ny, cell_type=mesh.CellType.triangle)
    
    # 3. Function space
    degree = 1
    V = fem.functionspace(domain, ("Lagrange", degree))
    
    # Spatial coordinate
    x = ufl.SpatialCoordinate(domain)
    
    # 4. Define kappa
    pi = np.pi
    kappa_expr = 1.0 + 0.6 * ufl.sin(2 * ufl.pi * x[0]) * ufl.sin(2 * ufl.pi * x[1])
    
    # 5. Source term
    f_expr = (ufl.sin(4 * ufl.pi * x[0]) * ufl.sin(3 * ufl.pi * x[1])
              + 0.3 * ufl.sin(10 * ufl.pi * x[0]) * ufl.sin(9 * ufl.pi * x[1]))
    
    # 6. Initial condition: u0 = sin(pi*x)*sin(pi*y)
    u_n = fem.Function(V, name="u_n")
    u_n.interpolate(lambda X: np.sin(np.pi * X[0]) * np.sin(np.pi * X[1]))
    
    # Store initial condition for output
    u_initial_func = fem.Function(V)
    u_initial_func.interpolate(lambda X: np.sin(np.pi * X[0]) * np.sin(np.pi * X[1]))
    
    # 7. Boundary conditions (homogeneous Dirichlet: g=0 on boundary)
    tdim = domain.topology.dim
    fdim = tdim - 1
    
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda X: np.ones(X.shape[1], dtype=bool)
    )
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc = fem.dirichletbc(PETSc.ScalarType(0.0), boundary_dofs, V)
    
    # 8. Variational form for backward Euler
    # (u - u_n)/dt - div(kappa * grad(u)) = f
    # Weak form: (u, v)/dt + (kappa*grad(u), grad(v)) = (f, v) + (u_n, v)/dt
    
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    
    dt_const = fem.Constant(domain, PETSc.ScalarType(dt))
    
    a_form = (ufl.inner(u, v) / dt_const * ufl.dx
              + ufl.inner(kappa_expr * ufl.grad(u), ufl.grad(v)) * ufl.dx)
    
    L_form = (ufl.inner(f_expr, v) * ufl.dx
              + ufl.inner(u_n, v) / dt_const * ufl.dx)
    
    # 9. Compile forms
    a_compiled = fem.form(a_form)
    L_compiled = fem.form(L_form)
    
    # 10. Assemble matrix (constant in time since kappa doesn't depend on t)
    A = petsc.assemble_matrix(a_compiled, bcs=[bc])
    A.assemble()
    
    # Create RHS vector
    b = fem.petsc.create_vector(L_compiled)
    
    # Solution function
    u_sol = fem.Function(V)
    
    # 11. Setup KSP solver
    solver = PETSc.KSP().create(domain.comm)
    solver.setOperators(A)
    solver.setType(PETSc.KSP.Type.CG)
    pc = solver.getPC()
    pc.setType(PETSc.PC.Type.HYPRE)
    solver.setTolerances(rtol=1e-10, atol=1e-12, max_it=1000)
    solver.setUp()
    
    total_iterations = 0
    
    # 12. Time-stepping loop
    for step in range(n_steps):
        # Assemble RHS
        with b.localForm() as loc:
            loc.set(0.0)
        petsc.assemble_vector(b, L_compiled)
        petsc.apply_lifting(b, [a_compiled], bcs=[[bc]])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        petsc.set_bc(b, [bc])
        
        # Solve
        solver.solve(b, u_sol.x.petsc_vec)
        u_sol.x.scatter_forward()
        
        total_iterations += solver.getIterationNumber()
        
        # Update u_n for next step
        u_n.x.array[:] = u_sol.x.array[:]
    
    # 13. Extract solution on 50x50 uniform grid
    nx_out, ny_out = 50, 50
    xs = np.linspace(0, 1, nx_out)
    ys = np.linspace(0, 1, ny_out)
    X, Y = np.meshgrid(xs, ys, indexing='ij')
    points = np.zeros((3, nx_out * ny_out))
    points[0, :] = X.ravel()
    points[1, :] = Y.ravel()
    
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
        vals = u_sol.eval(pts_arr, cells_arr)
        u_values[eval_map] = vals.flatten()
    
    u_grid = u_values.reshape((nx_out, ny_out))
    
    # Also extract initial condition on same grid
    u_init_values = np.full(nx_out * ny_out, np.nan)
    if len(points_on_proc) > 0:
        vals_init = u_initial_func.eval(pts_arr, cells_arr)
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
            "pc_type": "hypre",
            "rtol": 1e-10,
            "iterations": total_iterations,
            "dt": dt,
            "n_steps": n_steps,
            "time_scheme": "backward_euler",
        }
    }