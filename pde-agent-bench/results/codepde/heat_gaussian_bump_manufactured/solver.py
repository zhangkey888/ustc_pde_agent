import numpy as np
from dolfinx import mesh, fem, default_scalar_type, geometry
from dolfinx.fem import petsc
from mpi4py import MPI
import ufl
from petsc4py import PETSc


def solve(case_spec: dict) -> dict:
    # 1. Parse case_spec
    pde = case_spec.get("pde", {})
    coeffs = pde.get("coefficients", {})
    kappa = float(coeffs.get("kappa", 1.0))
    
    time_params = pde.get("time", {})
    t_end = float(time_params.get("t_end", 0.1))
    dt_suggested = float(time_params.get("dt", 0.01))
    scheme = time_params.get("scheme", "backward_euler")
    
    # Use a finer dt for accuracy
    dt = dt_suggested
    n_steps = int(round(t_end / dt))
    dt = t_end / n_steps  # adjust to hit t_end exactly
    
    # Mesh resolution - use higher for accuracy
    nx = ny = 80
    element_degree = 2
    
    # 2. Create mesh
    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_square(comm, nx, ny, cell_type=mesh.CellType.triangle)
    
    # 3. Function space
    V = fem.functionspace(domain, ("Lagrange", element_degree))
    
    # Spatial coordinate
    x = ufl.SpatialCoordinate(domain)
    
    # Time as a constant that we update
    t_const = fem.Constant(domain, default_scalar_type(0.0))
    
    # Manufactured solution: u = exp(-t)*exp(-40*((x-0.5)^2 + (y-0.5)^2))
    # Compute source term f = du/dt - kappa * laplacian(u)
    # u_exact = exp(-t) * exp(-40*((x0-0.5)^2 + (x1-0.5)^2))
    # du/dt = -exp(-t) * exp(-40*(...)) = -u_exact
    # grad(u) = exp(-t) * exp(-40*(...)) * [-80*(x0-0.5), -80*(x1-0.5)]
    # laplacian(u) = exp(-t) * exp(-40*(...)) * [(-80 + 80^2*(x0-0.5)^2) + (-80 + 80^2*(x1-0.5)^2)]
    #             = exp(-t) * exp(-40*(...)) * [-160 + 6400*((x0-0.5)^2 + (x1-0.5)^2)]
    # f = du/dt - kappa * laplacian(u)
    #   = -u_exact - kappa * u_exact * [-160 + 6400*((x0-0.5)^2 + (x1-0.5)^2)]
    #   = u_exact * (-1 - kappa*(-160 + 6400*((x0-0.5)^2 + (x1-0.5)^2)))
    #   = u_exact * (-1 + 160*kappa - 6400*kappa*((x0-0.5)^2 + (x1-0.5)^2))
    
    r2 = (x[0] - 0.5)**2 + (x[1] - 0.5)**2
    u_exact_ufl = ufl.exp(-t_const) * ufl.exp(-40.0 * r2)
    
    f_expr = u_exact_ufl * (-1.0 + 160.0 * kappa - 6400.0 * kappa * r2)
    
    # 4. Define variational forms for backward Euler
    # (u^{n+1} - u^n)/dt - kappa * laplacian(u^{n+1}) = f^{n+1}
    # Weak form: (u^{n+1}, v)/dt + kappa*(grad(u^{n+1}), grad(v)) = (u^n, v)/dt + (f^{n+1}, v)
    
    u_trial = ufl.TrialFunction(V)
    v_test = ufl.TestFunction(V)
    
    u_n = fem.Function(V)  # solution at previous time step
    u_h = fem.Function(V)  # solution at current time step
    
    dt_const = fem.Constant(domain, default_scalar_type(dt))
    kappa_const = fem.Constant(domain, default_scalar_type(kappa))
    
    a = (u_trial * v_test / dt_const + kappa_const * ufl.inner(ufl.grad(u_trial), ufl.grad(v_test))) * ufl.dx
    L = (u_n * v_test / dt_const + f_expr * v_test) * ufl.dx
    
    # 5. Initial condition: u(x, 0) = exp(-40*((x-0.5)^2 + (y-0.5)^2))
    u_n.interpolate(lambda x_arr: np.exp(-40.0 * ((x_arr[0] - 0.5)**2 + (x_arr[1] - 0.5)**2)))
    
    # Store initial condition for output
    # We'll evaluate it on the grid later
    
    # 6. Boundary conditions - update each time step
    # g = exp(-t) * exp(-40*((x-0.5)^2 + (y-0.5)^2))
    u_bc_func = fem.Function(V)
    
    tdim = domain.topology.dim
    fdim = tdim - 1
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x_arr: np.ones(x_arr.shape[1], dtype=bool)
    )
    dofs_bc = fem.locate_dofs_topological(V, fdim, boundary_facets)
    
    def update_bc(t_val):
        u_bc_func.interpolate(
            lambda x_arr: np.exp(-t_val) * np.exp(-40.0 * ((x_arr[0] - 0.5)**2 + (x_arr[1] - 0.5)**2))
        )
    
    update_bc(dt)  # initial BC for first step
    bc = fem.dirichletbc(u_bc_func, dofs_bc)
    
    # 7. Assemble and solve using manual assembly for efficiency in time loop
    a_form = fem.form(a)
    L_form = fem.form(L)
    
    A = petsc.assemble_matrix(a_form, bcs=[bc])
    A.assemble()
    
    b = fem.petsc.create_vector(L_form)
    
    solver = PETSc.KSP().create(domain.comm)
    solver.setOperators(A)
    solver.setType(PETSc.KSP.Type.CG)
    pc = solver.getPC()
    pc.setType(PETSc.PC.Type.HYPRE)
    solver.setTolerances(rtol=1e-10, atol=1e-12, max_it=1000)
    solver.setUp()
    
    total_iterations = 0
    t_current = 0.0
    
    for step in range(n_steps):
        t_current += dt
        t_const.value = t_current
        
        # Update boundary condition
        update_bc(t_current)
        
        # Reassemble matrix (BCs might change sparsity pattern - but here they don't change structure)
        # Actually for changing Dirichlet values, we need to reassemble with new BC values
        A.zeroEntries()
        petsc.assemble_matrix(A, a_form, bcs=[bc])
        A.assemble()
        
        # Assemble RHS
        with b.localForm() as loc:
            loc.set(0.0)
        petsc.assemble_vector(b, L_form)
        petsc.apply_lifting(b, [a_form], bcs=[[bc]])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        petsc.set_bc(b, [bc])
        
        # Solve
        solver.solve(b, u_h.x.petsc_vec)
        u_h.x.scatter_forward()
        
        total_iterations += solver.getIterationNumber()
        
        # Update u_n for next step
        u_n.x.array[:] = u_h.x.array[:]
    
    # 8. Extract solution on 50x50 grid
    nx_out, ny_out = 50, 50
    xs = np.linspace(0.0, 1.0, nx_out)
    ys = np.linspace(0.0, 1.0, ny_out)
    XX, YY = np.meshgrid(xs, ys, indexing='ij')
    
    points_2d = np.zeros((3, nx_out * ny_out))
    points_2d[0, :] = XX.ravel()
    points_2d[1, :] = YY.ravel()
    
    bb_tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, points_2d.T)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points_2d.T)
    
    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    for i in range(points_2d.shape[1]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(points_2d[:, i])
            cells_on_proc.append(links[0])
            eval_map.append(i)
    
    u_values = np.full(nx_out * ny_out, np.nan)
    if len(points_on_proc) > 0:
        pts_arr = np.array(points_on_proc)
        cells_arr = np.array(cells_on_proc, dtype=np.int32)
        vals = u_h.eval(pts_arr, cells_arr)
        u_values[eval_map] = vals.flatten()
    
    u_grid = u_values.reshape((nx_out, ny_out))
    
    # Also extract initial condition on the same grid
    u_init_func = fem.Function(V)
    u_init_func.interpolate(lambda x_arr: np.exp(-40.0 * ((x_arr[0] - 0.5)**2 + (x_arr[1] - 0.5)**2)))
    
    u_init_values = np.full(nx_out * ny_out, np.nan)
    if len(points_on_proc) > 0:
        vals_init = u_init_func.eval(pts_arr, cells_arr)
        u_init_values[eval_map] = vals_init.flatten()
    
    u_initial_grid = u_init_values.reshape((nx_out, ny_out))
    
    # Cleanup PETSc objects
    solver.destroy()
    A.destroy()
    b.destroy()
    
    return {
        "u": u_grid,
        "u_initial": u_initial_grid,
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