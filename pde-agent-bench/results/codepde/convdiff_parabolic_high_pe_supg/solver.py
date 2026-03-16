import numpy as np
from dolfinx import mesh, fem, default_scalar_type, geometry
from dolfinx.fem import petsc
from mpi4py import MPI
import ufl
from petsc4py import PETSc


def solve(case_spec: dict) -> dict:
    # 1. Parse case_spec
    pde = case_spec["pde"]
    eps_val = pde["diffusion"]["eps"]
    beta = pde["convection"]["beta"]
    time_params = pde["time"]
    t_end = time_params["t_end"]
    dt_suggested = time_params["dt"]
    
    # Use a finer time step for accuracy at high Peclet
    dt = dt_suggested
    n_steps = int(round(t_end / dt))
    dt = t_end / n_steps  # exact division
    
    # Mesh resolution - need fine enough for high Pe
    N = 80
    
    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
    
    # P1 elements with SUPG stabilization
    degree = 1
    V = fem.functionspace(domain, ("Lagrange", degree))
    
    # Spatial coordinate and time
    x = ufl.SpatialCoordinate(domain)
    
    # Time as a constant that we update
    t_const = fem.Constant(domain, default_scalar_type(0.0))
    dt_const = fem.Constant(domain, default_scalar_type(dt))
    
    # Exact solution: u = exp(-t)*sin(2*pi*x)*sin(pi*y)
    pi = np.pi
    u_exact_ufl = ufl.exp(-t_const) * ufl.sin(2 * ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])
    
    # Source term: f = du/dt - eps*laplacian(u) + beta . grad(u)
    # du/dt = -exp(-t)*sin(2*pi*x)*sin(pi*y)
    # laplacian(u) = exp(-t)*(-4*pi^2 - pi^2)*sin(2*pi*x)*sin(pi*y) = -5*pi^2 * u
    # grad(u) = exp(-t)*(2*pi*cos(2*pi*x)*sin(pi*y), sin(2*pi*x)*pi*cos(pi*y))
    # f = -u - eps*(-5*pi^2)*u + beta[0]*exp(-t)*2*pi*cos(2*pi*x)*sin(pi*y) + beta[1]*exp(-t)*sin(2*pi*x)*pi*cos(pi*y)
    
    f_ufl = (
        -ufl.exp(-t_const) * ufl.sin(2 * ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])
        + eps_val * (4 * ufl.pi**2 + ufl.pi**2) * ufl.exp(-t_const) * ufl.sin(2 * ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])
        + beta[0] * ufl.exp(-t_const) * 2 * ufl.pi * ufl.cos(2 * ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])
        + beta[1] * ufl.exp(-t_const) * ufl.sin(2 * ufl.pi * x[0]) * ufl.pi * ufl.cos(ufl.pi * x[1])
    )
    
    # Trial and test functions
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    
    # Previous time step solution
    u_n = fem.Function(V)
    
    # Velocity vector
    beta_ufl = ufl.as_vector([fem.Constant(domain, default_scalar_type(beta[0])),
                               fem.Constant(domain, default_scalar_type(beta[1]))])
    eps_c = fem.Constant(domain, default_scalar_type(eps_val))
    
    # SUPG stabilization parameter
    h = ufl.CellDiameter(domain)
    beta_mag = ufl.sqrt(ufl.dot(beta_ufl, beta_ufl))
    Pe_cell = beta_mag * h / (2.0 * eps_c)
    # Classical SUPG parameter
    tau = h / (2.0 * beta_mag) * (ufl.conditional(ufl.gt(Pe_cell, 1.0),
                                                     1.0 - 1.0 / Pe_cell,
                                                     Pe_cell / 3.0))
    
    # Backward Euler: (u - u_n)/dt - eps*lap(u) + beta.grad(u) = f
    # Weak form (Galerkin part):
    # (u/dt, v) + eps*(grad(u), grad(v)) + (beta.grad(u), v) = (f, v) + (u_n/dt, v)
    
    a_galerkin = (
        u / dt_const * v * ufl.dx
        + eps_c * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
        + ufl.dot(beta_ufl, ufl.grad(u)) * v * ufl.dx
    )
    
    L_galerkin = (
        f_ufl * v * ufl.dx
        + u_n / dt_const * v * ufl.dx
    )
    
    # SUPG stabilization terms
    # Residual of the strong form applied to trial function (linearized):
    # R(u) = u/dt - eps*lap(u) + beta.grad(u) - f
    # For P1 elements, lap(u) = 0 within each cell
    # So strong residual: u/dt + beta.grad(u) - f - u_n/dt
    
    # SUPG test function modification: v_supg = tau * beta . grad(v)
    v_supg = tau * ufl.dot(beta_ufl, ufl.grad(v))
    
    a_supg = (
        u / dt_const * v_supg * ufl.dx
        + ufl.dot(beta_ufl, ufl.grad(u)) * v_supg * ufl.dx
        # For P1: -eps * div(grad(u)) = 0 in each element, so skip
    )
    
    L_supg = (
        f_ufl * v_supg * ufl.dx
        + u_n / dt_const * v_supg * ufl.dx
    )
    
    a_form = a_galerkin + a_supg
    L_form = L_galerkin + L_supg
    
    # Boundary conditions - Dirichlet with exact solution
    tdim = domain.topology.dim
    fdim = tdim - 1
    
    # All boundary
    domain.topology.create_connectivity(fdim, tdim)
    boundary_facets = mesh.exterior_facet_indices(domain.topology)
    
    bc_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    
    u_bc = fem.Function(V)
    
    # Set initial condition
    u_n.interpolate(lambda x_arr: np.exp(0.0) * np.sin(2 * np.pi * x_arr[0]) * np.sin(np.pi * x_arr[1]))
    
    # Store initial condition for output
    # Evaluate on grid
    nx_out, ny_out = 50, 50
    xs = np.linspace(0, 1, nx_out)
    ys = np.linspace(0, 1, ny_out)
    XX, YY = np.meshgrid(xs, ys, indexing='ij')
    points_2d = np.zeros((3, nx_out * ny_out))
    points_2d[0] = XX.ravel()
    points_2d[1] = YY.ravel()
    
    # Build point evaluation structures
    bb_tree = geometry.bb_tree(domain, tdim)
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
    
    points_on_proc = np.array(points_on_proc) if len(points_on_proc) > 0 else np.zeros((0, 3))
    cells_on_proc = np.array(cells_on_proc, dtype=np.int32) if len(cells_on_proc) > 0 else np.zeros(0, dtype=np.int32)
    
    def evaluate_on_grid(func):
        vals = np.full(nx_out * ny_out, np.nan)
        if len(points_on_proc) > 0:
            v_eval = func.eval(points_on_proc, cells_on_proc)
            for idx_local, idx_global in enumerate(eval_map):
                vals[idx_global] = v_eval[idx_local, 0]
        return vals.reshape((nx_out, ny_out))
    
    u_initial_grid = evaluate_on_grid(u_n)
    
    # Compile forms
    a_compiled = fem.form(a_form)
    L_compiled = fem.form(L_form)
    
    # Assemble matrix (changes each step because tau depends on nothing time-varying in the bilinear form,
    # but actually a_form has 1/dt which is constant, so A is constant if dt doesn't change)
    # Actually, a_form does NOT depend on t_const, so A is constant across time steps!
    
    # Set BC for initial time
    t_val = 0.0
    
    # We need to update BCs each time step
    # Create solution function
    u_sol = fem.Function(V)
    
    # Setup KSP solver
    A = petsc.assemble_matrix(a_compiled, bcs=[])
    A.assemble()
    
    b_vec = petsc.create_vector(L_compiled)
    
    solver = PETSc.KSP().create(domain.comm)
    solver.setOperators(A)
    solver.setType(PETSc.KSP.Type.GMRES)
    pc = solver.getPC()
    pc.setType(PETSc.PC.Type.ILU)
    solver.setTolerances(rtol=1e-10, atol=1e-12, max_it=2000)
    solver.setUp()
    
    total_iterations = 0
    
    # A is constant, assemble once
    # But we need BCs applied to A as well
    # Let's reassemble with BCs
    A.zeroEntries()
    petsc.assemble_matrix(A, a_compiled, bcs=[])
    A.assemble()
    
    for step in range(n_steps):
        t_val += dt
        t_const.value = t_val
        
        # Update BC
        u_bc_expr = fem.Expression(u_exact_ufl, V.element.interpolation_points)
        u_bc.interpolate(u_bc_expr)
        bc = fem.dirichletbc(u_bc, bc_dofs)
        
        # A needs to be reassembled with new BCs? No, A doesn't change.
        # But we need to apply BCs. For efficiency, reassemble A each step with BCs.
        # Actually for Dirichlet BCs, we need to apply lifting.
        
        # Reassemble A with BCs
        A.zeroEntries()
        petsc.assemble_matrix(A, a_compiled, bcs=[bc])
        A.assemble()
        solver.setOperators(A)
        
        # Assemble RHS
        with b_vec.localForm() as loc:
            loc.set(0.0)
        petsc.assemble_vector(b_vec, L_compiled)
        petsc.apply_lifting(b_vec, [a_compiled], bcs=[[bc]])
        b_vec.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        petsc.set_bc(b_vec, [bc])
        
        # Solve
        solver.solve(b_vec, u_sol.x.petsc_vec)
        u_sol.x.scatter_forward()
        
        total_iterations += solver.getIterationNumber()
        
        # Update u_n
        u_n.x.array[:] = u_sol.x.array[:]
    
    # Evaluate final solution on grid
    u_grid = evaluate_on_grid(u_sol)
    
    solver.destroy()
    A.destroy()
    b_vec.destroy()
    
    return {
        "u": u_grid,
        "u_initial": u_initial_grid,
        "solver_info": {
            "mesh_resolution": N,
            "element_degree": degree,
            "ksp_type": "gmres",
            "pc_type": "ilu",
            "rtol": 1e-10,
            "iterations": total_iterations,
            "dt": dt,
            "n_steps": n_steps,
            "time_scheme": "backward_euler",
        }
    }