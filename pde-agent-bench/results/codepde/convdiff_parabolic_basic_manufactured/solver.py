import numpy as np
from dolfinx import mesh, fem, default_scalar_type, geometry
from dolfinx.fem import petsc
from mpi4py import MPI
import ufl
from petsc4py import PETSc


def solve(case_spec: dict) -> dict:
    # 1. Parse case_spec
    pde = case_spec.get("pde", {})
    params = pde.get("params", {})
    epsilon = params.get("epsilon", 0.1)
    beta = params.get("beta", [1.0, 0.5])
    
    time_params = pde.get("time", {})
    t_end = time_params.get("t_end", 0.1)
    dt_suggested = time_params.get("dt", 0.02)
    scheme = time_params.get("scheme", "backward_euler")
    
    # Use a finer mesh and smaller dt for accuracy
    N = 80
    degree = 1
    dt = 0.005  # smaller dt for accuracy
    n_steps = int(round(t_end / dt))
    dt = t_end / n_steps  # adjust to hit t_end exactly
    
    # 2. Create mesh
    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
    
    # 3. Function space
    V = fem.functionspace(domain, ("Lagrange", degree))
    
    # Spatial coordinate and time
    x = ufl.SpatialCoordinate(domain)
    
    # 4. Define exact solution and source term
    # u_exact = exp(-t)*sin(pi*x)*sin(pi*y)
    # du/dt = -exp(-t)*sin(pi*x)*sin(pi*y)
    # grad(u) = exp(-t)*[pi*cos(pi*x)*sin(pi*y), pi*sin(pi*x)*cos(pi*y)]
    # laplacian(u) = -2*pi^2*exp(-t)*sin(pi*x)*sin(pi*y)
    # f = du/dt - eps*laplacian(u) + beta.grad(u)
    # f = -exp(-t)*sin(pi*x)*sin(pi*y) + eps*2*pi^2*exp(-t)*sin(pi*x)*sin(pi*y) 
    #     + beta[0]*exp(-t)*pi*cos(pi*x)*sin(pi*y) + beta[1]*exp(-t)*pi*sin(pi*x)*cos(pi*y)
    
    t_const = fem.Constant(domain, default_scalar_type(0.0))
    pi = np.pi
    
    # UFL expressions
    sin_px = ufl.sin(ufl.pi * x[0])
    sin_py = ufl.sin(ufl.pi * x[1])
    cos_px = ufl.cos(ufl.pi * x[0])
    cos_py = ufl.cos(ufl.pi * x[1])
    exp_t = ufl.exp(-t_const)
    
    u_exact_ufl = exp_t * sin_px * sin_py
    
    f_expr = (
        -exp_t * sin_px * sin_py
        + epsilon * 2.0 * ufl.pi**2 * exp_t * sin_px * sin_py
        + beta[0] * exp_t * ufl.pi * cos_px * sin_py
        + beta[1] * exp_t * ufl.pi * sin_px * cos_py
    )
    
    # Trial and test functions
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    
    # Previous solution
    u_n = fem.Function(V)
    
    # Initial condition: u(x,0) = sin(pi*x)*sin(pi*y)
    u_n.interpolate(lambda x: np.sin(np.pi * x[0]) * np.sin(np.pi * x[1]))
    
    # Store initial condition for output
    u_initial_func = fem.Function(V)
    u_initial_func.x.array[:] = u_n.x.array[:]
    
    # Velocity vector
    beta_ufl = ufl.as_vector([fem.Constant(domain, default_scalar_type(beta[0])),
                               fem.Constant(domain, default_scalar_type(beta[1]))])
    
    dt_const = fem.Constant(domain, default_scalar_type(dt))
    eps_const = fem.Constant(domain, default_scalar_type(epsilon))
    
    # SUPG stabilization
    h = ufl.CellDiameter(domain)
    beta_norm = ufl.sqrt(ufl.dot(beta_ufl, beta_ufl))
    Pe_cell = beta_norm * h / (2.0 * eps_const)
    tau = h / (2.0 * beta_norm) * (1.0 / ufl.tanh(Pe_cell) - 1.0 / Pe_cell)
    
    # Backward Euler: (u - u_n)/dt - eps*laplacian(u) + beta.grad(u) = f
    # Weak form:
    # (u/dt, v) + eps*(grad(u), grad(v)) + (beta.grad(u), v) = (f, v) + (u_n/dt, v)
    
    # Standard Galerkin terms
    a_standard = (
        u / dt_const * v * ufl.dx
        + eps_const * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
        + ufl.dot(beta_ufl, ufl.grad(u)) * v * ufl.dx
    )
    
    L_standard = (
        u_n / dt_const * v * ufl.dx
        + f_expr * v * ufl.dx
    )
    
    # SUPG stabilization terms
    # Residual of strong form applied to trial function (linearized):
    # R = u/dt - eps*laplacian(u) + beta.grad(u) - f - u_n/dt
    # For SUPG with backward Euler, the test function modification is: v + tau * beta.grad(v)
    # We add: tau * (u/dt + beta.grad(u)) * (beta.grad(v)) * dx  on LHS
    # and:    tau * (f + u_n/dt) * (beta.grad(v)) * dx  on RHS
    # Note: we skip the diffusion part of the residual in SUPG (consistent for linear elements)
    
    supg_test = tau * ufl.dot(beta_ufl, ufl.grad(v))
    
    a_supg = (
        u / dt_const * supg_test * ufl.dx
        + ufl.dot(beta_ufl, ufl.grad(u)) * supg_test * ufl.dx
    )
    
    L_supg = (
        u_n / dt_const * supg_test * ufl.dx
        + f_expr * supg_test * ufl.dx
    )
    
    a_form = a_standard + a_supg
    L_form = L_standard + L_supg
    
    # 5. Boundary conditions - u = g = exp(-t)*sin(pi*x)*sin(pi*y) on boundary
    # On the unit square boundary, sin(pi*x)*sin(pi*y) = 0, so g = 0 for all t
    tdim = domain.topology.dim
    fdim = tdim - 1
    
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
    )
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc = fem.dirichletbc(default_scalar_type(0.0), boundary_dofs, V)
    bcs = [bc]
    
    # 6. Compile forms
    a_compiled = fem.form(a_form)
    L_compiled = fem.form(L_form)
    
    # Assemble matrix (constant in time since coefficients don't change except t_const in f)
    # Actually, t_const appears in f_expr which is in L, so we need to reassemble L each step
    # But a_form doesn't depend on t_const, so A is constant
    A = petsc.assemble_matrix(a_compiled, bcs=bcs)
    A.assemble()
    
    b = fem.petsc.create_vector(L_compiled)
    
    # Setup solver
    solver = PETSc.KSP().create(comm)
    solver.setOperators(A)
    solver.setType(PETSc.KSP.Type.GMRES)
    pc = solver.getPC()
    pc.setType(PETSc.PC.Type.ILU)
    solver.setTolerances(rtol=1e-10, atol=1e-12, max_it=1000)
    solver.setUp()
    
    # Solution function
    u_sol = fem.Function(V)
    u_sol.x.array[:] = u_n.x.array[:]
    
    total_iterations = 0
    
    # 7. Time stepping
    t = 0.0
    for step in range(n_steps):
        t += dt
        t_const.value = t
        
        # Assemble RHS
        with b.localForm() as loc:
            loc.set(0.0)
        petsc.assemble_vector(b, L_compiled)
        petsc.apply_lifting(b, [a_compiled], bcs=[bcs])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        petsc.set_bc(b, bcs)
        
        # Solve
        solver.solve(b, u_sol.x.petsc_vec)
        u_sol.x.scatter_forward()
        
        total_iterations += solver.getIterationNumber()
        
        # Update previous solution
        u_n.x.array[:] = u_sol.x.array[:]
    
    # 8. Extract solution on 50x50 grid
    nx_out, ny_out = 50, 50
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