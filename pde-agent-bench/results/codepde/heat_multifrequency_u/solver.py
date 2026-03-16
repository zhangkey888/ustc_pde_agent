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
    kappa = float(coeffs.get("kappa", 1.0))
    
    time_params = pde_config.get("time", {})
    t_end = float(time_params.get("t_end", 0.1))
    dt_suggested = float(time_params.get("dt", 0.01))
    scheme = time_params.get("scheme", "backward_euler")
    
    # Choose mesh resolution and element degree for accuracy
    N = 80
    degree = 2
    dt = 0.005  # Use smaller dt for accuracy with multi-frequency solution
    
    n_steps = int(round(t_end / dt))
    dt = t_end / n_steps  # Adjust dt to exactly hit t_end
    
    # 2. Create mesh
    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
    
    # 3. Function space
    V = fem.functionspace(domain, ("Lagrange", degree))
    
    # Spatial coordinate and time
    x = ufl.SpatialCoordinate(domain)
    pi = np.pi
    
    # Time as a constant that we update
    t_const = fem.Constant(domain, default_scalar_type(0.0))
    
    # Manufactured solution: u = exp(-t)*(sin(pi*x)*sin(pi*y) + 0.2*sin(6*pi*x)*sin(6*pi*y))
    # du/dt = -exp(-t)*(sin(pi*x)*sin(pi*y) + 0.2*sin(6*pi*x)*sin(6*pi*y))
    # -kappa * laplacian(u) = kappa * exp(-t) * (2*pi^2*sin(pi*x)*sin(pi*y) + 0.2*72*pi^2*sin(6*pi*x)*sin(6*pi*y))
    # f = du/dt - kappa*laplacian(u) (note: equation is du/dt - div(kappa*grad(u)) = f)
    # f = -exp(-t)*(sin(pi*x)*sin(pi*y) + 0.2*sin(6*pi*x)*sin(6*pi*y))
    #     + kappa*exp(-t)*(2*pi^2*sin(pi*x)*sin(pi*y) + 0.2*72*pi^2*sin(6*pi*x)*sin(6*pi*y))
    
    def u_exact_ufl(t_expr):
        return ufl.exp(-t_expr) * (
            ufl.sin(pi * x[0]) * ufl.sin(pi * x[1])
            + 0.2 * ufl.sin(6 * pi * x[0]) * ufl.sin(6 * pi * x[1])
        )
    
    def f_source_ufl(t_expr):
        s1 = ufl.sin(pi * x[0]) * ufl.sin(pi * x[1])
        s6 = ufl.sin(6 * pi * x[0]) * ufl.sin(6 * pi * x[1])
        # du/dt = -exp(-t)*(s1 + 0.2*s6)
        dudt = -ufl.exp(-t_expr) * (s1 + 0.2 * s6)
        # -laplacian(u) = exp(-t)*(2*pi^2*s1 + 0.2*72*pi^2*s6)
        neg_lap_u = ufl.exp(-t_expr) * (2.0 * pi**2 * s1 + 0.2 * 72.0 * pi**2 * s6)
        # f = du/dt + kappa * neg_lap_u
        return dudt + kappa * neg_lap_u
    
    # 4. Define functions
    u_n = fem.Function(V)  # solution at previous time step
    u_h = fem.Function(V)  # solution at current time step (will be the unknown)
    
    # Initial condition: u(x, 0) = sin(pi*x)*sin(pi*y) + 0.2*sin(6*pi*x)*sin(6*pi*y)
    u_init_expr = fem.Expression(
        ufl.sin(pi * x[0]) * ufl.sin(pi * x[1]) + 0.2 * ufl.sin(6 * pi * x[0]) * ufl.sin(6 * pi * x[1]),
        V.element.interpolation_points
    )
    u_n.interpolate(u_init_expr)
    
    # Store initial condition for output
    u_initial_grid = None
    
    # 5. Variational form for backward Euler
    # (u - u_n)/dt - kappa * laplacian(u) = f(t_{n+1})
    # Weak form: (u, v)/dt + kappa*(grad(u), grad(v)) = (u_n, v)/dt + (f, v)
    
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    
    dt_const = fem.Constant(domain, default_scalar_type(dt))
    kappa_const = fem.Constant(domain, default_scalar_type(kappa))
    
    f_expr = f_source_ufl(t_const)
    
    a = (u * v / dt_const + kappa_const * ufl.inner(ufl.grad(u), ufl.grad(v))) * ufl.dx
    L = (u_n * v / dt_const + f_expr * v) * ufl.dx
    
    # 6. Boundary conditions - u = g on boundary, g = exact solution at current time
    u_exact_expr_ufl = u_exact_ufl(t_const)
    
    tdim = domain.topology.dim
    fdim = tdim - 1
    
    # All boundary facets
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x_arr: np.ones(x_arr.shape[1], dtype=bool)
    )
    
    bc_func = fem.Function(V)
    bc_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc = fem.dirichletbc(bc_func, bc_dofs)
    
    # Compile forms
    a_form = fem.form(a)
    L_form = fem.form(L)
    
    # Assemble matrix (constant in time for this problem)
    A = petsc.assemble_matrix(a_form, bcs=[bc])
    A.assemble()
    
    # Create RHS vector
    b = fem.Function(V)
    b_vec = b.x.petsc_vec
    
    # Setup solver
    solver = PETSc.KSP().create(domain.comm)
    solver.setOperators(A)
    solver.setType(PETSc.KSP.Type.CG)
    pc = solver.getPC()
    pc.setType(PETSc.PC.Type.HYPRE)
    solver.setTolerances(rtol=1e-10, atol=1e-12, max_it=1000)
    solver.setUp()
    
    total_iterations = 0
    
    # 7. Time stepping
    t_current = 0.0
    for step in range(n_steps):
        t_current += dt
        t_const.value = t_current
        
        # Update boundary condition
        bc_expr = fem.Expression(u_exact_expr_ufl, V.element.interpolation_points)
        bc_func.interpolate(bc_expr)
        
        # Assemble RHS
        with b_vec.localForm() as loc:
            loc.set(0.0)
        petsc.assemble_vector(b_vec, L_form)
        petsc.apply_lifting(b_vec, [a_form], bcs=[[bc]])
        b_vec.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        petsc.set_bc(b_vec, [bc])
        
        # Solve
        solver.solve(b_vec, u_h.x.petsc_vec)
        u_h.x.scatter_forward()
        
        total_iterations += solver.getIterationNumber()
        
        # Update previous solution
        u_n.x.array[:] = u_h.x.array[:]
    
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
        vals = u_h.eval(pts_arr, cells_arr)
        u_values[eval_map] = vals.flatten()
    
    u_grid = u_values.reshape((nx_out, ny_out))
    
    # Also extract initial condition on the same grid
    # Recompute u_initial by interpolating the initial condition
    u_init_func = fem.Function(V)
    u_init_func.interpolate(u_init_expr)
    
    u_init_values = np.full(nx_out * ny_out, np.nan)
    if len(points_on_proc) > 0:
        vals_init = u_init_func.eval(pts_arr, cells_arr)
        u_init_values[eval_map] = vals_init.flatten()
    
    u_initial_grid = u_init_values.reshape((nx_out, ny_out))
    
    # Cleanup
    solver.destroy()
    A.destroy()
    
    return {
        "u": u_grid,
        "u_initial": u_initial_grid,
        "solver_info": {
            "mesh_resolution": N,
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