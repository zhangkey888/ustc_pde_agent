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
    time_params = pde.get("time", {})
    
    kappa = float(coeffs.get("kappa", 1.0))
    t_end = float(time_params.get("t_end", 0.08))
    dt_suggested = float(time_params.get("dt", 0.008))
    scheme = time_params.get("scheme", "backward_euler")
    
    # Use a finer mesh and smaller dt for accuracy with high-frequency solution
    nx = ny = 80
    degree = 2
    dt = dt_suggested / 2.0  # Use half the suggested dt for better accuracy
    n_steps = int(round(t_end / dt))
    dt = t_end / n_steps  # Adjust to hit t_end exactly
    
    # 2. Create mesh
    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_square(comm, nx, ny, cell_type=mesh.CellType.triangle)
    
    # 3. Function space
    V = fem.functionspace(domain, ("Lagrange", degree))
    
    # Spatial coordinate and pi
    x = ufl.SpatialCoordinate(domain)
    pi = np.pi
    
    # Time as a constant that we update
    t_const = fem.Constant(domain, default_scalar_type(0.0))
    
    # Manufactured solution: u = exp(-t)*sin(3*pi*x)*sin(3*pi*y)
    # du/dt = -exp(-t)*sin(3*pi*x)*sin(3*pi*y)
    # -kappa * laplacian(u) = kappa * 2*(3*pi)^2 * exp(-t)*sin(3*pi*x)*sin(3*pi*y)
    # f = du/dt - kappa*laplacian(u) ... wait, the equation is du/dt - div(kappa*grad(u)) = f
    # So f = du/dt - kappa*laplacian(u)
    # du/dt = -exp(-t)*sin(3*pi*x)*sin(3*pi*y)
    # laplacian(u) = -2*(3*pi)^2 * exp(-t)*sin(3*pi*x)*sin(3*pi*y)
    # -kappa*laplacian(u) = 2*kappa*(3*pi)^2 * exp(-t)*sin(3*pi*x)*sin(3*pi*y)
    # Wait: div(kappa*grad(u)) = kappa * laplacian(u) = kappa * (-2*(3*pi)^2) * exp(-t)*sin(...)
    # f = du/dt - div(kappa*grad(u)) = -exp(-t)*sin(...) - kappa*(-2*(3*pi)^2)*exp(-t)*sin(...)
    # f = exp(-t)*sin(3*pi*x)*sin(3*pi*y) * (-1 + 2*kappa*(3*pi)^2)
    # f = exp(-t)*sin(3*pi*x)*sin(3*pi*y) * (18*kappa*pi^2 - 1)
    
    u_exact_ufl = ufl.exp(-t_const) * ufl.sin(3 * pi * x[0]) * ufl.sin(3 * pi * x[1])
    f_coeff = 18.0 * kappa * pi**2 - 1.0
    f_expr = f_coeff * ufl.exp(-t_const) * ufl.sin(3 * pi * x[0]) * ufl.sin(3 * pi * x[1])
    
    # 4. Trial and test functions
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    
    # Previous solution
    u_n = fem.Function(V)
    
    # Initial condition: u(x,0) = sin(3*pi*x)*sin(3*pi*y)
    u_n.interpolate(lambda X: np.sin(3 * pi * X[0]) * np.sin(3 * pi * X[1]))
    
    # Store initial condition for output
    u_initial_func = fem.Function(V)
    u_initial_func.x.array[:] = u_n.x.array[:]
    
    # 5. Backward Euler: (u - u_n)/dt - kappa*laplacian(u) = f(t_{n+1})
    # Weak form: (u/dt)*v*dx + kappa*grad(u)·grad(v)*dx = (u_n/dt)*v*dx + f*v*dx
    dt_const = fem.Constant(domain, default_scalar_type(dt))
    kappa_const = fem.Constant(domain, default_scalar_type(kappa))
    
    a = (u / dt_const) * v * ufl.dx + kappa_const * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = (u_n / dt_const) * v * ufl.dx + f_expr * v * ufl.dx
    
    # 6. Boundary conditions - u = exact solution on boundary
    # g = exp(-t)*sin(3*pi*x)*sin(3*pi*y) which is 0 on all boundaries of [0,1]^2
    # since sin(0) = sin(3*pi) = 0. So homogeneous Dirichlet BC.
    tdim = domain.topology.dim
    fdim = tdim - 1
    
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda X: np.ones(X.shape[1], dtype=bool)
    )
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    
    u_bc = fem.Function(V)
    u_bc.x.array[:] = 0.0  # Homogeneous BC for all times
    bc = fem.dirichletbc(u_bc, boundary_dofs)
    
    # 7. Assemble and solve with manual assembly for efficiency in time loop
    a_form = fem.form(a)
    L_form = fem.form(L)
    
    A = petsc.assemble_matrix(a_form, bcs=[bc])
    A.assemble()
    
    b = fem.Function(V)
    b_vec = b.x.petsc_vec
    
    solver = PETSc.KSP().create(domain.comm)
    solver.setOperators(A)
    solver.setType(PETSc.KSP.Type.CG)
    pc = solver.getPC()
    pc.setType(PETSc.PC.Type.HYPRE)
    solver.setTolerances(rtol=1e-10, atol=1e-12, max_it=1000)
    solver.setUp()
    
    u_sol = fem.Function(V)
    
    total_iterations = 0
    
    # Time stepping
    t = 0.0
    for step in range(n_steps):
        t += dt
        t_const.value = t
        
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
    
    # 8. Extract solution on 50x50 grid
    n_eval = 50
    xs = np.linspace(0, 1, n_eval)
    ys = np.linspace(0, 1, n_eval)
    XX, YY = np.meshgrid(xs, ys, indexing='ij')
    
    points = np.zeros((3, n_eval * n_eval))
    points[0, :] = XX.ravel()
    points[1, :] = YY.ravel()
    
    bb_tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, points.T)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points.T)
    
    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    for i in range(n_eval * n_eval):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(points.T[i])
            cells_on_proc.append(links[0])
            eval_map.append(i)
    
    u_values = np.full(n_eval * n_eval, np.nan)
    if len(points_on_proc) > 0:
        vals = u_sol.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u_values[eval_map] = vals.flatten()
    
    u_grid = u_values.reshape((n_eval, n_eval))
    
    # Also extract initial condition on same grid
    u_init_values = np.full(n_eval * n_eval, np.nan)
    if len(points_on_proc) > 0:
        vals_init = u_initial_func.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u_init_values[eval_map] = vals_init.flatten()
    
    u_initial_grid = u_init_values.reshape((n_eval, n_eval))
    
    # Cleanup
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