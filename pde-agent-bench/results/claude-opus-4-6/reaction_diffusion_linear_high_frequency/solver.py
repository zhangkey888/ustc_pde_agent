import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc

ScalarType = PETSc.ScalarType


def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD

    # Parse case spec
    pde = case_spec.get("pde", {})
    time_params = pde.get("time", {})
    t_end = time_params.get("t_end", 0.3)
    dt_suggested = time_params.get("dt", 0.005)
    scheme = time_params.get("scheme", "crank_nicolson")

    # Parameters
    mesh_resolution = 80
    element_degree = 2
    dt = dt_suggested
    epsilon = pde.get("epsilon", 1.0)
    # For reaction-diffusion, check if epsilon is specified
    if "coefficients" in pde:
        epsilon = pde["coefficients"].get("epsilon", 1.0)
    
    # If epsilon not found in typical places, default to 1.0
    # The manufactured solution: u = exp(-t)*(sin(4*pi*x)*sin(3*pi*y))
    # For reaction-diffusion: du/dt - eps*laplacian(u) + R(u) = f
    # We need to figure out R(u) and epsilon from the case spec
    
    # Try to get reaction type
    reaction_type = "linear"
    reaction_coeff = 1.0  # default linear reaction coefficient
    if "reaction" in pde:
        reaction_type = pde["reaction"].get("type", "linear")
        reaction_coeff = pde["reaction"].get("coefficient", 1.0)
    
    # If epsilon not explicitly given, assume 1.0
    if "coefficients" in pde:
        epsilon = pde["coefficients"].get("epsilon", 1.0)
        reaction_coeff = pde["coefficients"].get("reaction_coefficient", reaction_coeff)

    # Create mesh
    nx = ny = mesh_resolution
    domain = mesh.create_unit_square(comm, nx, ny, cell_type=mesh.CellType.triangle)
    
    # Function space
    V = fem.functionspace(domain, ("Lagrange", element_degree))
    
    # Spatial coordinates and time
    x = ufl.SpatialCoordinate(domain)
    
    # Time constant
    t_const = fem.Constant(domain, ScalarType(0.0))
    
    # Manufactured solution: u_exact = exp(-t) * sin(4*pi*x) * sin(3*pi*y)
    pi = ufl.pi
    u_exact_ufl = ufl.exp(-t_const) * ufl.sin(4 * pi * x[0]) * ufl.sin(3 * pi * x[1])
    
    # Compute source term f for: du/dt - eps*laplacian(u) + R(u) = f
    # du/dt = -exp(-t)*sin(4*pi*x)*sin(3*pi*y)
    # laplacian(u) = exp(-t)*(-16*pi^2 - 9*pi^2)*sin(4*pi*x)*sin(3*pi*y) = -25*pi^2*exp(-t)*sin(...)
    # -eps*laplacian(u) = 25*eps*pi^2*exp(-t)*sin(...)
    # For linear reaction R(u) = reaction_coeff * u
    # f = du/dt - eps*laplacian(u) + R(u)
    # f = -exp(-t)*sin(...) + 25*eps*pi^2*exp(-t)*sin(...) + reaction_coeff*exp(-t)*sin(...)
    # f = exp(-t)*sin(...)*(-1 + 25*eps*pi^2 + reaction_coeff)
    
    # But let's compute it symbolically via UFL to be safe
    # du/dt = -u_exact
    du_dt_ufl = -ufl.exp(-t_const) * ufl.sin(4 * pi * x[0]) * ufl.sin(3 * pi * x[1])
    
    # Gradient and laplacian
    grad_u_exact = ufl.grad(u_exact_ufl)
    laplacian_u_exact = ufl.div(grad_u_exact)
    
    # Reaction term (linear)
    R_u_exact = reaction_coeff * u_exact_ufl
    
    # Source term
    f_ufl = du_dt_ufl - epsilon * laplacian_u_exact + R_u_exact
    
    # Trial and test functions
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    
    # Solution at current and previous time step
    u_n = fem.Function(V)  # solution at previous time step
    u_h = fem.Function(V)  # solution at current time step
    
    # Set initial condition: u(x, 0) = sin(4*pi*x)*sin(3*pi*y)
    t_const.value = 0.0
    u_init_expr = fem.Expression(u_exact_ufl, V.element.interpolation_points)
    u_n.interpolate(u_init_expr)
    
    # Store initial condition for output
    # Evaluate on grid
    nx_out, ny_out = 70, 70
    xs = np.linspace(0, 1, nx_out)
    ys = np.linspace(0, 1, ny_out)
    X, Y = np.meshgrid(xs, ys, indexing='ij')
    points_2d = np.vstack([X.ravel(), Y.ravel(), np.zeros(nx_out * ny_out)])
    
    # Boundary conditions
    tdim = domain.topology.dim
    fdim = tdim - 1
    
    # All boundary
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
    )
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    
    # BC function (time-dependent)
    u_bc = fem.Function(V)
    
    def update_bc(t_val):
        t_const.value = t_val
        bc_expr = fem.Expression(u_exact_ufl, V.element.interpolation_points)
        u_bc.interpolate(bc_expr)
    
    bc = fem.dirichletbc(u_bc, boundary_dofs)
    bcs = [bc]
    
    # Time stepping parameters
    n_steps = int(round(t_end / dt))
    dt_actual = t_end / n_steps
    dt_c = fem.Constant(domain, ScalarType(dt_actual))
    eps_c = fem.Constant(domain, ScalarType(epsilon))
    react_c = fem.Constant(domain, ScalarType(reaction_coeff))
    
    # Crank-Nicolson scheme:
    # (u^{n+1} - u^n)/dt - eps * 0.5*(laplacian(u^{n+1}) + laplacian(u^n)) 
    #   + 0.5*(R(u^{n+1}) + R(u^n)) = 0.5*(f^{n+1} + f^n)
    # 
    # Weak form (linear reaction):
    # (u^{n+1}, v)/dt + 0.5*eps*(grad(u^{n+1}), grad(v)) + 0.5*react*u^{n+1}*v
    # = (u^n, v)/dt - 0.5*eps*(grad(u^n), grad(v)) - 0.5*react*u^n*v + 0.5*(f^{n+1} + f^n)*v
    
    theta = 0.5  # Crank-Nicolson
    
    # Source term at two time levels
    t_n_const = fem.Constant(domain, ScalarType(0.0))  # t^n
    t_np1_const = fem.Constant(domain, ScalarType(dt_actual))  # t^{n+1}
    
    # We need f at t^n and t^{n+1}
    # Redefine f using the time constants
    u_exact_n = ufl.exp(-t_n_const) * ufl.sin(4 * pi * x[0]) * ufl.sin(3 * pi * x[1])
    u_exact_np1 = ufl.exp(-t_np1_const) * ufl.sin(4 * pi * x[0]) * ufl.sin(3 * pi * x[1])
    
    du_dt_n = -u_exact_n
    du_dt_np1 = -u_exact_np1
    
    grad_u_exact_n = ufl.grad(u_exact_n)
    laplacian_u_exact_n = ufl.div(grad_u_exact_n)
    grad_u_exact_np1 = ufl.grad(u_exact_np1)
    laplacian_u_exact_np1 = ufl.div(grad_u_exact_np1)
    
    f_n = du_dt_n - epsilon * laplacian_u_exact_n + reaction_coeff * u_exact_n
    f_np1 = du_dt_np1 - epsilon * laplacian_u_exact_np1 + reaction_coeff * u_exact_np1
    
    # Bilinear form (LHS)
    a = (ufl.inner(u, v) / dt_c * ufl.dx
         + theta * eps_c * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
         + theta * react_c * ufl.inner(u, v) * ufl.dx)
    
    # Linear form (RHS)
    L = (ufl.inner(u_n, v) / dt_c * ufl.dx
         - (1.0 - theta) * eps_c * ufl.inner(ufl.grad(u_n), ufl.grad(v)) * ufl.dx
         - (1.0 - theta) * react_c * ufl.inner(u_n, v) * ufl.dx
         + (theta * f_np1 + (1.0 - theta) * f_n) * v * ufl.dx)
    
    # Compile forms
    a_form = fem.form(a)
    L_form = fem.form(L)
    
    # Assemble matrix (constant in time for linear reaction)
    A = petsc.assemble_matrix(a_form, bcs=bcs)
    A.assemble()
    
    b = petsc.create_vector(L_form)
    
    # Setup solver
    solver = PETSc.KSP().create(domain.comm)
    solver.setOperators(A)
    solver.setType(PETSc.KSP.Type.GMRES)
    pc = solver.getPC()
    pc.setType(PETSc.PC.Type.ILU)
    solver.setTolerances(rtol=1e-10, atol=1e-12, max_it=1000)
    solver.setUp()
    
    total_iterations = 0
    
    # Time stepping loop
    t = 0.0
    for step in range(n_steps):
        t_old = t
        t += dt_actual
        
        # Update time constants
        t_n_const.value = t_old
        t_np1_const.value = t
        
        # Update boundary condition
        update_bc(t)
        
        # Assemble RHS
        with b.localForm() as loc:
            loc.set(0)
        petsc.assemble_vector(b, L_form)
        petsc.apply_lifting(b, [a_form], bcs=[bcs])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        petsc.set_bc(b, bcs)
        
        # Solve
        solver.solve(b, u_h.x.petsc_vec)
        u_h.x.scatter_forward()
        
        total_iterations += solver.getIterationNumber()
        
        # Update previous solution
        u_n.x.array[:] = u_h.x.array[:]
    
    # Evaluate solution on output grid
    bb_tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, points_2d.T)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points_2d.T)
    
    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    for i in range(points_2d.shape[1]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(points_2d.T[i])
            cells_on_proc.append(links[0])
            eval_map.append(i)
    
    u_values = np.full(points_2d.shape[1], np.nan)
    if len(points_on_proc) > 0:
        vals = u_h.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u_values[eval_map] = vals.flatten()
    
    u_grid = u_values.reshape((nx_out, ny_out))
    
    # Also evaluate initial condition for output
    t_const.value = 0.0
    u_init_func = fem.Function(V)
    u_init_func.interpolate(fem.Expression(
        ufl.sin(4 * pi * x[0]) * ufl.sin(3 * pi * x[1]),
        V.element.interpolation_points
    ))
    
    u_init_values = np.full(points_2d.shape[1], np.nan)
    if len(points_on_proc) > 0:
        vals_init = u_init_func.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u_init_values[eval_map] = vals_init.flatten()
    
    u_initial_grid = u_init_values.reshape((nx_out, ny_out))
    
    solver.destroy()
    A.destroy()
    b.destroy()
    
    return {
        "u": u_grid,
        "u_initial": u_initial_grid,
        "solver_info": {
            "mesh_resolution": mesh_resolution,
            "element_degree": element_degree,
            "ksp_type": "gmres",
            "pc_type": "ilu",
            "rtol": 1e-10,
            "iterations": total_iterations,
            "dt": dt_actual,
            "n_steps": n_steps,
            "time_scheme": "crank_nicolson",
        }
    }