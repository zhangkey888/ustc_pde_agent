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

    # The manufactured solution: u = exp(-t)*(exp(4*x)*sin(pi*y))
    # Reaction-diffusion: du/dt - eps*laplacian(u) + R(u) = f
    # For linear reaction-diffusion, R(u) = sigma*u typically
    # We need to figure out epsilon and reaction from the case spec
    
    # From the problem: -eps*laplacian(u) + R(u) = f (steady part)
    # Let's check what parameters are given
    params = pde.get("parameters", {})
    epsilon = params.get("epsilon", 1.0)
    
    # For a linear reaction term R(u) = sigma * u
    # Default sigma from params, or deduce from manufactured solution
    sigma = params.get("sigma", 1.0)
    
    # Choose mesh resolution and element degree for accuracy
    # The solution has exp(4*x) which can be large at x=1 (exp(4) ~ 54.6)
    # and sin(pi*y) oscillation. Need fine mesh especially near x=1.
    mesh_resolution = 100
    element_degree = 2
    dt = dt_suggested

    # Create mesh
    domain = mesh.create_unit_square(comm, mesh_resolution, mesh_resolution,
                                     cell_type=mesh.CellType.triangle)
    
    V = fem.functionspace(domain, ("Lagrange", element_degree))
    
    # Spatial coordinates and time
    x = ufl.SpatialCoordinate(domain)
    
    # Time parameter as a Constant
    t_const = fem.Constant(domain, ScalarType(0.0))
    
    # Manufactured solution in UFL
    u_exact_ufl = ufl.exp(-t_const) * (ufl.exp(4.0 * x[0]) * ufl.sin(ufl.pi * x[1]))
    
    # Compute the source term from the manufactured solution
    # du/dt = -exp(-t)*(exp(4*x)*sin(pi*y)) = -u_exact
    # laplacian(u) = exp(-t) * [16*exp(4x)*sin(pi*y) - pi^2*exp(4x)*sin(pi*y)]
    #              = exp(-t) * exp(4x)*sin(pi*y) * (16 - pi^2)
    #              = u_exact * (16 - pi^2)
    # So -eps*laplacian(u) = -eps * u_exact * (16 - pi^2)
    # R(u) = sigma * u_exact
    # f = du/dt - eps*laplacian(u) + R(u)
    #   = -u_exact - eps*(16 - pi^2)*u_exact + sigma*u_exact
    #   = u_exact * (-1 - eps*(16 - pi^2) + sigma)
    
    # But let's compute it symbolically to be safe
    # Actually, let me just compute f from the PDE symbolically
    
    # The PDE is: du/dt - eps * laplacian(u) + sigma * u = f
    # du/dt for manufactured solution:
    dudt = -ufl.exp(-t_const) * (ufl.exp(4.0 * x[0]) * ufl.sin(ufl.pi * x[1]))
    
    # grad(u_exact)
    grad_u_exact = ufl.grad(u_exact_ufl)
    laplacian_u_exact = ufl.div(grad_u_exact)
    
    # Source term
    f_expr = dudt - epsilon * laplacian_u_exact + sigma * u_exact_ufl
    
    # Trial and test functions
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    
    # Solution function
    u_n = fem.Function(V)  # solution at previous time step
    u_sol = fem.Function(V)  # solution at current time step
    
    # Initial condition: u(x, 0) = exp(0)*(exp(4*x)*sin(pi*y)) = exp(4*x)*sin(pi*y)
    t_const.value = 0.0
    u_n.interpolate(fem.Expression(u_exact_ufl, V.element.interpolation_points))
    
    # Store initial condition for output
    # Evaluate on grid
    nx_out, ny_out = 75, 75
    xs = np.linspace(0, 1, nx_out)
    ys = np.linspace(0, 1, ny_out)
    XX, YY = np.meshgrid(xs, ys, indexing='ij')
    points_2d = np.vstack([XX.ravel(), YY.ravel(), np.zeros(nx_out * ny_out)])
    
    def evaluate_on_grid(u_func):
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
        
        u_values = np.full(points_2d.shape[1], np.nan)
        if len(points_on_proc) > 0:
            pts = np.array(points_on_proc)
            cls = np.array(cells_on_proc, dtype=np.int32)
            vals = u_func.eval(pts, cls)
            u_values[eval_map] = vals.flatten()
        return u_values.reshape(nx_out, ny_out)
    
    u_initial = evaluate_on_grid(u_n)
    
    # Backward Euler time stepping
    # (u - u_n)/dt - eps*laplacian(u) + sigma*u = f
    # Weak form: (u/dt)*v*dx + eps*grad(u).grad(v)*dx + sigma*u*v*dx 
    #          = (u_n/dt)*v*dx + f*v*dx
    
    dt_const = fem.Constant(domain, ScalarType(dt))
    eps_const = fem.Constant(domain, ScalarType(epsilon))
    sigma_const = fem.Constant(domain, ScalarType(sigma))
    
    a = (u / dt_const * v * ufl.dx 
         + eps_const * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
         + sigma_const * u * v * ufl.dx)
    
    L = (u_n / dt_const * v * ufl.dx 
         + f_expr * v * ufl.dx)
    
    # Boundary conditions: u = g on boundary (from manufactured solution)
    tdim = domain.topology.dim
    fdim = tdim - 1
    
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool))
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    
    u_bc = fem.Function(V)
    
    # Compile forms
    a_form = fem.form(a)
    L_form = fem.form(L)
    
    # Assemble matrix (constant in time for linear problem with constant coefficients)
    A = petsc.assemble_matrix(a_form, bcs=[fem.dirichletbc(u_bc, boundary_dofs)])
    A.assemble()
    
    b = petsc.create_vector(L_form)
    
    # Set up KSP solver
    solver = PETSc.KSP().create(domain.comm)
    solver.setOperators(A)
    solver.setType(PETSc.KSP.Type.GMRES)
    pc = solver.getPC()
    pc.setType(PETSc.PC.Type.ILU)
    solver.setTolerances(rtol=1e-10, atol=1e-12, max_it=2000)
    solver.setUp()
    
    # Time stepping
    n_steps = int(np.round(t_end / dt))
    dt = t_end / n_steps  # adjust dt to hit t_end exactly
    dt_const.value = dt
    
    # Reassemble A since dt changed
    A.zeroEntries()
    petsc.assemble_matrix(A, a_form, bcs=[fem.dirichletbc(u_bc, boundary_dofs)])
    A.assemble()
    
    total_iterations = 0
    
    for step in range(n_steps):
        t_current = (step + 1) * dt
        t_const.value = t_current
        
        # Update boundary condition
        u_bc.interpolate(fem.Expression(u_exact_ufl, V.element.interpolation_points))
        bc = fem.dirichletbc(u_bc, boundary_dofs)
        
        # Assemble RHS
        with b.localForm() as loc:
            loc.set(0)
        petsc.assemble_vector(b, L_form)
        petsc.apply_lifting(b, [a_form], bcs=[[bc]])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        petsc.set_bc(b, [bc])
        
        # Solve
        solver.solve(b, u_sol.x.petsc_vec)
        u_sol.x.scatter_forward()
        
        total_iterations += solver.getIterationNumber()
        
        # Update for next step
        u_n.x.array[:] = u_sol.x.array[:]
    
    # Evaluate final solution on grid
    u_grid = evaluate_on_grid(u_sol)
    
    solver_info = {
        "mesh_resolution": mesh_resolution,
        "element_degree": element_degree,
        "ksp_type": "gmres",
        "pc_type": "ilu",
        "rtol": 1e-10,
        "iterations": total_iterations,
        "dt": dt,
        "n_steps": n_steps,
        "time_scheme": "backward_euler",
    }
    
    return {
        "u": u_grid,
        "u_initial": u_initial,
        "solver_info": solver_info,
    }