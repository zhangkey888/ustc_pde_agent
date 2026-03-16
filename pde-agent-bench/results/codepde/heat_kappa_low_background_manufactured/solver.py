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
    dt_suggested = time_params.get("dt", 0.01)
    
    # Use a smaller dt for accuracy
    dt_val = 0.005
    n_steps = int(round(t_end / dt_val))
    dt_val = t_end / n_steps  # exact division
    
    # Mesh resolution
    N = 80
    
    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
    
    # Function space - P1
    degree = 1
    V = fem.functionspace(domain, ("Lagrange", degree))
    
    # Spatial coordinates
    x = ufl.SpatialCoordinate(domain)
    
    # Kappa: 0.2 + exp(-120*((x-0.55)**2 + (y-0.45)**2))
    kappa_expr = 0.2 + ufl.exp(-120.0 * ((x[0] - 0.55)**2 + (x[1] - 0.45)**2))
    
    # Manufactured solution: u_exact = exp(-t)*sin(pi*x)*sin(pi*y)
    # du/dt = -exp(-t)*sin(pi*x)*sin(pi*y)
    # grad(u) = exp(-t) * (pi*cos(pi*x)*sin(pi*y), sin(pi*x)*pi*cos(pi*y))
    # div(kappa * grad(u)) = kappa * (-2*pi^2*exp(-t)*sin(pi*x)*sin(pi*y)) + grad(kappa) . grad(u)
    #
    # f = du/dt - div(kappa * grad(u))
    # We'll compute f symbolically using UFL
    
    # Time constant
    t_const = fem.Constant(domain, default_scalar_type(0.0))
    dt_const = fem.Constant(domain, default_scalar_type(dt_val))
    
    # Exact solution as UFL expression (parametric in t_const)
    pi = ufl.pi
    u_exact_ufl = ufl.exp(-t_const) * ufl.sin(pi * x[0]) * ufl.sin(pi * x[1])
    
    # Source term: f = du/dt - div(kappa * grad(u))
    # du/dt = -exp(-t)*sin(pi*x)*sin(pi*y) = -u_exact
    dudt = -ufl.exp(-t_const) * ufl.sin(pi * x[0]) * ufl.sin(pi * x[1])
    
    # grad(u_exact)
    grad_u_exact = ufl.as_vector([
        ufl.exp(-t_const) * pi * ufl.cos(pi * x[0]) * ufl.sin(pi * x[1]),
        ufl.exp(-t_const) * ufl.sin(pi * x[0]) * pi * ufl.cos(pi * x[1])
    ])
    
    # div(kappa * grad(u_exact))
    # We need to compute this carefully. Let's use UFL's div
    # kappa * grad(u_exact) is a vector
    kappa_grad_u = kappa_expr * grad_u_exact
    div_kappa_grad_u = ufl.div(kappa_grad_u)
    
    f_expr = dudt - div_kappa_grad_u
    
    # Trial and test functions
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    
    # Previous solution
    u_n = fem.Function(V)
    
    # Initial condition: u(x, 0) = sin(pi*x)*sin(pi*y)
    u_n.interpolate(lambda X: np.sin(np.pi * X[0]) * np.sin(np.pi * X[1]))
    
    # Store initial condition for output
    u_initial_func = fem.Function(V)
    u_initial_func.x.array[:] = u_n.x.array[:]
    
    # Backward Euler: (u - u_n)/dt - div(kappa * grad(u)) = f(t_{n+1})
    # Weak form: (u/dt)*v*dx + kappa*inner(grad(u), grad(v))*dx = (u_n/dt)*v*dx + f*v*dx
    
    a_form = (u / dt_const) * v * ufl.dx + kappa_expr * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
    L_form = (u_n / dt_const) * v * ufl.dx + f_expr * v * ufl.dx
    
    # Boundary conditions: u = g = u_exact on boundary
    # We need to update g at each time step
    tdim = domain.topology.dim
    fdim = tdim - 1
    
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda X: np.ones(X.shape[1], dtype=bool)
    )
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    
    u_bc = fem.Function(V)
    # At t=0, boundary values
    u_bc.interpolate(lambda X: np.exp(0.0) * np.sin(np.pi * X[0]) * np.sin(np.pi * X[1]))
    bc = fem.dirichletbc(u_bc, boundary_dofs)
    
    # Compile forms
    a_compiled = fem.form(a_form)
    L_compiled = fem.form(L_form)
    
    # Assemble matrix (constant in time since kappa doesn't depend on t)
    A = petsc.assemble_matrix(a_compiled, bcs=[bc])
    A.assemble()
    
    # Create RHS vector
    b = fem.Function(V)
    b_vec = b.x.petsc_vec
    
    # Solution function
    u_sol = fem.Function(V)
    
    # Setup KSP solver
    solver = PETSc.KSP().create(domain.comm)
    solver.setOperators(A)
    solver.setType(PETSc.KSP.Type.CG)
    pc = solver.getPC()
    pc.setType(PETSc.PC.Type.ILU)
    solver.setTolerances(rtol=1e-10, atol=1e-12, max_it=1000)
    solver.setUp()
    
    total_iterations = 0
    
    # Time stepping
    t = 0.0
    for step in range(n_steps):
        t += dt_val
        
        # Update time constant for source term
        t_const.value = t
        
        # Update boundary condition
        t_current = t
        u_bc.interpolate(lambda X, tc=t_current: np.exp(-tc) * np.sin(np.pi * X[0]) * np.sin(np.pi * X[1]))
        
        # Assemble RHS
        with b_vec.localForm() as loc:
            loc.set(0.0)
        petsc.assemble_vector(b_vec, L_compiled)
        
        # Apply lifting and BCs
        petsc.apply_lifting(b_vec, [a_compiled], bcs=[[bc]])
        b_vec.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        petsc.set_bc(b_vec, [bc])
        
        # Solve
        solver.solve(b_vec, u_sol.x.petsc_vec)
        u_sol.x.scatter_forward()
        
        total_iterations += solver.getIterationNumber()
        
        # Update previous solution
        u_n.x.array[:] = u_sol.x.array[:]
    
    # Extract solution on 50x50 grid
    nx_out, ny_out = 50, 50
    xs = np.linspace(0, 1, nx_out)
    ys = np.linspace(0, 1, ny_out)
    XX, YY = np.meshgrid(xs, ys, indexing='ij')
    points = np.zeros((3, nx_out * ny_out))
    points[0, :] = XX.ravel()
    points[1, :] = YY.ravel()
    
    # Point evaluation
    bb_tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, points.T)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points.T)
    
    def eval_on_grid(func, points, colliding_cells, nx, ny):
        points_on_proc = []
        cells_on_proc = []
        eval_map = []
        n_pts = points.shape[1]
        for i in range(n_pts):
            links = colliding_cells.links(i)
            if len(links) > 0:
                points_on_proc.append(points[:, i])
                cells_on_proc.append(links[0])
                eval_map.append(i)
        
        u_values = np.full(n_pts, np.nan)
        if len(points_on_proc) > 0:
            pts_arr = np.array(points_on_proc)
            cells_arr = np.array(cells_on_proc, dtype=np.int32)
            vals = func.eval(pts_arr, cells_arr)
            u_values[eval_map] = vals.flatten()
        
        return u_values.reshape((nx, ny))
    
    u_grid = eval_on_grid(u_sol, points, colliding_cells, nx_out, ny_out)
    u_initial_grid = eval_on_grid(u_initial_func, points, colliding_cells, nx_out, ny_out)
    
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
            "pc_type": "ilu",
            "rtol": 1e-10,
            "iterations": total_iterations,
            "dt": dt_val,
            "n_steps": n_steps,
            "time_scheme": "backward_euler",
        }
    }