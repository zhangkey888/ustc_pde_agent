import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc


def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD
    
    # Parse case spec
    pde = case_spec.get("pde", {})
    domain_spec = case_spec.get("domain", {})
    
    # Parameters
    epsilon = pde.get("epsilon", 1.0)
    reaction_alpha = pde.get("reaction_alpha", 1.0)
    t_end = pde.get("time", {}).get("t_end", 0.5)
    dt_suggested = pde.get("time", {}).get("dt", 0.01)
    time_scheme = pde.get("time", {}).get("scheme", "crank_nicolson")
    
    # Use suggested dt
    dt_val = dt_suggested
    n_steps = int(round(t_end / dt_val))
    dt_val = t_end / n_steps  # adjust to hit t_end exactly
    
    # Mesh resolution
    nx = ny = 80
    element_degree = 2
    
    # Create mesh
    domain = mesh.create_unit_square(comm, nx, ny, cell_type=mesh.CellType.triangle)
    
    # Function space
    V = fem.functionspace(domain, ("Lagrange", element_degree))
    
    # Spatial coordinates
    x = ufl.SpatialCoordinate(domain)
    
    # Source term: f = sin(5*pi*x)*sin(3*pi*y) + 0.5*sin(9*pi*x)*sin(7*pi*y)
    pi = ufl.pi
    f_expr = ufl.sin(5 * pi * x[0]) * ufl.sin(3 * pi * x[1]) + \
             0.5 * ufl.sin(9 * pi * x[0]) * ufl.sin(7 * pi * x[1])
    
    # Trial and test functions
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    
    # Solution at current and previous time step
    u_n = fem.Function(V)  # previous time step
    u_sol = fem.Function(V)  # current solution
    
    # Initial condition: u0 = sin(pi*x)*sin(pi*y)
    u_n.interpolate(lambda X: np.sin(np.pi * X[0]) * np.sin(np.pi * X[1]))
    
    # Store initial condition for output
    u_initial_func = fem.Function(V)
    u_initial_func.interpolate(lambda X: np.sin(np.pi * X[0]) * np.sin(np.pi * X[1]))
    
    # Boundary conditions: u = 0 on all boundaries (homogeneous Dirichlet)
    # Since sin(pi*x)*sin(pi*y) = 0 on boundary, and source is 0 on boundary
    tdim = domain.topology.dim
    fdim = tdim - 1
    
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
    )
    dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc = fem.dirichletbc(PETSc.ScalarType(0.0), dofs, V)
    bcs = [bc]
    
    # Time stepping constants
    dt_const = fem.Constant(domain, PETSc.ScalarType(dt_val))
    eps_const = fem.Constant(domain, PETSc.ScalarType(epsilon))
    alpha_const = fem.Constant(domain, PETSc.ScalarType(reaction_alpha))
    
    # Crank-Nicolson scheme:
    # (u - u_n)/dt - eps * 0.5*(nabla^2 u + nabla^2 u_n) + alpha * 0.5*(u + u_n) = f
    # Bilinear form (LHS):
    # (u/dt)*v + 0.5*eps*grad(u).grad(v) + 0.5*alpha*u*v
    # Linear form (RHS):
    # (u_n/dt)*v - 0.5*eps*grad(u_n).grad(v) - 0.5*alpha*u_n*v + f*v
    
    theta = 0.5  # Crank-Nicolson
    
    a = (u * v / dt_const) * ufl.dx + \
        theta * eps_const * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx + \
        theta * alpha_const * u * v * ufl.dx
    
    L = (u_n * v / dt_const) * ufl.dx - \
        (1.0 - theta) * eps_const * ufl.inner(ufl.grad(u_n), ufl.grad(v)) * ufl.dx - \
        (1.0 - theta) * alpha_const * u_n * v * ufl.dx + \
        f_expr * v * ufl.dx
    
    # Compile forms
    a_form = fem.form(a)
    L_form = fem.form(L)
    
    # Assemble matrix (constant in time for linear reaction)
    A = petsc.assemble_matrix(a_form, bcs=bcs)
    A.assemble()
    
    # Create RHS vector
    b = petsc.create_vector(L_form)
    
    # Setup KSP solver
    solver = PETSc.KSP().create(domain.comm)
    solver.setOperators(A)
    solver.setType(PETSc.KSP.Type.CG)
    pc = solver.getPC()
    pc.setType(PETSc.PC.Type.HYPRE)
    solver.setTolerances(rtol=1e-10, atol=1e-12, max_it=1000)
    solver.setUp()
    
    total_iterations = 0
    
    # Time stepping loop
    for step in range(n_steps):
        # Assemble RHS
        with b.localForm() as loc:
            loc.set(0.0)
        petsc.assemble_vector(b, L_form)
        petsc.apply_lifting(b, [a_form], bcs=[bcs])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        petsc.set_bc(b, bcs)
        
        # Solve
        solver.solve(b, u_sol.x.petsc_vec)
        u_sol.x.scatter_forward()
        
        total_iterations += solver.getIterationNumber()
        
        # Update previous solution
        u_n.x.array[:] = u_sol.x.array[:]
    
    # Evaluate on 80x80 grid
    nx_eval = 80
    ny_eval = 80
    xs = np.linspace(0.0, 1.0, nx_eval)
    ys = np.linspace(0.0, 1.0, ny_eval)
    XX, YY = np.meshgrid(xs, ys, indexing='ij')
    
    points = np.zeros((3, nx_eval * ny_eval))
    points[0, :] = XX.ravel()
    points[1, :] = YY.ravel()
    
    # Point evaluation
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
    
    u_values = np.full(nx_eval * ny_eval, np.nan)
    if len(points_on_proc) > 0:
        pts_arr = np.array(points_on_proc)
        cells_arr = np.array(cells_on_proc, dtype=np.int32)
        vals = u_sol.eval(pts_arr, cells_arr)
        u_values[eval_map] = vals.flatten()
    
    u_grid = u_values.reshape((nx_eval, ny_eval))
    
    # Also evaluate initial condition
    u_init_values = np.full(nx_eval * ny_eval, np.nan)
    if len(points_on_proc) > 0:
        vals_init = u_initial_func.eval(pts_arr, cells_arr)
        u_init_values[eval_map] = vals_init.flatten()
    
    u_init_grid = u_init_values.reshape((nx_eval, ny_eval))
    
    # Cleanup
    solver.destroy()
    A.destroy()
    b.destroy()
    
    return {
        "u": u_grid,
        "u_initial": u_init_grid,
        "solver_info": {
            "mesh_resolution": nx,
            "element_degree": element_degree,
            "ksp_type": "cg",
            "pc_type": "hypre",
            "rtol": 1e-10,
            "iterations": total_iterations,
            "dt": dt_val,
            "n_steps": n_steps,
            "time_scheme": "crank_nicolson",
        }
    }