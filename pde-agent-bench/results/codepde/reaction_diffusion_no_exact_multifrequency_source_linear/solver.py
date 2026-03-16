import numpy as np
from dolfinx import mesh, fem, default_scalar_type, geometry
from dolfinx.fem import petsc
from mpi4py import MPI
import ufl
from petsc4py import PETSc


def solve(case_spec: dict) -> dict:
    # 1. Parse case_spec
    pde_config = case_spec.get("pde", case_spec.get("oracle_config", {}).get("pde", {}))
    
    # Extract parameters
    epsilon = float(pde_config.get("epsilon", 0.01))
    reaction_alpha = float(pde_config.get("reaction_alpha", 1.0))
    
    # Time parameters
    time_params = pde_config.get("time", None)
    if time_params is None:
        time_params = {}
    
    t_end = float(time_params.get("t_end", 0.5))
    dt_suggested = float(time_params.get("dt", 0.01))
    scheme = time_params.get("scheme", "crank_nicolson")
    
    # Use a fine mesh for accuracy
    nx, ny = 100, 100
    degree = 2
    
    # Use a reasonable dt
    dt_val = dt_suggested
    n_steps = int(round(t_end / dt_val))
    dt_val = t_end / n_steps  # adjust to hit t_end exactly
    
    # 2. Create mesh
    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_square(comm, nx, ny, cell_type=mesh.CellType.triangle)
    
    # 3. Function space
    V = fem.functionspace(domain, ("Lagrange", degree))
    
    # Spatial coordinate
    x = ufl.SpatialCoordinate(domain)
    pi = np.pi
    
    # Source term: f = sin(5*pi*x)*sin(3*pi*y) + 0.5*sin(9*pi*x)*sin(7*pi*y)
    f_expr = (ufl.sin(5 * pi * x[0]) * ufl.sin(3 * pi * x[1]) 
              + 0.5 * ufl.sin(9 * pi * x[0]) * ufl.sin(7 * pi * x[1]))
    
    # 4. Define functions
    u_n = fem.Function(V, name="u_n")  # solution at previous time step
    u_h = fem.Function(V, name="u_h")  # solution at current time step
    
    # Initial condition: u0 = sin(pi*x)*sin(pi*y)
    u_n.interpolate(lambda X: np.sin(np.pi * X[0]) * np.sin(np.pi * X[1]))
    
    # Store initial condition for output
    # Evaluate on grid
    nx_out, ny_out = 80, 80
    xs = np.linspace(0, 1, nx_out)
    ys = np.linspace(0, 1, ny_out)
    Xg, Yg = np.meshgrid(xs, ys, indexing='ij')
    points_2d = np.zeros((3, nx_out * ny_out))
    points_2d[0] = Xg.ravel()
    points_2d[1] = Yg.ravel()
    
    # Build point evaluation infrastructure
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
    
    points_on_proc_arr = np.array(points_on_proc) if len(points_on_proc) > 0 else np.zeros((0, 3))
    cells_on_proc_arr = np.array(cells_on_proc, dtype=np.int32) if len(cells_on_proc) > 0 else np.zeros(0, dtype=np.int32)
    
    def evaluate_on_grid(func):
        u_values = np.full(nx_out * ny_out, np.nan)
        if len(points_on_proc) > 0:
            vals = func.eval(points_on_proc_arr, cells_on_proc_arr)
            for idx, global_idx in enumerate(eval_map):
                u_values[global_idx] = vals[idx, 0] if vals.ndim > 1 else vals[idx]
        return u_values.reshape((nx_out, ny_out))
    
    u_initial = evaluate_on_grid(u_n)
    
    # 5. Boundary conditions (homogeneous Dirichlet u=0 on boundary)
    tdim = domain.topology.dim
    fdim = tdim - 1
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
    )
    dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc = fem.dirichletbc(PETSc.ScalarType(0.0), dofs, V)
    bcs = [bc]
    
    # 6. Time stepping with Crank-Nicolson
    # Crank-Nicolson: (u - u_n)/dt - epsilon * 0.5*(nabla^2 u + nabla^2 u_n) + alpha * 0.5*(u + u_n) = f
    # Weak form:
    #   (u - u_n)/dt * v dx + epsilon * 0.5*(grad(u) + grad(u_n)) . grad(v) dx 
    #   + alpha * 0.5*(u + u_n) * v dx = f * v dx
    
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    dt_c = fem.Constant(domain, PETSc.ScalarType(dt_val))
    eps_c = fem.Constant(domain, PETSc.ScalarType(epsilon))
    alpha_c = fem.Constant(domain, PETSc.ScalarType(reaction_alpha))
    
    if scheme == "crank_nicolson":
        theta = 0.5
    elif scheme == "backward_euler":
        theta = 1.0
    else:
        theta = 0.5  # default to CN
    
    theta_c = fem.Constant(domain, PETSc.ScalarType(theta))
    one_minus_theta = fem.Constant(domain, PETSc.ScalarType(1.0 - theta))
    
    # Bilinear form (LHS)
    a_form = (
        u * v * ufl.dx
        + dt_c * theta_c * eps_c * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
        + dt_c * theta_c * alpha_c * u * v * ufl.dx
    )
    
    # Linear form (RHS)
    L_form = (
        u_n * v * ufl.dx
        - dt_c * one_minus_theta * eps_c * ufl.inner(ufl.grad(u_n), ufl.grad(v)) * ufl.dx
        - dt_c * one_minus_theta * alpha_c * u_n * v * ufl.dx
        + dt_c * f_expr * v * ufl.dx
    )
    
    # Compile forms
    a_compiled = fem.form(a_form)
    L_compiled = fem.form(L_form)
    
    # Assemble matrix (constant in time for linear reaction)
    A = petsc.assemble_matrix(a_compiled, bcs=bcs)
    A.assemble()
    
    # Create RHS vector
    b = petsc.create_vector(L_compiled)
    
    # Setup KSP solver
    solver = PETSc.KSP().create(domain.comm)
    solver.setOperators(A)
    solver.setType(PETSc.KSP.Type.CG)
    pc = solver.getPC()
    pc.setType(PETSc.PC.Type.HYPRE)
    solver.setTolerances(rtol=1e-10, atol=1e-12, max_it=2000)
    solver.setUp()
    
    total_iterations = 0
    
    # Time stepping loop
    for step in range(n_steps):
        # Assemble RHS
        with b.localForm() as loc:
            loc.set(0.0)
        petsc.assemble_vector(b, L_compiled)
        petsc.apply_lifting(b, [a_compiled], bcs=[bcs])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        petsc.set_bc(b, bcs)
        
        # Solve
        solver.solve(b, u_h.x.petsc_vec)
        u_h.x.scatter_forward()
        
        total_iterations += solver.getIterationNumber()
        
        # Update for next step
        u_n.x.array[:] = u_h.x.array[:]
    
    # 7. Extract solution on grid
    u_grid = evaluate_on_grid(u_h)
    
    # Cleanup
    solver.destroy()
    A.destroy()
    b.destroy()
    
    time_scheme_str = "crank_nicolson" if scheme == "crank_nicolson" else ("backward_euler" if scheme == "backward_euler" else scheme)
    
    return {
        "u": u_grid,
        "u_initial": u_initial,
        "solver_info": {
            "mesh_resolution": nx,
            "element_degree": degree,
            "ksp_type": "cg",
            "pc_type": "hypre",
            "rtol": 1e-10,
            "iterations": total_iterations,
            "dt": dt_val,
            "n_steps": n_steps,
            "time_scheme": time_scheme_str,
        }
    }