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
    time_params = pde.get("time", {})
    t_end = time_params.get("t_end", 0.4)
    dt_suggested = time_params.get("dt", 0.02)
    scheme = time_params.get("scheme", "backward_euler")
    
    # Parameters
    epsilon = pde.get("epsilon", 1.0)
    # For this manufactured solution, we need to figure out epsilon and reaction
    # The PDE is: du/dt - eps*laplacian(u) + R(u) = f
    # With linear reaction R(u) = r*u (linear case)
    # Manufactured solution: u = exp(-t)*sin(pi*x)*sin(pi*y)
    
    # From the case description: "reaction_diffusion_transient_linear_time_decay"
    # Linear reaction: R(u) = r*u for some r
    # Let's check what epsilon and r are. Default epsilon=1 if not specified.
    
    reaction = pde.get("reaction", {})
    # Try to get reaction coefficient
    r_coeff = reaction.get("coefficient", 1.0) if isinstance(reaction, dict) else 1.0
    
    # If epsilon not in pde, check other locations
    if "epsilon" not in pde:
        epsilon = pde.get("coefficients", {}).get("epsilon", 1.0)
    else:
        epsilon = pde["epsilon"]
    
    # Mesh resolution - use fine enough mesh for accuracy
    N = 64
    dt = dt_suggested
    degree = 2
    
    domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", degree))
    
    # Spatial coordinates and time
    x = ufl.SpatialCoordinate(domain)
    
    # Time as a constant that we update
    t_const = fem.Constant(domain, PETSc.ScalarType(0.0))
    
    # Manufactured solution: u_exact = exp(-t)*sin(pi*x)*sin(pi*y)
    pi = np.pi
    u_exact_ufl = ufl.exp(-t_const) * ufl.sin(ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])
    
    # Compute source term f from: du/dt - eps*laplacian(u) + R(u) = f
    # du/dt = -exp(-t)*sin(pi*x)*sin(pi*y)
    # -eps*laplacian(u) = eps*2*pi^2*exp(-t)*sin(pi*x)*sin(pi*y)
    # R(u) = r*u = r*exp(-t)*sin(pi*x)*sin(pi*y)
    # f = (-1 + eps*2*pi^2 + r)*exp(-t)*sin(pi*x)*sin(pi*y)
    
    # But we should derive it symbolically to be safe
    # du/dt with respect to t_const - we'll compute f analytically
    # f = du/dt - eps*lap(u) + r*u
    # = -exp(-t)*sin(pi*x)*sin(pi*y) + eps*2*pi^2*exp(-t)*sin(pi*x)*sin(pi*y) + r*exp(-t)*sin(pi*x)*sin(pi*y)
    # = exp(-t)*sin(pi*x)*sin(pi*y)*(-1 + 2*eps*pi^2 + r)
    
    f_ufl = ufl.exp(-t_const) * ufl.sin(ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1]) * (-1.0 + 2.0 * epsilon * ufl.pi**2 + r_coeff)
    
    # Functions
    u_n = fem.Function(V)  # solution at previous time step
    u_h = fem.Function(V)  # solution at current time step
    v = ufl.TestFunction(V)
    u = ufl.TrialFunction(V)
    
    # Initial condition: u(x,0) = sin(pi*x)*sin(pi*y)
    u_n.interpolate(lambda x: np.sin(np.pi * x[0]) * np.sin(np.pi * x[1]))
    
    # Store initial condition for output
    u_initial_func = fem.Function(V)
    u_initial_func.interpolate(lambda x: np.sin(np.pi * x[0]) * np.sin(np.pi * x[1]))
    
    dt_const = fem.Constant(domain, PETSc.ScalarType(dt))
    eps_const = fem.Constant(domain, PETSc.ScalarType(epsilon))
    r_const = fem.Constant(domain, PETSc.ScalarType(r_coeff))
    
    # Backward Euler: (u - u_n)/dt - eps*lap(u) + r*u = f
    # Weak form: (u - u_n)/dt * v dx + eps*grad(u).grad(v) dx + r*u*v dx = f*v dx
    # Linear form:
    a = (u * v / dt_const) * ufl.dx + eps_const * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx + r_const * u * v * ufl.dx
    L = (u_n * v / dt_const) * ufl.dx + f_ufl * v * ufl.dx
    
    # Boundary conditions: u = g = u_exact on boundary
    # We need to update BC each time step
    tdim = domain.topology.dim
    fdim = tdim - 1
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
    )
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    
    u_bc = fem.Function(V)
    
    # Set initial BC
    t_val = dt  # first time step
    u_bc.interpolate(lambda x: np.exp(-t_val) * np.sin(np.pi * x[0]) * np.sin(np.pi * x[1]))
    bc = fem.dirichletbc(u_bc, boundary_dofs)
    
    # Compile forms
    a_form = fem.form(a)
    L_form = fem.form(L)
    
    # Assemble matrix (constant in time for linear reaction)
    A = petsc.assemble_matrix(a_form, bcs=[bc])
    A.assemble()
    
    b = petsc.create_vector(L_form)
    
    # Setup solver
    solver = PETSc.KSP().create(domain.comm)
    solver.setOperators(A)
    solver.setType(PETSc.KSP.Type.CG)
    pc = solver.getPC()
    pc.setType(PETSc.PC.Type.HYPRE)
    solver.setTolerances(rtol=1e-10, atol=1e-12, max_it=1000)
    solver.setUp()
    
    # Time stepping
    n_steps = int(np.round(t_end / dt))
    total_iterations = 0
    
    for step in range(n_steps):
        t_val = (step + 1) * dt
        t_const.value = t_val
        
        # Update BC
        t_current = t_val
        u_bc.interpolate(lambda x, t=t_current: np.exp(-t) * np.sin(np.pi * x[0]) * np.sin(np.pi * x[1]))
        
        # Re-assemble matrix (BCs might change pattern - but for homogeneous-like BCs on boundary, 
        # the matrix structure doesn't change, only BC rows)
        A.zeroEntries()
        petsc.assemble_matrix(A, a_form, bcs=[bc])
        A.assemble()
        
        # Assemble RHS
        with b.localForm() as loc:
            loc.set(0)
        petsc.assemble_vector(b, L_form)
        petsc.apply_lifting(b, [a_form], bcs=[[bc]])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        petsc.set_bc(b, [bc])
        
        # Solve
        solver.solve(b, u_h.x.petsc_vec)
        u_h.x.scatter_forward()
        
        total_iterations += solver.getIterationNumber()
        
        # Update previous solution
        u_n.x.array[:] = u_h.x.array[:]
    
    # Evaluate on 60x60 grid
    nx_out, ny_out = 60, 60
    xs = np.linspace(0, 1, nx_out)
    ys = np.linspace(0, 1, ny_out)
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
    
    # Also evaluate initial condition on same grid
    u_init_values = np.full(nx_out * ny_out, np.nan)
    if len(points_on_proc) > 0:
        vals_init = u_initial_func.eval(pts_arr, cells_arr)
        u_init_values[eval_map] = vals_init.flatten()
    u_init_grid = u_init_values.reshape((nx_out, ny_out))
    
    solver.destroy()
    A.destroy()
    b.destroy()
    
    return {
        "u": u_grid,
        "u_initial": u_init_grid,
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