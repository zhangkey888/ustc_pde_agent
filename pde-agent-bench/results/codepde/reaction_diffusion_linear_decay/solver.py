import numpy as np
from dolfinx import mesh, fem, default_scalar_type, geometry
from dolfinx.fem import petsc
from mpi4py import MPI
import ufl
from petsc4py import PETSc


def solve(case_spec: dict) -> dict:
    # 1. Parse case_spec
    pde_config = case_spec.get("pde", {})
    time_params = pde_config.get("time", {})
    
    t_end = time_params.get("t_end", 0.6)
    dt_suggested = time_params.get("dt", 0.01)
    scheme = time_params.get("scheme", "backward_euler")
    
    # Extract diffusion and reaction parameters
    params = pde_config.get("params", {})
    epsilon = params.get("epsilon", 1.0)
    reaction_alpha = params.get("reaction_alpha", 1.0)
    
    # Agent-selectable parameters for accuracy
    nx = 80
    ny = 80
    degree = 2
    dt = 0.005  # smaller than suggested for better accuracy
    
    n_steps = int(round(t_end / dt))
    dt = t_end / n_steps  # adjust to hit t_end exactly
    
    # 2. Create mesh
    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_square(comm, nx, ny, cell_type=mesh.CellType.triangle)
    
    # 3. Function space
    V = fem.functionspace(domain, ("Lagrange", degree))
    
    # Spatial coordinates
    x = ufl.SpatialCoordinate(domain)
    pi = np.pi
    
    # Time parameter as a Constant
    t_const = fem.Constant(domain, default_scalar_type(0.0))
    
    # Manufactured solution: u = exp(-t)*(cos(2*pi*x)*sin(pi*y))
    u_exact_ufl = ufl.exp(-t_const) * ufl.cos(2 * pi * x[0]) * ufl.sin(pi * x[1])
    
    # Compute source term f from: du/dt - eps * laplacian(u) + alpha * u = f
    # u = exp(-t)*cos(2*pi*x)*sin(pi*y)
    # du/dt = -exp(-t)*cos(2*pi*x)*sin(pi*y) = -u
    # laplacian(u) = exp(-t)*[-(2*pi)^2*cos(2*pi*x)*sin(pi*y) - pi^2*cos(2*pi*x)*sin(pi*y)]
    #              = exp(-t)*cos(2*pi*x)*sin(pi*y)*[-(4*pi^2 + pi^2)]
    #              = -5*pi^2 * u
    # So: -eps * laplacian(u) = eps * 5*pi^2 * u
    # R(u) = alpha * u (linear decay)
    # f = du/dt - eps*laplacian(u) + alpha*u = -u + 5*eps*pi^2*u + alpha*u
    #   = u*(-1 + 5*eps*pi^2 + alpha)
    
    f_expr = u_exact_ufl * (-1.0 + 5.0 * epsilon * pi**2 + reaction_alpha)
    
    # 4. Define functions
    u_n = fem.Function(V)  # solution at previous time step
    u_h = fem.Function(V)  # solution at current time step
    
    # Trial and test functions
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    
    dt_const = fem.Constant(domain, default_scalar_type(dt))
    eps_const = fem.Constant(domain, default_scalar_type(epsilon))
    alpha_const = fem.Constant(domain, default_scalar_type(reaction_alpha))
    
    # Backward Euler: (u - u_n)/dt - eps*laplacian(u) + alpha*u = f
    # Weak form: (u - u_n)/dt * v dx + eps * grad(u) . grad(v) dx + alpha * u * v dx = f * v dx
    a = (u * v / dt_const * ufl.dx
         + eps_const * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
         + alpha_const * u * v * ufl.dx)
    
    L = (u_n * v / dt_const * ufl.dx
         + f_expr * v * ufl.dx)
    
    # 5. Boundary conditions
    # g = u_exact on boundary, time-dependent
    tdim = domain.topology.dim
    fdim = tdim - 1
    
    # All boundary facets
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
    )
    
    bc_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    
    u_bc = fem.Function(V)
    
    # Set initial condition: u(x, 0) = cos(2*pi*x)*sin(pi*y)
    t_const.value = 0.0
    
    u_n.interpolate(lambda x: np.cos(2 * np.pi * x[0]) * np.sin(np.pi * x[1]))
    
    # Initial condition for output
    # Build evaluation grid first
    nx_eval, ny_eval = 65, 65
    xs = np.linspace(0, 1, nx_eval)
    ys = np.linspace(0, 1, ny_eval)
    XX, YY = np.meshgrid(xs, ys, indexing='ij')
    points_2d = np.vstack([XX.ravel(), YY.ravel()])
    points_3d = np.vstack([points_2d, np.zeros(points_2d.shape[1])])
    
    # Build point evaluation infrastructure
    bb_tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, points_3d.T)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points_3d.T)
    
    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    for i in range(points_3d.shape[1]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(points_3d.T[i])
            cells_on_proc.append(links[0])
            eval_map.append(i)
    
    points_on_proc = np.array(points_on_proc) if len(points_on_proc) > 0 else np.empty((0, 3))
    cells_on_proc = np.array(cells_on_proc, dtype=np.int32) if len(cells_on_proc) > 0 else np.empty(0, dtype=np.int32)
    
    # Evaluate initial condition
    u_initial_vals = np.full(points_3d.shape[1], np.nan)
    if len(points_on_proc) > 0:
        vals = u_n.eval(points_on_proc, cells_on_proc)
        u_initial_vals[eval_map] = vals.flatten()
    u_initial_grid = u_initial_vals.reshape((nx_eval, ny_eval))
    
    # 6. Assemble and solve with manual assembly for efficiency in time loop
    a_form = fem.form(a)
    L_form = fem.form(L)
    
    # Assemble LHS matrix (constant in time for backward Euler with linear reaction)
    # But BC function changes, so we need to reassemble or use lifting properly
    # Actually, with time-dependent BCs, we can still keep A constant and just update RHS + lifting
    
    A = petsc.assemble_matrix(a_form, bcs=[fem.dirichletbc(u_bc, bc_dofs)])
    A.assemble()
    
    b = fem.petsc.create_vector(L_form)
    
    # Set up KSP solver
    solver = PETSc.KSP().create(domain.comm)
    solver.setOperators(A)
    solver.setType(PETSc.KSP.Type.GMRES)
    pc = solver.getPC()
    pc.setType(PETSc.PC.Type.ILU)
    solver.setTolerances(rtol=1e-10, atol=1e-12, max_it=2000)
    
    total_iterations = 0
    
    # 7. Time stepping
    t = 0.0
    for step in range(n_steps):
        t += dt
        t_const.value = t
        
        # Update boundary condition
        t_val = t
        u_bc.interpolate(lambda x, tv=t_val: np.exp(-tv) * np.cos(2 * np.pi * x[0]) * np.sin(np.pi * x[1]))
        
        bc = fem.dirichletbc(u_bc, bc_dofs)
        
        # We need to reassemble A because the BC values changed
        # Actually for lifting approach, A stays the same if we use homogeneous version
        # Let's just reassemble both each step for correctness
        A.zeroEntries()
        petsc.assemble_matrix(A, a_form, bcs=[bc])
        A.assemble()
        
        with b.localForm() as loc:
            loc.set(0)
        petsc.assemble_vector(b, L_form)
        petsc.apply_lifting(b, [a_form], bcs=[[bc]])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        petsc.set_bc(b, [bc])
        
        solver.solve(b, u_h.x.petsc_vec)
        u_h.x.scatter_forward()
        
        total_iterations += solver.getIterationNumber()
        
        # Update for next step
        u_n.x.array[:] = u_h.x.array[:]
    
    # 8. Extract solution on evaluation grid
    u_values = np.full(points_3d.shape[1], np.nan)
    if len(points_on_proc) > 0:
        vals = u_h.eval(points_on_proc, cells_on_proc)
        u_values[eval_map] = vals.flatten()
    u_grid = u_values.reshape((nx_eval, ny_eval))
    
    solver_info = {
        "mesh_resolution": nx,
        "element_degree": degree,
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
        "u_initial": u_initial_grid,
        "solver_info": solver_info,
    }