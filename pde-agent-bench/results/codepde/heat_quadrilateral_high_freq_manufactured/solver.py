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
    t_end = float(time_params.get("t_end", 0.1))
    dt_suggested = float(time_params.get("dt", 0.005))
    scheme = time_params.get("scheme", "backward_euler")
    
    # Choose mesh resolution and dt for accuracy
    # High frequency (4*pi) needs fine mesh. Let's use a good resolution.
    N = 80  # mesh resolution
    degree = 2  # quadratic elements for better accuracy
    dt = 0.0025  # smaller dt for accuracy with high frequency
    
    n_steps = int(round(t_end / dt))
    dt = t_end / n_steps  # adjust to hit t_end exactly
    
    # 2. Create mesh (quadrilateral as suggested by case name)
    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.quadrilateral)
    
    # 3. Function space
    V = fem.functionspace(domain, ("Lagrange", degree))
    
    # Spatial coordinate and time
    x = ufl.SpatialCoordinate(domain)
    
    # Manufactured solution: u = exp(-t)*sin(4*pi*x)*sin(4*pi*y)
    # du/dt = -exp(-t)*sin(4*pi*x)*sin(4*pi*y)
    # -kappa * laplacian(u) = kappa * 32*pi^2 * exp(-t)*sin(4*pi*x)*sin(4*pi*y)
    # f = du/dt - kappa*laplacian(u) ... wait, the PDE is du/dt - div(kappa*grad(u)) = f
    # So f = du/dt - kappa*laplacian(u)
    # f = -exp(-t)*sin(4*pi*x)*sin(4*pi*y) + kappa*32*pi^2*exp(-t)*sin(4*pi*x)*sin(4*pi*y)
    # f = exp(-t)*sin(4*pi*x)*sin(4*pi*y)*(-1 + 32*kappa*pi^2)
    
    # Time parameter as a Constant so we can update it
    t_const = fem.Constant(domain, PETSc.ScalarType(0.0))
    
    # We'll handle the source term and BC using interpolation at each time step
    
    # 4. Define functions
    u_n = fem.Function(V)  # solution at previous time step
    u_h = fem.Function(V)  # solution at current time step (will be the unknown)
    
    # Initial condition: u(x,0) = sin(4*pi*x)*sin(4*pi*y)
    u_n.interpolate(lambda x: np.sin(4 * np.pi * x[0]) * np.sin(4 * np.pi * x[1]))
    
    # Store initial condition for output
    # Evaluate on grid
    nx_out, ny_out = 50, 50
    xs = np.linspace(0, 1, nx_out)
    ys = np.linspace(0, 1, ny_out)
    XX, YY = np.meshgrid(xs, ys, indexing='ij')
    points_2d = np.vstack([XX.ravel(), YY.ravel()])
    points_3d = np.vstack([points_2d, np.zeros(points_2d.shape[1])])
    
    # Point evaluation setup
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
    
    def evaluate_function(func):
        values = np.full(points_3d.shape[1], np.nan)
        if len(points_on_proc) > 0:
            vals = func.eval(points_on_proc, cells_on_proc)
            values[eval_map] = vals.flatten()
        return values.reshape(nx_out, ny_out)
    
    u_initial = evaluate_function(u_n)
    
    # 5. Variational form for backward Euler
    # (u - u_n)/dt - kappa*laplacian(u) = f
    # Weak form: (u - u_n)/dt * v dx + kappa * grad(u) . grad(v) dx = f * v dx
    
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    
    f_func = fem.Function(V)  # source term, updated each time step
    
    # Bilinear form (constant in time for backward Euler with constant kappa)
    a = (u * v / fem.Constant(domain, PETSc.ScalarType(dt))) * ufl.dx + \
        fem.Constant(domain, PETSc.ScalarType(kappa)) * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
    
    L = (u_n * v / fem.Constant(domain, PETSc.ScalarType(dt))) * ufl.dx + \
        f_func * v * ufl.dx
    
    # 6. Boundary conditions
    # At each time step, we need g = exp(-t)*sin(4*pi*x)*sin(4*pi*y) on boundary
    u_bc = fem.Function(V)
    
    tdim = domain.topology.dim
    fdim = tdim - 1
    
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
    )
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    
    bc = fem.dirichletbc(u_bc, boundary_dofs)
    
    # 7. Assemble and solve with manual assembly for efficiency
    a_form = fem.form(a)
    L_form = fem.form(L)
    
    A = petsc.assemble_matrix(a_form, bcs=[bc])
    A.assemble()
    
    b = fem.Function(V)  # we'll use petsc vec
    b_vec = petsc.create_vector(L_form)
    
    # Setup KSP solver
    solver = PETSc.KSP().create(domain.comm)
    solver.setOperators(A)
    solver.setType(PETSc.KSP.Type.CG)
    pc = solver.getPC()
    pc.setType(PETSc.PC.Type.HYPRE)
    solver.setTolerances(rtol=1e-10, atol=1e-12, max_it=2000)
    solver.setUp()
    
    u_sol = fem.Function(V)
    
    total_iterations = 0
    coeff_val = -1.0 + 32.0 * kappa * np.pi**2
    
    # 8. Time stepping
    t = 0.0
    for step in range(n_steps):
        t += dt
        
        # Update source term: f = exp(-t)*sin(4*pi*x)*sin(4*pi*y)*(-1 + 32*kappa*pi^2)
        exp_t = np.exp(-t)
        f_func.interpolate(
            lambda x, et=exp_t, cv=coeff_val: et * cv * np.sin(4 * np.pi * x[0]) * np.sin(4 * np.pi * x[1])
        )
        
        # Update boundary condition
        u_bc.interpolate(
            lambda x, et=exp_t: et * np.sin(4 * np.pi * x[0]) * np.sin(4 * np.pi * x[1])
        )
        
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
        
        # Update u_n for next step
        u_n.x.array[:] = u_sol.x.array[:]
    
    # 9. Extract solution on grid
    u_grid = evaluate_function(u_sol)
    
    # Cleanup
    solver.destroy()
    A.destroy()
    b_vec.destroy()
    
    return {
        "u": u_grid,
        "u_initial": u_initial,
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