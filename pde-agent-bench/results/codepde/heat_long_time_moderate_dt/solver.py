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
    
    kappa = float(coeffs.get("kappa", 0.5))
    t_end = float(time_params.get("t_end", 0.2))
    dt_suggested = float(time_params.get("dt", 0.01))
    scheme = time_params.get("scheme", "backward_euler")
    
    # Use suggested dt
    dt = dt_suggested
    n_steps = int(round(t_end / dt))
    dt = t_end / n_steps  # adjust to hit t_end exactly
    
    # Mesh resolution
    nx = ny = 64
    
    # 2. Create mesh
    domain = mesh.create_unit_square(MPI.COMM_WORLD, nx, ny, cell_type=mesh.CellType.triangle)
    
    # 3. Function space
    degree = 1
    V = fem.functionspace(domain, ("Lagrange", degree))
    
    # Spatial coordinate and time
    x = ufl.SpatialCoordinate(domain)
    
    # Manufactured solution: u = exp(-2*t)*sin(pi*x)*sin(pi*y)
    # du/dt = -2*exp(-2*t)*sin(pi*x)*sin(pi*y)
    # nabla^2 u = -2*pi^2*exp(-2*t)*sin(pi*x)*sin(pi*y)
    # f = du/dt - kappa * nabla^2 u
    #   = -2*exp(-2*t)*sin(pi*x)*sin(pi*y) + kappa*2*pi^2*exp(-2*t)*sin(pi*x)*sin(pi*y)
    #   = exp(-2*t)*sin(pi*x)*sin(pi*y)*(-2 + 2*kappa*pi^2)
    
    pi = np.pi
    
    # Time as a constant that we update
    t_const = fem.Constant(domain, default_scalar_type(0.0))
    
    # For source term, we need f at time t^{n+1} for backward Euler
    # f(x, t) = exp(-2*t) * sin(pi*x) * sin(pi*y) * (-2 + 2*kappa*pi^2)
    coeff_f = -2.0 + 2.0 * kappa * pi**2
    
    # We'll use fem.Expression for interpolation of time-dependent functions
    # But for the variational form, we can use UFL expressions with t_const
    
    # Define functions
    u_n = fem.Function(V)  # solution at previous time step
    
    # Initial condition: u(x, 0) = sin(pi*x)*sin(pi*y)
    u_n.interpolate(lambda X: np.sin(np.pi * X[0]) * np.sin(np.pi * X[1]))
    
    # Store initial condition for output
    # Evaluate on grid
    nx_out, ny_out = 50, 50
    xs = np.linspace(0, 1, nx_out)
    ys = np.linspace(0, 1, ny_out)
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
    
    def eval_function(func):
        values = np.full(points_3d.shape[1], np.nan)
        if len(points_on_proc) > 0:
            vals = func.eval(points_on_proc, cells_on_proc)
            values[eval_map] = vals.flatten()
        return values.reshape(nx_out, ny_out)
    
    u_initial = eval_function(u_n)
    
    # 4. Variational problem for backward Euler
    # (u^{n+1} - u^n)/dt - kappa * nabla^2 u^{n+1} = f^{n+1}
    # Weak form:
    # (u^{n+1}, v)/dt + kappa*(grad(u^{n+1}), grad(v)) = (u^n, v)/dt + (f^{n+1}, v)
    
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    
    dt_c = fem.Constant(domain, default_scalar_type(dt))
    kappa_c = fem.Constant(domain, default_scalar_type(kappa))
    
    # Source term as a fem.Function that we update each time step
    f_func = fem.Function(V)
    
    a = (u * v / dt_c + kappa_c * ufl.inner(ufl.grad(u), ufl.grad(v))) * ufl.dx
    L = (u_n * v / dt_c + f_func * v) * ufl.dx
    
    # Boundary conditions: u = g on boundary
    # g(x, t) = exp(-2*t)*sin(pi*x)*sin(pi*y) = 0 on boundary of unit square
    # since sin(pi*0)=sin(pi*1)=0
    # So homogeneous Dirichlet BCs
    
    tdim = domain.topology.dim
    fdim = tdim - 1
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda X: np.ones(X.shape[1], dtype=bool)
    )
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    
    # For safety, use a function for BC that we can update (though it's zero here)
    u_bc = fem.Function(V)
    u_bc.x.array[:] = 0.0
    bc = fem.dirichletbc(u_bc, boundary_dofs)
    
    # 5. Assemble and solve with manual assembly for efficiency in time loop
    a_form = fem.form(a)
    L_form = fem.form(L)
    
    A = petsc.assemble_matrix(a_form, bcs=[bc])
    A.assemble()
    
    b = fem.petsc.create_vector(L_form)
    
    solver = PETSc.KSP().create(domain.comm)
    solver.setOperators(A)
    solver.setType(PETSc.KSP.Type.CG)
    pc = solver.getPC()
    pc.setType(PETSc.PC.Type.ILU)
    solver.setTolerances(rtol=1e-10, atol=1e-12, max_it=1000)
    
    u_sol = fem.Function(V)
    
    total_iterations = 0
    t = 0.0
    
    for step in range(n_steps):
        t += dt
        
        # Update source term f at time t
        # f(x, t) = exp(-2*t) * sin(pi*x) * sin(pi*y) * (-2 + 2*kappa*pi^2)
        exp_val = np.exp(-2.0 * t)
        f_func.interpolate(lambda X, ev=exp_val, cf=coeff_f: ev * np.sin(np.pi * X[0]) * np.sin(np.pi * X[1]) * cf)
        
        # Update BC (still zero for this problem, but in general would update)
        # u_bc stays zero
        
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
        
        # Update u_n for next step
        u_n.x.array[:] = u_sol.x.array[:]
    
    # 7. Extract solution on grid
    u_grid = eval_function(u_sol)
    
    # Cleanup
    solver.destroy()
    A.destroy()
    b.destroy()
    
    return {
        "u": u_grid,
        "u_initial": u_initial,
        "solver_info": {
            "mesh_resolution": nx,
            "element_degree": degree,
            "ksp_type": "cg",
            "pc_type": "ilu",
            "rtol": 1e-10,
            "iterations": total_iterations,
            "dt": dt,
            "n_steps": n_steps,
            "time_scheme": "backward_euler",
        }
    }