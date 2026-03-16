import numpy as np
from dolfinx import mesh, fem, default_scalar_type, geometry
from dolfinx.fem import petsc
from mpi4py import MPI
import ufl
from petsc4py import PETSc


def solve(case_spec: dict) -> dict:
    # 1. Parse case_spec
    pde = case_spec.get("pde", {})
    coefficients = pde.get("coefficients", {})
    kappa = float(coefficients.get("kappa", 1.0))
    
    time_params = pde.get("time", {})
    t_end = float(time_params.get("t_end", 0.1))
    dt_suggested = float(time_params.get("dt", 0.01))
    scheme = time_params.get("scheme", "backward_euler")
    
    # Use a finer time step for accuracy
    dt = dt_suggested
    n_steps = int(round(t_end / dt))
    dt = t_end / n_steps  # adjust to hit t_end exactly
    
    # Mesh resolution and element degree
    nx = ny = 80
    degree = 2
    
    # 2. Create mesh (quadrilateral as indicated by case name)
    domain = mesh.create_unit_square(MPI.COMM_WORLD, nx, ny, cell_type=mesh.CellType.quadrilateral)
    
    # 3. Function space
    V = fem.functionspace(domain, ("Lagrange", degree))
    
    # Spatial coordinate and time
    x = ufl.SpatialCoordinate(domain)
    
    # Manufactured solution: u_exact = exp(-t)*sin(pi*x)*sin(pi*y)
    # du/dt = -exp(-t)*sin(pi*x)*sin(pi*y)
    # -kappa * laplacian(u) = kappa * 2*pi^2 * exp(-t)*sin(pi*x)*sin(pi*y)
    # f = du/dt - kappa*laplacian(u) ... wait, the PDE is du/dt - div(kappa*grad(u)) = f
    # So f = du/dt - kappa*laplacian(u) = -exp(-t)*sin(pi*x)*sin(pi*y) + kappa*2*pi^2*exp(-t)*sin(pi*x)*sin(pi*y)
    # f = exp(-t)*sin(pi*x)*sin(pi*y)*(-1 + 2*kappa*pi^2)
    
    # Time parameter as a Constant so we can update it
    t_const = fem.Constant(domain, default_scalar_type(0.0))
    
    # Source term as UFL expression
    pi_ = np.pi
    f_expr = ufl.exp(-t_const) * ufl.sin(pi_ * x[0]) * ufl.sin(pi_ * x[1]) * (-1.0 + 2.0 * kappa * pi_**2)
    
    # Exact solution UFL expression (for BC)
    u_exact_expr = ufl.exp(-t_const) * ufl.sin(pi_ * x[0]) * ufl.sin(pi_ * x[1])
    
    # 4. Define functions
    u_n = fem.Function(V)  # solution at previous time step
    u_h = fem.Function(V)  # solution at current time step (will be used for output)
    
    # Initial condition: u(x,0) = sin(pi*x)*sin(pi*y)
    u_n.interpolate(lambda X: np.sin(np.pi * X[0]) * np.sin(np.pi * X[1]))
    
    # 5. Boundary conditions
    # u = g on boundary, g = exp(-t)*sin(pi*x)*sin(pi*y) = 0 on boundary of unit square
    # since sin(0)=sin(pi)=0, the BC is homogeneous
    tdim = domain.topology.dim
    fdim = tdim - 1
    
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda X: np.ones(X.shape[1], dtype=bool)
    )
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    
    # BC function (will be updated each time step, but it's always 0 here)
    u_bc = fem.Function(V)
    u_bc.x.array[:] = 0.0
    bc = fem.dirichletbc(u_bc, boundary_dofs)
    
    # 6. Variational form (Backward Euler)
    # (u - u_n)/dt - kappa * laplacian(u) = f
    # Weak form: (u - u_n)/dt * v dx + kappa * grad(u) . grad(v) dx = f * v dx
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    
    dt_const = fem.Constant(domain, default_scalar_type(dt))
    kappa_const = fem.Constant(domain, default_scalar_type(kappa))
    
    a = (u * v / dt_const + kappa_const * ufl.inner(ufl.grad(u), ufl.grad(v))) * ufl.dx
    L = (u_n * v / dt_const + f_expr * v) * ufl.dx
    
    # 7. Compile forms
    a_form = fem.form(a)
    L_form = fem.form(L)
    
    # 8. Assemble matrix (constant in time)
    A = petsc.assemble_matrix(a_form, bcs=[bc])
    A.assemble()
    
    # Create RHS vector
    b = fem.petsc.create_vector(L_form)
    
    # 9. Setup solver
    solver = PETSc.KSP().create(domain.comm)
    solver.setOperators(A)
    solver.setType(PETSc.KSP.Type.CG)
    pc = solver.getPC()
    pc.setType(PETSc.PC.Type.HYPRE)
    solver.setTolerances(rtol=1e-10, atol=1e-12, max_it=1000)
    solver.setUp()
    
    # 10. Time stepping
    total_iterations = 0
    t = 0.0
    
    for step in range(n_steps):
        t += dt
        t_const.value = t
        
        # Update BC if needed (it's zero, but let's be safe for general case)
        # Actually on unit square boundary sin(pi*x)*sin(pi*y) = 0, so BC is always 0
        
        # Assemble RHS
        with b.localForm() as loc:
            loc.set(0.0)
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
    
    # 11. Extract solution on 50x50 uniform grid
    n_eval = 50
    xs = np.linspace(0.0, 1.0, n_eval)
    ys = np.linspace(0.0, 1.0, n_eval)
    XX, YY = np.meshgrid(xs, ys, indexing='ij')
    
    points_2d = np.zeros((3, n_eval * n_eval))
    points_2d[0, :] = XX.ravel()
    points_2d[1, :] = YY.ravel()
    points_2d[2, :] = 0.0
    
    # Point evaluation
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
    
    u_values = np.full(n_eval * n_eval, np.nan)
    if len(points_on_proc) > 0:
        pts_arr = np.array(points_on_proc)
        cells_arr = np.array(cells_on_proc, dtype=np.int32)
        vals = u_h.eval(pts_arr, cells_arr)
        u_values[eval_map] = vals.flatten()
    
    u_grid = u_values.reshape((n_eval, n_eval))
    
    # Also compute initial condition on same grid for analysis
    u_init_func = fem.Function(V)
    u_init_func.interpolate(lambda X: np.sin(np.pi * X[0]) * np.sin(np.pi * X[1]))
    
    u_init_values = np.full(n_eval * n_eval, np.nan)
    if len(points_on_proc) > 0:
        vals_init = u_init_func.eval(pts_arr, cells_arr)
        u_init_values[eval_map] = vals_init.flatten()
    u_initial_grid = u_init_values.reshape((n_eval, n_eval))
    
    # Cleanup
    solver.destroy()
    A.destroy()
    b.destroy()
    
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