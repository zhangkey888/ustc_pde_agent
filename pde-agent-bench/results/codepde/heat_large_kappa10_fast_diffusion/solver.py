import numpy as np
from dolfinx import mesh, fem, default_scalar_type, geometry
from dolfinx.fem import petsc
from mpi4py import MPI
import ufl
from petsc4py import PETSc


def solve(case_spec: dict) -> dict:
    # 1. Parse case_spec
    pde_config = case_spec.get("pde", {})
    coeffs = pde_config.get("coefficients", {})
    kappa = coeffs.get("kappa", 10.0)
    
    time_config = pde_config.get("time", {})
    t_end = time_config.get("t_end", 0.05)
    dt_suggested = time_config.get("dt", 0.005)
    scheme = time_config.get("scheme", "backward_euler")
    
    # Choose parameters for accuracy and speed
    N = 64
    degree = 2
    dt = dt_suggested
    n_steps = int(round(t_end / dt))
    dt = t_end / n_steps  # exact division
    
    # 2. Create mesh
    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
    
    # 3. Function space
    V = fem.functionspace(domain, ("Lagrange", degree))
    
    # 4. Spatial coordinate and exact solution components
    x = ufl.SpatialCoordinate(domain)
    pi = np.pi
    
    # Time as a constant that we update
    t_const = fem.Constant(domain, default_scalar_type(0.0))
    dt_const = fem.Constant(domain, default_scalar_type(dt))
    kappa_const = fem.Constant(domain, default_scalar_type(kappa))
    
    # Exact solution: u_exact = exp(-t)*sin(pi*x)*sin(pi*y)
    # du/dt = -exp(-t)*sin(pi*x)*sin(pi*y)
    # -kappa * laplacian(u) = kappa * 2*pi^2 * exp(-t)*sin(pi*x)*sin(pi*y)
    # f = du/dt - kappa*laplacian(u) ... wait, the PDE is du/dt - div(kappa*grad(u)) = f
    # -div(kappa*grad(u)) = kappa * 2*pi^2 * exp(-t)*sin(pi*x)*sin(pi*y)
    # du/dt = -exp(-t)*sin(pi*x)*sin(pi*y)
    # f = du/dt + kappa*2*pi^2*u = (-1 + 2*kappa*pi^2)*exp(-t)*sin(pi*x)*sin(pi*y)
    
    u_exact_ufl = ufl.exp(-t_const) * ufl.sin(pi * x[0]) * ufl.sin(pi * x[1])
    f_ufl = (-1.0 + 2.0 * kappa * pi**2) * ufl.exp(-t_const) * ufl.sin(pi * x[0]) * ufl.sin(pi * x[1])
    
    # 5. Functions
    u_n = fem.Function(V, name="u_n")  # solution at previous time step
    u_h = fem.Function(V, name="u_h")  # solution at current time step
    
    # Initial condition: u(x,0) = sin(pi*x)*sin(pi*y)
    u_n.interpolate(lambda x: np.sin(pi * x[0]) * np.sin(pi * x[1]))
    
    # Store initial condition for output
    nx_out, ny_out = 50, 50
    xs = np.linspace(0, 1, nx_out)
    ys = np.linspace(0, 1, ny_out)
    XX, YY = np.meshgrid(xs, ys, indexing='ij')
    points_2d = np.vstack([XX.ravel(), YY.ravel(), np.zeros(nx_out * ny_out)])
    
    # Evaluate initial condition
    u_initial = _evaluate_on_grid(domain, u_n, points_2d, nx_out, ny_out)
    
    # 6. Variational form (backward Euler)
    # (u - u_n)/dt - kappa*laplacian(u) = f
    # Weak form: (u - u_n)/dt * v dx + kappa * grad(u) . grad(v) dx = f * v dx
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    
    a = (u * v / dt_const) * ufl.dx + kappa_const * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = (u_n * v / dt_const) * ufl.dx + f_ufl * v * ufl.dx
    
    # 7. Boundary conditions (u = 0 on boundary since sin(pi*x)*sin(pi*y) = 0 on boundary)
    tdim = domain.topology.dim
    fdim = tdim - 1
    
    # For the exact solution, on the boundary sin(pi*x)*sin(pi*y)=0 when x=0,1 or y=0,1
    # So g = exp(-t)*0 = 0 on boundary. But let's use the exact BC to be safe.
    
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
    )
    
    u_bc_func = fem.Function(V)
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    
    # BC is zero on boundary for all time
    u_bc_func.x.array[:] = 0.0
    bc = fem.dirichletbc(u_bc_func, boundary_dofs)
    
    # 8. Assemble and solve with manual assembly for efficiency
    a_form = fem.form(a)
    L_form = fem.form(L)
    
    A = petsc.assemble_matrix(a_form, bcs=[bc])
    A.assemble()
    
    b = fem.petsc.create_vector(L_form)
    
    solver = PETSc.KSP().create(domain.comm)
    solver.setOperators(A)
    solver.setType(PETSc.KSP.Type.CG)
    pc = solver.getPC()
    pc.setType(PETSc.PC.Type.HYPRE)
    solver.setTolerances(rtol=1e-10, atol=1e-12, max_it=1000)
    solver.setUp()
    
    total_iterations = 0
    
    # 9. Time stepping
    t = 0.0
    for step in range(n_steps):
        t += dt
        t_const.value = t
        
        # Update BC if needed (it's zero, so no update needed)
        
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
    
    # 10. Extract solution on grid
    u_grid = _evaluate_on_grid(domain, u_h, points_2d, nx_out, ny_out)
    
    solver.destroy()
    A.destroy()
    b.destroy()
    
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


def _evaluate_on_grid(domain, u_func, points_3d, nx, ny):
    """Evaluate a function on a grid of points and return (nx, ny) array."""
    bb_tree = geometry.bb_tree(domain, domain.topology.dim)
    
    n_points = points_3d.shape[1]
    points_T = points_3d.T  # (N, 3)
    
    cell_candidates = geometry.compute_collisions_points(bb_tree, points_T)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points_T)
    
    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    
    for i in range(n_points):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(points_T[i])
            cells_on_proc.append(links[0])
            eval_map.append(i)
    
    u_values = np.full(n_points, np.nan)
    if len(points_on_proc) > 0:
        pts_arr = np.array(points_on_proc)
        cells_arr = np.array(cells_on_proc, dtype=np.int32)
        vals = u_func.eval(pts_arr, cells_arr)
        u_values[eval_map] = vals.flatten()
    
    return u_values.reshape(nx, ny)