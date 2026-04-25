import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc

def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD
    
    pde = case_spec["pde"]
    coeffs = pde["coefficients"]
    time_params = pde["time"]
    grid_spec = case_spec["output"]["grid"]
    
    kappa = float(coeffs["kappa"])
    t0 = float(time_params["t0"])
    t_end = float(time_params["t_end"])
    dt_suggested = float(time_params["dt"])
    
    nx_out = grid_spec["nx"]
    ny_out = grid_spec["ny"]
    bbox = grid_spec["bbox"]
    
    # Solver parameters - optimized for accuracy within time budget
    mesh_res = 64
    element_degree = 2
    dt = dt_suggested
    rtol = 1e-10
    
    # Create mesh and function space
    domain = mesh.create_unit_square(comm, mesh_res, mesh_res, cell_type=mesh.CellType.triangle)
    tdim = domain.topology.dim
    fdim = tdim - 1
    V = fem.functionspace(domain, ("Lagrange", element_degree))
    
    # Define functions
    f_func = fem.Function(V)  # Source term (time-dependent)
    g_func = fem.Function(V)  # Boundary condition (time-dependent)
    u_n = fem.Function(V)     # Solution at previous time step
    
    # Initial condition: u(x,0) = sin(2*pi*x)*sin(pi*y)
    u_n.interpolate(lambda x: np.sin(2*np.pi*x[0]) * np.sin(np.pi*x[1]))
    
    # Variational formulation: Backward Euler
    # (u^{n+1} - u^n)/dt = kappa*div(grad(u^{n+1})) + f^{n+1}
    # Weak form: ∫ u*v dx + dt*kappa*∫ grad(u)·grad(v) dx = ∫ u_n*v dx + dt*∫ f*v dx
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    
    a = ufl.inner(u, v) * ufl.dx + dt * kappa * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = ufl.inner(u_n, v) * ufl.dx + dt * ufl.inner(f_func, v) * ufl.dx
    
    # Boundary setup
    bf = mesh.locate_entities_boundary(domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool))
    bd = fem.locate_dofs_topological(V, fdim, bf)
    
    # Time stepping
    n_steps = int(round((t_end - t0) / dt))
    t = t0
    u_sol = None
    
    for step in range(n_steps):
        t_new = t + dt
        
        # Update source term f at t_new
        # f = (5*kappa*pi^2 - 1)*exp(-t)*sin(2*pi*x)*sin(pi*y)
        f_func.interpolate(lambda x, _tn=t_new: (5*kappa*np.pi**2 - 1)*np.exp(-_tn)*np.sin(2*np.pi*x[0])*np.sin(np.pi*x[1]))
        
        # Update boundary condition g at t_new
        # g = exp(-t)*sin(2*pi*x)*sin(pi*y)
        g_func.interpolate(lambda x, _tn=t_new: np.exp(-_tn)*np.sin(2*np.pi*x[0])*np.sin(np.pi*x[1]))
        
        bc = fem.dirichletbc(g_func, bd)
        
        # Solve using LinearProblem (handles assembly and BCs correctly)
        problem = petsc.LinearProblem(a, L, bcs=[bc],
                                       petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
                                       petsc_options_prefix="heat_")
        u_sol = problem.solve()
        
        # Update for next step
        u_n.x.array[:] = u_sol.x.array[:]
        t = t_new
    
    # Sample solution on output grid
    xmin, xmax, ymin, ymax = bbox
    xs = np.linspace(xmin, xmax, nx_out)
    ys = np.linspace(ymin, ymax, ny_out)
    XX, YY = np.meshgrid(xs, ys)
    pts = np.column_stack([XX.ravel(), YY.ravel(), np.zeros(nx_out * ny_out)])
    
    bb_tree = geometry.bb_tree(domain, tdim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, pts)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, pts)
    
    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    for i in range(pts.shape[0]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(pts[i])
            cells_on_proc.append(links[0])
            eval_map.append(i)
    
    u_values = np.full((nx_out * ny_out,), np.nan)
    if len(points_on_proc) > 0:
        vals = u_sol.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u_values[eval_map] = vals.flatten()
    
    u_grid = u_values.reshape(ny_out, nx_out)
    
    # Sample initial condition
    u_init_func = fem.Function(V)
    u_init_func.interpolate(lambda x: np.sin(2*np.pi*x[0]) * np.sin(np.pi*x[1]))
    
    u_initial_vals = np.full((nx_out * ny_out,), np.nan)
    if len(points_on_proc) > 0:
        vals_init = u_init_func.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u_initial_vals[eval_map] = vals_init.flatten()
    u_initial_grid = u_initial_vals.reshape(ny_out, nx_out)
    
    # L2 error verification
    u_exact = fem.Function(V)
    u_exact.interpolate(lambda x: np.exp(-t_end)*np.sin(2*np.pi*x[0])*np.sin(np.pi*x[1]))
    
    error_sq = fem.form(ufl.inner(u_sol-u_exact, u_sol-u_exact)*ufl.dx)
    error_local = fem.assemble_scalar(error_sq)
    l2_error = np.sqrt(domain.comm.allreduce(float(error_local), op=MPI.SUM))
    
    if comm.rank == 0:
        print(f"L2 error: {l2_error:.6e}")
    
    return {
        "u": u_grid,
        "u_initial": u_initial_grid,
        "solver_info": {
            "mesh_resolution": mesh_res,
            "element_degree": element_degree,
            "ksp_type": "preonly",
            "pc_type": "lu",
            "rtol": rtol,
            "iterations": n_steps,
            "dt": dt,
            "n_steps": n_steps,
            "time_scheme": "backward_euler"
        }
    }
