import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry, log
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc
import time

ScalarType = PETSc.ScalarType

def solve(case_spec: dict) -> dict:
    start_wall = time.time()
    
    # Grid spec
    grid_spec = case_spec.get("output", {}).get("grid", {})
    nx_out = grid_spec.get("nx", 50)
    ny_out = grid_spec.get("ny", 50)
    bbox = grid_spec.get("bbox", [0.0, 1.0, 0.0, 1.0])
    
    # PDE spec
    pde_spec = case_spec.get("pde", {})
    eps = pde_spec.get("epsilon", 0.01) # Default if missing
    
    # Time params
    time_spec = pde_spec.get("time", {})
    t0 = time_spec.get("t0", 0.0)
    t_end = time_spec.get("t_end", 0.15)
    dt_val = time_spec.get("dt", 0.005)
    
    # Solver params
    mesh_res = 64
    elem_degree = 2
    
    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_square(comm, mesh_res, mesh_res, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", elem_degree))
    
    # Time variable
    t_ufl = fem.Constant(domain, ScalarType(t0))
    dt = fem.Constant(domain, ScalarType(dt_val))
    
    x = ufl.SpatialCoordinate(domain)
    # Manufactured solution
    u_exact = ufl.exp(-t_ufl) * (0.3 * ufl.sin(ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1]))
    
    def R(u):
        return u**3 - u
    
    # Source term mathematically derived
    # f = du_ex/dt - eps * Laplacian(u_ex) + R(u_ex)
    # We can compute it manually or with ufl
    u_exact_t = -ufl.exp(-t_ufl) * (0.3 * ufl.sin(ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1]))
    laplacian_u_exact = ufl.exp(-t_ufl) * 0.3 * (
        -ufl.pi**2 * ufl.sin(ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])
        -ufl.pi**2 * ufl.sin(ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])
    )
    f_ufl = u_exact_t - eps * laplacian_u_exact + R(u_exact)
    
    # Unknown and test functions
    u = fem.Function(V)
    u_n = fem.Function(V)
    v = ufl.TestFunction(V)
    
    # Initial Condition
    class ExactSol:
        def __init__(self, t_val):
            self.t = t_val
        def __call__(self, x_pts):
            return np.exp(-self.t) * (0.3 * np.sin(np.pi * x_pts[0]) * np.sin(np.pi * x_pts[1]))
    
    u_n.interpolate(ExactSol(t0))
    u.interpolate(ExactSol(t0))
    
    u_initial_arr = sample_on_grid(domain, u_n, nx_out, ny_out, bbox)
    
    # Boundary Conditions
    fdim = domain.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(domain, fdim, lambda x: np.full(x.shape[1], True, dtype=bool))
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    u_bc = fem.Function(V)
    
    # Variational Form (Backward Euler)
    # (u - u_n)/dt - eps * Laplace(u) + R(u) = f
    F = (u - u_n) * v * ufl.dx + dt * eps * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx \
        + dt * R(u) * v * ufl.dx - dt * f_ufl * v * ufl.dx
    
    J = ufl.derivative(F, u)
    
    t = t0
    n_steps = 0
    nl_iters = []
    
    # Time loop
    while t < t_end - 1e-8:
        t += dt_val
        t_ufl.value = t
        
        # Update BC
        u_bc.interpolate(ExactSol(t))
        bc = fem.dirichletbc(u_bc, boundary_dofs)
        
        problem = petsc.NonlinearProblem(F, u, bcs=[bc], J=J, petsc_options_prefix="rd_")
        solver = petsc.NewtonSolver(comm, problem)
        solver.atol = 1e-8
        solver.rtol = 1e-8
        solver.convergence_criterion = "incremental"
        solver.max_it = 50
        
        n_it, converged = solver.solve(u)
        
        nl_iters.append(n_it)
        u_n.x.array[:] = u.x.array
        n_steps += 1

    u_out_arr = sample_on_grid(domain, u, nx_out, ny_out, bbox)
    
    solver_info = {
        "mesh_resolution": mesh_res,
        "element_degree": elem_degree,
        "ksp_type": "preonly",
        "pc_type": "lu",
        "rtol": 1e-8,
        "iterations": sum(nl_iters), # approx
        "dt": dt_val,
        "n_steps": n_steps,
        "time_scheme": "backward_euler",
        "nonlinear_iterations": nl_iters
    }
    
    return {
        "u": u_out_arr,
        "u_initial": u_initial_arr,
        "solver_info": solver_info
    }

def sample_on_grid(domain, u_sol, nx, ny, bbox):
    xs = np.linspace(bbox[0], bbox[1], nx)
    ys = np.linspace(bbox[2], bbox[3], ny)
    XX, YY = np.meshgrid(xs, ys)
    pts = np.c_[XX.ravel(), YY.ravel(), np.zeros(nx * ny)]
    
    tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(tree, pts.T)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, pts.T)
    
    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    for i in range(pts.shape[0]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(pts[i])
            cells_on_proc.append(links[0])
            eval_map.append(i)
            
    u_values = np.full((pts.shape[0],), np.nan)
    if len(points_on_proc) > 0:
        vals = u_sol.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u_values[eval_map] = vals.flatten()
        
    return u_values.reshape(ny, nx)

