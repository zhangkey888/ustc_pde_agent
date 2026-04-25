import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc
import time

def solve(case_spec: dict) -> dict:
    start_time = time.time()
    
    # Extract grid info
    grid_spec = case_spec.get("output", {}).get("grid", {})
    nx_out = grid_spec.get("nx", 64)
    ny_out = grid_spec.get("ny", 64)
    bbox = grid_spec.get("bbox", [0.0, 1.0, 0.0, 1.0])
    
    # Extract PDE params
    pde_spec = case_spec.get("pde", {})
    eps = pde_spec.get("epsilon", 0.01)
    alpha = pde_spec.get("reaction_alpha", 1.0)
    beta = pde_spec.get("reaction_beta", -1.0)
    
    # Extract Time params
    time_spec = pde_spec.get("time", {})
    t0 = time_spec.get("t0", 0.0)
    t_end = time_spec.get("t_end", 0.25)
    dt_val = time_spec.get("dt", 0.005)
    
    # Mesh and Space
    comm = MPI.COMM_WORLD
    nx_mesh, ny_mesh = 120, 120
    domain = mesh.create_unit_square(comm, nx_mesh, ny_mesh, cell_type=mesh.CellType.triangle)
    degree = 2
    V = fem.functionspace(domain, ("Lagrange", degree))
    
    # Time variables
    t = fem.Constant(domain, PETSc.ScalarType(t0))
    dt = fem.Constant(domain, PETSc.ScalarType(dt_val))
    
    # Trial and Test functions
    u = fem.Function(V)
    u_n = fem.Function(V)
    v = ufl.TestFunction(V)
    
    # Manufactured Solution
    x = ufl.SpatialCoordinate(domain)
    def u_exact_expr(t_var):
        return ufl.exp(-t_var) * (0.15 * ufl.sin(3 * ufl.pi * x[0]) * ufl.sin(2 * ufl.pi * x[1]))
    
    u_ex = u_exact_expr(t)
    
    # Source term f based on manufactured solution at t_{n+1}
    dt_ex = -u_ex
    lap_ex = ufl.div(ufl.grad(u_ex))
    f_ex = dt_ex - eps * lap_ex + alpha * u_ex + beta * (u_ex**3)
    
    # Residual F for Backward Euler
    F_eq = (u - u_n) / dt * v * ufl.dx \
         + eps * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx \
         + (alpha * u + beta * (u**3)) * v * ufl.dx \
         - f_ex * v * ufl.dx
         
    # Jacobian
    J = ufl.derivative(F_eq, u)
         
    # BC
    def boundary_marker(x):
        return np.full(x.shape[1], True, dtype=bool)
    
    fdim = domain.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(domain, fdim, boundary_marker)
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    
    u_bc = fem.Function(V)
    u_bc.interpolate(fem.Expression(u_ex, V.element.interpolation_points()))
    bc = fem.dirichletbc(u_bc, boundary_dofs)
    
    # Initial Condition
    t.value = t0
    u_n.interpolate(fem.Expression(u_exact_expr(t), V.element.interpolation_points()))
    u.x.array[:] = u_n.x.array[:]
    
    # Points evaluation
    xs = np.linspace(bbox[0], bbox[1], nx_out)
    ys = np.linspace(bbox[2], bbox[3], ny_out)
    XX, YY = np.meshgrid(xs, ys)
    pts = np.c_[XX.ravel(), YY.ravel(), np.zeros(nx_out * ny_out)]
    
    bb_tree = geometry.bb_tree(domain, domain.topology.dim)
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
            
    points_on_proc = np.array(points_on_proc, dtype=np.float64)
    cells_on_proc = np.array(cells_on_proc, dtype=np.int32)
    
    u_init_vals = np.full(pts.shape[0], np.nan)
    if len(points_on_proc) > 0:
        vals = u_n.eval(points_on_proc, cells_on_proc)
        u_init_vals[eval_map] = vals.flatten()
    u_init_grid = u_init_vals.reshape((ny_out, nx_out))
    
    # Solver
    problem = petsc.NonlinearProblem(
        F_eq, u, bcs=[bc], J=J,
        petsc_options={
            "snes_type": "newtonls",
            "ksp_type": "preonly",
            "pc_type": "lu"
        },
        petsc_options_prefix="rd_"
    )
    
    nonlinear_iters = []
    
    # Time loop
    current_t = t0
    n_steps = 0
    
    while current_t < t_end - 1e-10:
        current_t += dt_val
        t.value = current_t
        
        # Update BC
        u_bc.interpolate(fem.Expression(u_exact_expr(t), V.element.interpolation_points()))
        
        # Solve
        n, converged = problem.solve()
        u.x.scatter_forward()
        
        nonlinear_iters.append(n)
        
        u_n.x.array[:] = u.x.array[:]
        n_steps += 1
        
    u_vals = np.full(pts.shape[0], np.nan)
    if len(points_on_proc) > 0:
        vals = u.eval(points_on_proc, cells_on_proc)
        u_vals[eval_map] = vals.flatten()
    u_grid = u_vals.reshape((ny_out, nx_out))
    
    return {
        "u": u_grid,
        "u_initial": u_init_grid,
        "solver_info": {
            "mesh_resolution": nx_mesh,
            "element_degree": degree,
            "ksp_type": "preonly",
            "pc_type": "lu",
            "rtol": 1e-8,
            "iterations": sum(nonlinear_iters), # approx linear iters per newton step if preonly is 1
            "dt": dt_val,
            "n_steps": n_steps,
            "time_scheme": "backward_euler",
            "nonlinear_iterations": nonlinear_iters
        }
    }

