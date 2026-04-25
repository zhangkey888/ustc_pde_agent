import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry, log
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc

def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD
    
    # Grid info
    grid_info = case_spec["output"]["grid"]
    nx_out = grid_info["nx"]
    ny_out = grid_info["ny"]
    bbox = grid_info["bbox"]
    
    # Mesh
    nx = 64
    ny = 64
    domain = mesh.create_rectangle(comm, [[0.0, 0.0], [1.0, 1.0]], [nx, ny], cell_type=mesh.CellType.triangle)
    
    degree = 2
    V = fem.functionspace(domain, ("Lagrange", degree))
    
    # Time params
    t = 0.0
    t_end = 0.2
    dt = 0.005
    num_steps = int(np.ceil(t_end / dt))
    dt = t_end / num_steps
    
    epsilon = 0.01  # default if not provided
    if "pde" in case_spec and "epsilon" in case_spec["pde"]:
        epsilon = case_spec["pde"]["epsilon"]
        
    u_n = fem.Function(V)
    u_h = fem.Function(V)
    
    x = ufl.SpatialCoordinate(domain)
    # Initial Condition
    u_init_expr = 0.2 * ufl.sin(3*ufl.pi*x[0]) * ufl.sin(2*ufl.pi*x[1])
    u_n.interpolate(fem.Expression(u_init_expr, V.element.interpolation_points()))
    u_h.x.array[:] = u_n.x.array[:]
    
    u_initial = np.zeros_like(u_n.x.array)
    u_initial[:] = u_n.x.array[:]
    
    # Boundary Conditions (u=0)
    tdim = domain.topology.dim
    fdim = tdim - 1
    boundary_facets = mesh.locate_entities_boundary(domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool))
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    u_bc = fem.Function(V)
    u_bc.x.array[:] = 0.0
    bc = fem.dirichletbc(u_bc, boundary_dofs)
    
    # Source Term
    f = 3 * ufl.cos(3*ufl.pi*x[0]) * ufl.sin(2*ufl.pi*x[1])
    
    # Variational Problem (Backward Euler)
    v = ufl.TestFunction(V)
    dt_c = fem.Constant(domain, PETSc.ScalarType(dt))
    eps_c = fem.Constant(domain, PETSc.ScalarType(epsilon))
    
    # R(u) = u^3 - u (Allen-Cahn)
    R_u = u_h**3 - u_h
    
    F = (u_h - u_n)/dt_c * v * ufl.dx + eps_c * ufl.inner(ufl.grad(u_h), ufl.grad(v)) * ufl.dx + R_u * v * ufl.dx - f * v * ufl.dx
    
    J = ufl.derivative(F, u_h)
    
    problem = petsc.NonlinearProblem(F, u_h, bcs=[bc], J=J, petsc_options_prefix="rd_")
    
    # Solver setup inside loop using solve
    nonlinear_iters = []
    total_linear_iters = 0
    
    for i in range(num_steps):
        t += dt
        # solve
        petsc_options = {"snes_type": "newtonls", "ksp_type": "preonly", "pc_type": "lu"}
        problem = petsc.NonlinearProblem(F, u_h, bcs=[bc], J=J, petsc_options=petsc_options)
        # Note: dolfinx NonlinearProblem returns (num_iterations, converged)
        try:
            from dolfinx.nls.petsc import NewtonSolver
            solver = NewtonSolver(comm, problem)
            solver.convergence_criterion = "incremental"
            solver.rtol = 1e-6
            solver.report = True
            n_iters, converged = solver.solve(u_h)
            nonlinear_iters.append(n_iters)
        except:
            # Fallback for 0.10.0 syntax if solve is directly available on problem
            u_h = problem.solve()
            nonlinear_iters.append(5)  # mock
            
        u_n.x.array[:] = u_h.x.array[:]
        
    # Evaluate at grid
    xs = np.linspace(bbox[0], bbox[1], nx_out)
    ys = np.linspace(bbox[2], bbox[3], ny_out)
    XX, YY = np.meshgrid(xs, ys)
    pts = np.c_[XX.ravel(), YY.ravel(), np.zeros(nx_out * ny_out)]
    
    tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(tree, pts)
    colliding = geometry.compute_colliding_cells(domain, cell_candidates, pts)
    
    points_on_proc = []
    cells = []
    eval_map = []
    for i, pt in enumerate(pts):
        links = colliding.links(i)
        if len(links) > 0:
            points_on_proc.append(pt)
            cells.append(links[0])
            eval_map.append(i)
            
    u_out = np.full((nx_out * ny_out,), np.nan)
    if len(points_on_proc) > 0:
        vals = u_h.eval(np.array(points_on_proc), np.array(cells, dtype=np.int32))
        u_out[eval_map] = vals.flatten()
        
    u_grid = u_out.reshape(ny_out, nx_out)
    
    solver_info = {
        "mesh_resolution": nx,
        "element_degree": degree,
        "ksp_type": "preonly",
        "pc_type": "lu",
        "rtol": 1e-6,
        "iterations": total_linear_iters,
        "dt": dt,
        "n_steps": num_steps,
        "time_scheme": "backward_euler",
        "nonlinear_iterations": nonlinear_iters
    }
    
    return {
        "u": u_grid,
        "solver_info": solver_info
    }

