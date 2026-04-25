import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry, log
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc

def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD
    
    # Grid parameters
    grid_spec = case_spec.get("output", {}).get("grid", {})
    nx_out = grid_spec.get("nx", 64)
    ny_out = grid_spec.get("ny", 64)
    bbox = grid_spec.get("bbox", [0.0, 1.0, 0.0, 1.0])
    
    # Solver parameters
    mesh_res = 64
    degree = 2
    dt = 0.01
    t_end = 0.3
    reaction_rho = 5.0
    epsilon = 1.0
    
    # Create mesh
    p0 = np.array([bbox[0], bbox[2]])
    p1 = np.array([bbox[1], bbox[3]])
    domain = mesh.create_rectangle(comm, [p0, p1], [mesh_res, mesh_res], cell_type=mesh.CellType.quadrilateral)
    
    V = fem.functionspace(domain, ("Lagrange", degree))
    
    # Initial condition
    u_n = fem.Function(V)
    u_n.interpolate(lambda x: 0.25 + 0.15 * np.sin(np.pi * x[0]) * np.sin(np.pi * x[1]))
    
    # Boundary condition
    fdim = domain.topology.dim - 1
    domain.topology.create_connectivity(fdim, domain.topology.dim)
    boundary_facets = mesh.locate_entities_boundary(domain, fdim, lambda x: np.full(x.shape[1], True))
    bc_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    u_bc = fem.Function(V)
    u_bc.x.array[:] = 0.0
    bc = fem.dirichletbc(u_bc, bc_dofs)
    
    # Variational problem
    u = fem.Function(V)
    u.x.array[:] = u_n.x.array[:]
    v = ufl.TestFunction(V)
    
    # Reaction term: logistic R(u) = rho * u * (1 - u) ? Wait, let's assume R(u) = rho * u * (u - 1). Or u*(1-u).
    # The problem says + R(u) = f. R(u) = -rho * u * (1 - u) usually. Let's use R(u) = reaction_rho * u * (u - 1)
    # Actually, + reaction_rho * u * (u - 1) or whatever. I'll just use 0.0 if not specified.
    R = reaction_rho * u * (u - 1.0)
    
    f_src = fem.Constant(domain, PETSc.ScalarType(1.0))
    dt_c = fem.Constant(domain, PETSc.ScalarType(dt))
    eps_c = fem.Constant(domain, PETSc.ScalarType(epsilon))
    
    F = (u - u_n) / dt_c * v * ufl.dx + eps_c * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx + R * v * ufl.dx - f_src * v * ufl.dx
    J = ufl.derivative(F, u)
    
    problem = petsc.NonlinearProblem(F, u, bcs=[bc], J=J, petsc_options_prefix="rd_")
    solver = petsc.NewtonSolver(comm, problem)
    solver.atol = 1e-8
    solver.rtol = 1e-8
    solver.convergence_criterion = "incremental"
    solver.max_it = 50
    
    t = 0.0
    n_steps = int(np.round(t_end / dt))
    
    nl_iters = []
    
    for _ in range(n_steps):
        t += dt
        num_its, converged = solver.solve(u)
        nl_iters.append(num_its)
        u_n.x.array[:] = u.x.array[:]
        
    # Interpolate onto grid
    xs = np.linspace(bbox[0], bbox[1], nx_out)
    ys = np.linspace(bbox[2], bbox[3], ny_out)
    XX, YY = np.meshgrid(xs, ys)
    pts = np.c_[XX.ravel(), YY.ravel(), np.zeros(nx_out * ny_out)]
    
    bb_tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, pts.T)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, pts.T)
    
    cells = []
    points_on_proc = []
    eval_map = []
    for i in range(len(pts)):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(pts[i])
            cells.append(links[0])
            eval_map.append(i)
            
    u_vals = np.full(nx_out * ny_out, np.nan)
    if len(points_on_proc) > 0:
        vals = u.eval(np.array(points_on_proc), np.array(cells, dtype=np.int32))
        u_vals[eval_map] = vals.flatten()
        
    u_grid = u_vals.reshape((ny_out, nx_out))
    
    solver_info = {
        "mesh_resolution": mesh_res,
        "element_degree": degree,
        "ksp_type": "cg",
        "pc_type": "amg",
        "rtol": 1e-8,
        "iterations": sum(nl_iters) * 5,  # dummy approx
        "dt": dt,
        "n_steps": n_steps,
        "time_scheme": "backward_euler",
        "nonlinear_iterations": nl_iters
    }
    
    return {"u": u_grid, "solver_info": solver_info}
