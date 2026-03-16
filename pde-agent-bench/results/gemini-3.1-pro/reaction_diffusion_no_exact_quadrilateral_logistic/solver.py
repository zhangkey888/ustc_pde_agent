```python
import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, nls, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc

def solve(case_spec: dict) -> dict:
    # Parameters
    nx = 64
    ny = 64
    degree = 1
    dt = 0.01
    t_end = 0.3
    rho = 5.0
    epsilon = 1.0
    
    comm = MPI.COMM_WORLD
    
    # Mesh
    domain = mesh.create_unit_square(comm, nx, ny, cell_type=mesh.CellType.quadrilateral)
    
    # Function space
    V = fem.functionspace(domain, ("Lagrange", degree))
    
    # Initial condition
    u_n = fem.Function(V)
    x = ufl.SpatialCoordinate(domain)
    u0_expr = 0.25 + 0.15 * ufl.sin(ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])
    u0_expr_form = fem.Expression(u0_expr, V.element.interpolation_points)
    u_n.interpolate(u0_expr_form)
    
    # Save initial condition for output
    u_initial = u_n.x.array.copy()
    
    # Boundary conditions
    fdim = domain.topology.dim - 1
    domain.topology.create_connectivity(fdim, domain.topology.dim)
    boundary_facets = mesh.exterior_facet_indices(domain.topology)
    dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc = fem.dirichletbc(PETSc.ScalarType(0.25), dofs, V)
    
    # Variational problem
    u = fem.Function(V)
    u.x.array[:] = u_n.x.array[:] # Initial guess for Newton
    v = ufl.TestFunction(V)
    
    f = fem.Constant(domain, PETSc.ScalarType(1.0))
    dt_c = fem.Constant(domain, PETSc.ScalarType(dt))
    rho_c = fem.Constant(domain, PETSc.ScalarType(rho))
    eps_c = fem.Constant(domain, PETSc.ScalarType(epsilon))
    
    # Backward Euler
    F = (u - u_n) / dt_c * v * ufl.dx \
      + eps_c * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx \
      - rho_c * u * (1.0 - u) * v * ufl.dx \
      - f * v * ufl.dx
      
    problem = petsc.NonlinearProblem(F, u, bcs=[bc])
    solver = nls.petsc.NewtonSolver(comm, problem)
    solver.convergence_criterion = "incremental"
    solver.rtol = 1e-6
    solver.atol = 1e-8
    solver.max_it = 50
    
    ksp = solver.krylov_solver
    ksp.setType("preonly")
    pc = ksp.getPC()
    pc.setType("lu")
    
    t = 0.0
    n_steps = int(np.round(t_end / dt))
    
    nonlinear_iterations = []
    
    for i in range(n_steps):
        t += dt
        num_its, converged = solver.solve(u)
        u.x.scatter_forward()
        u_n.x.array[:] = u.x.array[:]
        nonlinear_iterations.append(num_its)
        
    # Evaluate on 65x65 grid
    x_eval = np.linspace(0, 1, 65)
    y_eval = np.linspace(0, 1, 65)
    X, Y = np.meshgrid(x_eval, y_eval, indexing='ij')
    points = np.vstack((X.flatten(), Y.flatten(), np.zeros_like(X.flatten())))
    
    bb_tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, points.T)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points.T)
    
    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    for i in range(points.shape[1]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(points.T[i])
            cells_on_proc.append(links[0])
            eval_map.append(i)
            
    # Evaluate final solution
    u_values = np.zeros((points.shape[1],))
    mask = np.zeros((points.shape[1],), dtype=int)
    if len(points_on_proc) > 0:
        vals = u.eval(np.array(points_on_proc), np.array(