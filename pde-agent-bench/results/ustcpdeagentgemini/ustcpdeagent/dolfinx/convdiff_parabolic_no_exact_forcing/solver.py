import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
import ufl
from petsc4py import PETSc
from dolfinx.fem import petsc

def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD
    
    # Grid info
    nx_out = case_spec["output"]["grid"]["nx"]
    ny_out = case_spec["output"]["grid"]["ny"]
    bbox = case_spec["output"]["grid"]["bbox"]
    xmin, xmax, ymin, ymax = bbox
    
    # Mesh and function space
    nx, ny = 128, 128
    domain = mesh.create_rectangle(comm, [[xmin, ymin], [xmax, ymax]], [nx, ny], cell_type=mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", 1))
    
    # Time parameters
    t0 = 0.0
    t_end = 0.1
    dt = 0.01  # use smaller dt to be safe and accurate
    n_steps = int(np.ceil((t_end - t0) / dt))
    
    # Boundary conditions
    tdim = domain.topology.dim
    fdim = tdim - 1
    domain.topology.create_connectivity(fdim, tdim)
    boundary_facets = mesh.exterior_facet_indices(domain.topology)
    dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    u_bc = fem.Function(V)
    u_bc.x.array[:] = 0.0
    bc = fem.dirichletbc(u_bc, dofs)
    
    # Initial condition
    u_n = fem.Function(V)
    u_n.x.array[:] = 0.0
    
    u_initial = np.zeros((ny_out, nx_out))  # store exact IC on output grid later if needed
    
    # Variational problem
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    
    eps = 0.05
    beta = ufl.as_vector([2.0, 1.0])
    
    x = ufl.SpatialCoordinate(domain)
    f = ufl.sin(3 * ufl.pi * x[0]) * ufl.sin(2 * ufl.pi * x[1])
    
    # Backward Euler weak form
    # (u - u_n)/dt - eps * laplace(u) + beta . grad(u) = f
    # -> u*v/dt + eps * grad(u).grad(v) + (beta . grad(u))*v = f*v + u_n*v/dt
    
    dt_c = fem.Constant(domain, PETSc.ScalarType(dt))
    F = (u - u_n) * v / dt_c * ufl.dx \
      + eps * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx \
      + ufl.inner(beta, ufl.grad(u)) * v * ufl.dx \
      - f * v * ufl.dx
    
    a = ufl.lhs(F)
    L = ufl.rhs(F)
    
    # Solve
    problem = petsc.LinearProblem(
        a, L, bcs=[bc],
        petsc_options={"ksp_type": "preonly", "pc_type": "lu", "pc_factor_mat_solver_type": "mumps"},
        petsc_options_prefix="convdiff_"
    )
    
    u_sol = fem.Function(V)
    u_sol.x.array[:] = u_n.x.array[:]
    
    t = t0
    iterations = 0
    for i in range(n_steps):
        t += dt
        u_sol = problem.solve()
        u_n.x.array[:] = u_sol.x.array[:]
        iterations += 1  # 1 for direct solve
        
    # Interpolate to target grid
    xs = np.linspace(xmin, xmax, nx_out)
    ys = np.linspace(ymin, ymax, ny_out)
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
            
    u_values = np.full(pts.shape[0], np.nan)
    if len(points_on_proc) > 0:
        vals = u_sol.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u_values[eval_map] = vals.flatten()
        
    u_out = u_values.reshape((ny_out, nx_out))
    
    # Store initial
    u_initial_values = np.full(pts.shape[0], 0.0)
    u_init_out = u_initial_values.reshape((ny_out, nx_out))
    
    solver_info = {
        "mesh_resolution": nx,
        "element_degree": 1,
        "ksp_type": "preonly",
        "pc_type": "lu",
        "rtol": 1e-8,
        "iterations": iterations,
        "dt": dt,
        "n_steps": n_steps,
        "time_scheme": "backward_euler"
    }
    
    return {
        "u": u_out,
        "u_initial": u_init_out,
        "solver_info": solver_info
    }
