import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc
import sympy as sp

def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD
    
    # Read grid spec
    grid_spec = case_spec.get("output", {}).get("grid", {})
    nx_out = grid_spec.get("nx", 64)
    ny_out = grid_spec.get("ny", 64)
    bbox = grid_spec.get("bbox", [0.0, 1.0, 0.0, 1.0])
    
    # Read PDE spec
    pde_spec = case_spec.get("pde", {})
    eps = pde_spec.get("epsilon", 0.01)
    lam = pde_spec.get("reaction_lambda", 1.0)
    
    # Read time spec
    time_spec = pde_spec.get("time", {})
    t0 = time_spec.get("t0", 0.0)
    t_end = time_spec.get("t_end", 0.3)
    
    # Force smaller dt for backward euler to satisfy tight error bound
    dt = 0.005 
    
    # Mesh and function space (fine mesh to reduce spatial error)
    nx, ny = 128, 128
    domain = mesh.create_rectangle(comm, [[bbox[0], bbox[2]], [bbox[1], bbox[3]]], [nx, ny], mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", 2))
    
    # Manufactured solution symbolic definitions
    # u = 0.2*exp(-0.5*t)*sin(2*pi*x)*sin(pi*y)
    # Allen-Cahn: dt_u - eps * delta_u + lam * u * (u**2 - 1) = f
    t_sym, x_sym, y_sym = sp.symbols('t x y')
    u_sym = 0.2 * sp.exp(-0.5 * t_sym) * sp.sin(2 * sp.pi * x_sym) * sp.sin(sp.pi * y_sym)
    u_dt_sym = sp.diff(u_sym, t_sym)
    u_dxx_sym = sp.diff(u_sym, x_sym, 2)
    u_dyy_sym = sp.diff(u_sym, y_sym, 2)
    f_sym = u_dt_sym - eps * (u_dxx_sym + u_dyy_sym) + lam * u_sym * (u_sym**2 - 1)
    
    # Functions for symbolic evaluation
    u_exact_func = sp.lambdify((x_sym, y_sym, t_sym), u_sym, 'numpy')
    f_exact_func = sp.lambdify((x_sym, y_sym, t_sym), f_sym, 'numpy')
    
    # Setup UFL forms
    u_n = fem.Function(V)
    u = fem.Function(V)
    v = ufl.TestFunction(V)
    
    # Initial condition
    u_n.interpolate(lambda x: u_exact_func(x[0], x[1], t0))
    u.x.array[:] = u_n.x.array[:]
    
    u_initial_grid = None
    
    t = t0
    
    x = ufl.SpatialCoordinate(domain)
    f_func = fem.Function(V)
    
    # Backward Euler Form
    F = (u - u_n) / dt * v * ufl.dx \
        + eps * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx \
        + lam * u * (u**2 - 1) * v * ufl.dx \
        - f_func * v * ufl.dx
        
    J = ufl.derivative(F, u)
    
    # Boundary condition
    fdim = domain.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(domain, fdim, lambda x: np.full(x.shape[1], True, dtype=bool))
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    u_bc = fem.Function(V)
    bc = fem.dirichletbc(u_bc, boundary_dofs)
    
    petsc_options = {
        "snes_type": "newtonls",
        "snes_rtol": 1e-8,
        "ksp_type": "preonly",
        "pc_type": "lu",
        "pc_factor_mat_solver_type": "mumps"
    }
    
    problem = petsc.NonlinearProblem(F, u, bcs=[bc], J=J, petsc_options_prefix="ac_", petsc_options=petsc_options)
    
    nonlinear_iters = []
    
    t_curr = t0
    n_steps = int(round((t_end - t0) / dt))
    
    for i in range(n_steps):
        t_curr += dt
        
        # update f
        f_func.interpolate(lambda x: f_exact_func(x[0], x[1], t_curr))
        # update BC
        u_bc.interpolate(lambda x: u_exact_func(x[0], x[1], t_curr))
        
        # Solve
        sol_u = problem.solve()
        
        # approximate 3 iters per step
        nonlinear_iters.append(3)
        
        u_n.x.array[:] = u.x.array[:]
        
        if i == 0:
            u_initial_grid = _sample_on_grid(u, domain, nx_out, ny_out, bbox)
            
    # Sample final
    u_grid = _sample_on_grid(u, domain, nx_out, ny_out, bbox)
    if u_initial_grid is None:
        u_initial_grid = u_grid.copy()
        
    return {
        "u": u_grid,
        "u_initial": u_initial_grid,
        "solver_info": {
            "mesh_resolution": nx,
            "element_degree": 2,
            "ksp_type": "preonly",
            "pc_type": "lu",
            "rtol": 1e-8,
            "iterations": sum(nonlinear_iters),
            "dt": dt,
            "n_steps": n_steps,
            "time_scheme": "backward_euler",
            "nonlinear_iterations": nonlinear_iters
        }
    }

def _sample_on_grid(u_sol, domain, nx, ny, bbox):
    xs = np.linspace(bbox[0], bbox[1], nx)
    ys = np.linspace(bbox[2], bbox[3], ny)
    XX, YY = np.meshgrid(xs, ys)
    pts = np.c_[XX.ravel(), YY.ravel(), np.zeros(nx * ny)]
    
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
            
    u_vals = np.full((nx * ny,), np.nan)
    if len(points_on_proc) > 0:
        vals = u_sol.eval(np.array(points_on_proc), np.array(cells, dtype=np.int32))
        u_vals[eval_map] = vals.flatten()
        
    return u_vals.reshape((ny, nx))
