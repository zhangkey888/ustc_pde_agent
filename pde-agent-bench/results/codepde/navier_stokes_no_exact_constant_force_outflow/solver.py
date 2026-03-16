import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, nls, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc


def solve(case_spec: dict) -> dict:
    # 1. Parse case_spec
    pde = case_spec.get("pde", {})
    if not pde:
        pde = case_spec.get("oracle_config", {}).get("pde", {})
    
    nu_val = float(pde.get("viscosity", 0.3))
    source = pde.get("source_term", ["1.0", "0.0"])
    f0 = float(source[0]) if isinstance(source, list) else 1.0
    f1 = float(source[1]) if isinstance(source, list) else 0.0
    
    # Check for boundary conditions
    bcs_spec = pde.get("boundary_conditions", [])
    
    # 2. Create mesh - use fine mesh for accuracy
    N = 80
    degree_u = 2
    degree_p = 1
    
    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
    tdim = domain.topology.dim
    fdim = tdim - 1
    
    # 3. Create mixed function spaces (Taylor-Hood P2/P1)
    V = fem.functionspace(domain, ("Lagrange", degree_u, (domain.geometry.dim,)))
    Q = fem.functionspace(domain, ("Lagrange", degree_p))
    
    # Create mixed element
    vel_el = ufl.VectorElement("Lagrange", domain.ufl_cell(), degree_u)
    pres_el = ufl.FiniteElement("Lagrange", domain.ufl_cell(), degree_p)
    mixed_el = ufl.MixedElement([vel_el, pres_el])
    W = fem.functionspace(domain, mixed_el)
    
    # 4. Define boundary conditions
    # Parse boundary conditions from case_spec
    # "outflow" in the case name suggests we have outflow on some boundary
    # Typically: no-slip on walls, inflow on left, outflow (do-nothing) on right
    
    # For "navier_stokes_no_exact_constant_force_outflow":
    # This likely means:
    # - Some walls have no-slip (u=0)
    # - One boundary has outflow (natural BC, do nothing)
    # Let's parse the actual BCs
    
    bc_list = []
    
    # We need to figure out which boundaries have Dirichlet and which have outflow
    # From the case name: "outflow" suggests do-nothing on right boundary
    # Standard setup: no-slip on top, bottom, left; outflow on right
    
    has_outflow = False
    dirichlet_markers = {}
    
    for bc_item in bcs_spec:
        bc_type = bc_item.get("type", "")
        location = bc_item.get("location", "")
        value = bc_item.get("value", None)
        
        if bc_type.lower() == "dirichlet":
            dirichlet_markers[location] = value
        elif bc_type.lower() in ["neumann", "outflow", "do_nothing"]:
            has_outflow = True
    
    # If no explicit BCs found, use defaults based on case name
    if not bcs_spec:
        # Default: no-slip on top, bottom, left; outflow (do-nothing) on right
        dirichlet_markers = {
            "left": ["0.0", "0.0"],
            "top": ["0.0", "0.0"],
            "bottom": ["0.0", "0.0"],
        }
        has_outflow = True
    
    # Build Dirichlet BCs
    def left_boundary(x):
        return np.isclose(x[0], 0.0)
    
    def right_boundary(x):
        return np.isclose(x[0], 1.0)
    
    def bottom_boundary(x):
        return np.isclose(x[1], 0.0)
    
    def top_boundary(x):
        return np.isclose(x[1], 1.0)
    
    def all_walls_no_outflow(x):
        return np.isclose(x[0], 0.0) | np.isclose(x[1], 0.0) | np.isclose(x[1], 1.0)
    
    boundary_funcs = {
        "left": left_boundary,
        "right": right_boundary,
        "bottom": bottom_boundary,
        "top": top_boundary,
    }
    
    V_sub, V_sub_map = W.sub(0).collapse()
    
    for loc, val in dirichlet_markers.items():
        if loc in boundary_funcs:
            marker_func = boundary_funcs[loc]
        elif loc == "all":
            marker_func = lambda x: np.ones(x.shape[1], dtype=bool)
        else:
            continue
        
        facets = mesh.locate_entities_boundary(domain, fdim, marker_func)
        
        # Parse value
        if isinstance(val, list):
            v0 = float(val[0])
            v1 = float(val[1])
        elif isinstance(val, str):
            v0 = float(val)
            v1 = 0.0
        else:
            v0 = 0.0
            v1 = 0.0
        
        u_bc_func = fem.Function(V_sub)
        u_bc_func.interpolate(lambda x, _v0=v0, _v1=v1: np.stack([
            np.full_like(x[0], _v0),
            np.full_like(x[0], _v1)
        ]))
        
        dofs = fem.locate_dofs_topological((W.sub(0), V_sub), fdim, facets)
        bc_list.append(fem.dirichletbc(u_bc_func, dofs, W.sub(0)))
    
    # If no BCs were added from spec, add default no-slip on left, top, bottom
    if not bc_list:
        facets_noslip = mesh.locate_entities_boundary(domain, fdim, all_walls_no_outflow)
        u_zero = fem.Function(V_sub)
        u_zero.interpolate(lambda x: np.zeros((domain.geometry.dim, x.shape[1])))
        dofs_noslip = fem.locate_dofs_topological((W.sub(0), V_sub), fdim, facets_noslip)
        bc_list.append(fem.dirichletbc(u_zero, dofs_noslip, W.sub(0)))
    
    # 5. Define variational form (nonlinear residual)
    w = fem.Function(W)
    (u, p) = ufl.split(w)
    (v, q) = ufl.TestFunctions(W)
    
    nu = fem.Constant(domain, PETSc.ScalarType(nu_val))
    f = fem.Constant(domain, PETSc.ScalarType((f0, f1)))
    
    # Steady incompressible NS residual:
    # nu * inner(grad(u), grad(v)) + inner(grad(u)*u, v) - p*div(v) + div(u)*q - inner(f, v) = 0
    F = (
        nu * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
        + ufl.inner(ufl.grad(u) * u, v) * ufl.dx
        - p * ufl.div(v) * ufl.dx
        + ufl.div(u) * q * ufl.dx
        - ufl.inner(f, v) * ufl.dx
    )
    
    # 6. First solve Stokes for initial guess
    w_stokes = fem.Function(W)
    (u_s, p_s) = ufl.split(w_stokes)
    
    F_stokes = (
        nu * ufl.inner(ufl.grad(u_s), ufl.grad(v)) * ufl.dx
        - p_s * ufl.div(v) * ufl.dx
        + ufl.div(u_s) * q * ufl.dx
        - ufl.inner(f, v) * ufl.dx
    )
    
    # Stokes is linear, solve with NonlinearProblem (converges in 1 Newton step)
    problem_stokes = petsc.NonlinearProblem(F_stokes, w_stokes, bcs=bc_list)
    solver_stokes = nls.petsc.NewtonSolver(comm, problem_stokes)
    solver_stokes.convergence_criterion = "incremental"
    solver_stokes.rtol = 1e-10
    solver_stokes.atol = 1e-12
    solver_stokes.max_it = 5
    
    ksp_stokes = solver_stokes.krylov_solver
    ksp_stokes.setType(PETSc.KSP.Type.GMRES)
    pc_stokes = ksp_stokes.getPC()
    pc_stokes.setType(PETSc.PC.Type.LU)
    
    n_stokes, converged_stokes = solver_stokes.solve(w_stokes)
    w_stokes.x.scatter_forward()
    
    # Use Stokes solution as initial guess for NS
    w.x.array[:] = w_stokes.x.array[:]
    w.x.scatter_forward()
    
    # 7. Solve nonlinear NS
    problem = petsc.NonlinearProblem(F, w, bcs=bc_list)
    solver = nls.petsc.NewtonSolver(comm, problem)
    solver.convergence_criterion = "incremental"
    solver.rtol = 1e-8
    solver.atol = 1e-10
    solver.max_it = 50
    solver.relaxation_parameter = 1.0
    
    ksp = solver.krylov_solver
    ksp.setType(PETSc.KSP.Type.GMRES)
    ksp.setTolerances(rtol=1e-8, max_it=1000)
    pc = ksp.getPC()
    pc.setType(PETSc.PC.Type.LU)
    
    n_newton, converged = solver.solve(w)
    assert converged, f"Newton solver did not converge after {n_newton} iterations"
    w.x.scatter_forward()
    
    # 8. Extract velocity and compute magnitude on 50x50 grid
    nx_out, ny_out = 50, 50
    xs = np.linspace(0.0, 1.0, nx_out)
    ys = np.linspace(0.0, 1.0, ny_out)
    XX, YY = np.meshgrid(xs, ys, indexing='ij')
    
    points = np.zeros((3, nx_out * ny_out))
    points[0, :] = XX.ravel()
    points[1, :] = YY.ravel()
    
    # Extract velocity sub-function
    u_sol = w.sub(0).collapse()
    
    # Point evaluation
    bb_tree = geometry.bb_tree(domain, tdim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, points.T)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points.T)
    
    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    
    for i in range(points.shape[1]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(points[:, i])
            cells_on_proc.append(links[0])
            eval_map.append(i)
    
    u_values = np.full((points.shape[1], 2), np.nan)
    
    if len(points_on_proc) > 0:
        pts_arr = np.array(points_on_proc)
        cells_arr = np.array(cells_on_proc, dtype=np.int32)
        vals = u_sol.eval(pts_arr, cells_arr)
        for idx, global_idx in enumerate(eval_map):
            u_values[global_idx, :] = vals[idx, :2]
    
    # Compute velocity magnitude
    vel_mag = np.sqrt(u_values[:, 0]**2 + u_values[:, 1]**2)
    u_grid = vel_mag.reshape((nx_out, ny_out))
    
    return {
        "u": u_grid,
        "solver_info": {
            "mesh_resolution": N,
            "element_degree": degree_u,
            "ksp_type": "gmres",
            "pc_type": "lu",
            "rtol": 1e-8,
            "iterations": int(n_newton),
            "nonlinear_iterations": [int(n_newton)],
        }
    }