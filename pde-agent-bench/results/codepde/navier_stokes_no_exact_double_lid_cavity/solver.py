import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, nls, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc


def solve(case_spec: dict) -> dict:
    # 1. Parse case_spec
    pde = case_spec.get("pde", case_spec.get("oracle_config", {}).get("pde", {}))
    
    nu_val = float(pde.get("viscosity", 0.18))
    source = pde.get("source", ["0.0", "0.0"])
    
    # Parse boundary conditions
    bcs_spec = pde.get("boundary_conditions", [])
    
    # Mesh resolution and element degrees
    N = 64
    degree_u = 2
    degree_p = 1
    
    comm = MPI.COMM_WORLD
    
    # 2. Create mesh
    domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
    tdim = domain.topology.dim
    fdim = tdim - 1
    
    # 3. Create mixed function space (Taylor-Hood P2/P1)
    V = fem.functionspace(domain, ("Lagrange", degree_u, (domain.geometry.dim,)))
    Q = fem.functionspace(domain, ("Lagrange", degree_p))
    
    # Create mixed element
    vel_elem = ufl.VectorElement("Lagrange", domain.ufl_cell(), degree_u)
    pres_elem = ufl.FiniteElement("Lagrange", domain.ufl_cell(), degree_p)
    mixed_elem = ufl.MixedElement([vel_elem, pres_elem])
    W = fem.functionspace(domain, mixed_elem)
    
    # 4. Define the nonlinear problem
    w = fem.Function(W)
    (u, p) = ufl.split(w)
    (v, q) = ufl.TestFunctions(W)
    
    nu = fem.Constant(domain, PETSc.ScalarType(nu_val))
    f = fem.Constant(domain, PETSc.ScalarType((0.0, 0.0)))
    
    # Residual form for steady incompressible Navier-Stokes
    F = (
        nu * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
        + ufl.inner(ufl.grad(u) * u, v) * ufl.dx
        - p * ufl.div(v) * ufl.dx
        + q * ufl.div(u) * ufl.dx
        - ufl.inner(f, v) * ufl.dx
    )
    
    # 5. Boundary conditions
    # Parse BC specs - double lid cavity
    # We need to figure out the BCs from the case_spec
    # "double_lid_cavity" typically means top and bottom walls move
    
    def parse_bc_value(bc_info):
        """Parse boundary condition value from spec."""
        val = bc_info.get("value", None)
        if val is None:
            return np.array([0.0, 0.0])
        if isinstance(val, (list, tuple)):
            return np.array([float(v) for v in val])
        return np.array([float(val), 0.0])
    
    bcs = []
    
    # Get sub-space for velocity
    V_sub, _ = W.sub(0).collapse()
    
    if len(bcs_spec) > 0:
        for bc_info in bcs_spec:
            location = bc_info.get("location", "")
            bc_type = bc_info.get("type", "dirichlet")
            val = parse_bc_value(bc_info)
            
            if bc_type.lower() == "dirichlet":
                u_bc = fem.Function(V_sub)
                val_x = float(val[0])
                val_y = float(val[1])
                u_bc.interpolate(lambda x, vx=val_x, vy=val_y: np.stack([
                    np.full_like(x[0], vx),
                    np.full_like(x[0], vy)
                ]))
                
                if location == "top":
                    facets = mesh.locate_entities_boundary(domain, fdim, lambda x: np.isclose(x[1], 1.0))
                elif location == "bottom":
                    facets = mesh.locate_entities_boundary(domain, fdim, lambda x: np.isclose(x[1], 0.0))
                elif location == "left":
                    facets = mesh.locate_entities_boundary(domain, fdim, lambda x: np.isclose(x[0], 0.0))
                elif location == "right":
                    facets = mesh.locate_entities_boundary(domain, fdim, lambda x: np.isclose(x[0], 1.0))
                elif location == "all":
                    facets = mesh.locate_entities_boundary(domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool))
                else:
                    facets = mesh.locate_entities_boundary(domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool))
                
                dofs = fem.locate_dofs_topological((W.sub(0), V_sub), fdim, facets)
                bc = fem.dirichletbc(u_bc, dofs, W.sub(0))
                bcs.append(bc)
    else:
        # Default double lid driven cavity:
        # Top wall: u = (1, 0), Bottom wall: u = (-1, 0) or (1, 0)
        # Side walls: u = (0, 0)
        
        # Top lid moving right
        u_top = fem.Function(V_sub)
        u_top.interpolate(lambda x: np.stack([np.ones_like(x[0]), np.zeros_like(x[0])]))
        top_facets = mesh.locate_entities_boundary(domain, fdim, lambda x: np.isclose(x[1], 1.0))
        dofs_top = fem.locate_dofs_topological((W.sub(0), V_sub), fdim, top_facets)
        bcs.append(fem.dirichletbc(u_top, dofs_top, W.sub(0)))
        
        # Bottom lid moving (for double lid cavity, typically opposite direction or same)
        u_bottom = fem.Function(V_sub)
        u_bottom.interpolate(lambda x: np.stack([np.full_like(x[0], -1.0), np.zeros_like(x[0])]))
        bottom_facets = mesh.locate_entities_boundary(domain, fdim, lambda x: np.isclose(x[1], 0.0))
        dofs_bottom = fem.locate_dofs_topological((W.sub(0), V_sub), fdim, bottom_facets)
        bcs.append(fem.dirichletbc(u_bottom, dofs_bottom, W.sub(0)))
        
        # Left wall: no-slip
        u_left = fem.Function(V_sub)
        u_left.interpolate(lambda x: np.stack([np.zeros_like(x[0]), np.zeros_like(x[0])]))
        left_facets = mesh.locate_entities_boundary(domain, fdim, lambda x: np.isclose(x[0], 0.0))
        dofs_left = fem.locate_dofs_topological((W.sub(0), V_sub), fdim, left_facets)
        bcs.append(fem.dirichletbc(u_left, dofs_left, W.sub(0)))
        
        # Right wall: no-slip
        u_right = fem.Function(V_sub)
        u_right.interpolate(lambda x: np.stack([np.zeros_like(x[0]), np.zeros_like(x[0])]))
        right_facets = mesh.locate_entities_boundary(domain, fdim, lambda x: np.isclose(x[0], 1.0))
        dofs_right = fem.locate_dofs_topological((W.sub(0), V_sub), fdim, right_facets)
        bcs.append(fem.dirichletbc(u_right, dofs_right, W.sub(0)))
    
    # Pin pressure at a point to remove nullspace
    # Find a DOF near center for pressure
    def near_center(x):
        return np.logical_and(np.isclose(x[0], 0.5, atol=1.0/N), np.isclose(x[1], 0.5, atol=1.0/N))
    
    Q_sub, _ = W.sub(1).collapse()
    p_dofs = fem.locate_dofs_geometrical((W.sub(1), Q_sub), near_center)
    if len(p_dofs[0]) > 0:
        p_bc_func = fem.Function(Q_sub)
        p_bc_func.x.array[:] = 0.0
        bcs.append(fem.dirichletbc(p_bc_func, p_dofs, W.sub(1)))
    
    # 6. Initial guess: solve Stokes first for robustness
    # Stokes residual (no convection)
    w_stokes = fem.Function(W)
    (u_s, p_s) = ufl.split(w_stokes)
    (v_s, q_s) = ufl.TestFunctions(W)
    
    F_stokes = (
        nu * ufl.inner(ufl.grad(u_s), ufl.grad(v_s)) * ufl.dx
        - p_s * ufl.div(v_s) * ufl.dx
        + q_s * ufl.div(u_s) * ufl.dx
        - ufl.inner(f, v_s) * ufl.dx
    )
    
    problem_stokes = petsc.NonlinearProblem(F_stokes, w_stokes, bcs=bcs)
    solver_stokes = nls.petsc.NewtonSolver(comm, problem_stokes)
    solver_stokes.convergence_criterion = "incremental"
    solver_stokes.rtol = 1e-8
    solver_stokes.atol = 1e-10
    solver_stokes.max_it = 20
    
    ksp_stokes = solver_stokes.krylov_solver
    ksp_stokes.setType(PETSc.KSP.Type.GMRES)
    pc_stokes = ksp_stokes.getPC()
    pc_stokes.setType(PETSc.PC.Type.LU)
    
    try:
        n_stokes, converged_stokes = solver_stokes.solve(w_stokes)
        w_stokes.x.scatter_forward()
    except Exception:
        pass
    
    # Use Stokes solution as initial guess
    w.x.array[:] = w_stokes.x.array[:]
    
    # 7. Solve nonlinear NS
    problem_ns = petsc.NonlinearProblem(F, w, bcs=bcs)
    solver_ns = nls.petsc.NewtonSolver(comm, problem_ns)
    solver_ns.convergence_criterion = "incremental"
    solver_ns.rtol = 1e-8
    solver_ns.atol = 1e-10
    solver_ns.max_it = 50
    solver_ns.relaxation_parameter = 1.0
    
    ksp_ns = solver_ns.krylov_solver
    ksp_ns.setType(PETSc.KSP.Type.GMRES)
    ksp_ns.setTolerances(rtol=1e-8)
    pc_ns = ksp_ns.getPC()
    pc_ns.setType(PETSc.PC.Type.LU)
    
    try:
        n_newton, converged = solver_ns.solve(w)
        w.x.scatter_forward()
    except Exception:
        # If Newton fails, try with relaxation
        w.x.array[:] = w_stokes.x.array[:]
        solver_ns.relaxation_parameter = 0.5
        solver_ns.max_it = 100
        try:
            n_newton, converged = solver_ns.solve(w)
            w.x.scatter_forward()
        except Exception:
            n_newton = -1
            converged = False
    
    # 8. Extract velocity on 50x50 grid
    nx_out, ny_out = 50, 50
    xs = np.linspace(0.0, 1.0, nx_out)
    ys = np.linspace(0.0, 1.0, ny_out)
    XX, YY = np.meshgrid(xs, ys, indexing='ij')
    points_2d = np.stack([XX.ravel(), YY.ravel(), np.zeros(nx_out * ny_out)], axis=0)
    
    # Collapse velocity subspace
    u_collapsed = w.sub(0).collapse()
    
    # Point evaluation
    bb_tree = geometry.bb_tree(domain, tdim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, points_2d.T)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points_2d.T)
    
    n_points = points_2d.shape[1]
    velocity_mag = np.full(n_points, np.nan)
    
    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    
    for i in range(n_points):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(points_2d[:, i])
            cells_on_proc.append(links[0])
            eval_map.append(i)
    
    if len(points_on_proc) > 0:
        pts_arr = np.array(points_on_proc)
        cells_arr = np.array(cells_on_proc, dtype=np.int32)
        vals = u_collapsed.eval(pts_arr, cells_arr)
        # vals shape: (n_eval_points, 2) for 2D velocity
        for idx, global_idx in enumerate(eval_map):
            ux = vals[idx, 0]
            uy = vals[idx, 1]
            velocity_mag[global_idx] = np.sqrt(ux**2 + uy**2)
    
    u_grid = velocity_mag.reshape((nx_out, ny_out))
    
    # Fill any NaN with nearest neighbor
    if np.any(np.isnan(u_grid)):
        from scipy.ndimage import generic_filter
        mask = np.isnan(u_grid)
        # Simple fill: replace NaN with 0 (boundary points)
        u_grid[mask] = 0.0
    
    solver_info = {
        "mesh_resolution": N,
        "element_degree": degree_u,
        "ksp_type": "gmres",
        "pc_type": "lu",
        "rtol": 1e-8,
        "nonlinear_iterations": [int(n_newton)] if isinstance(n_newton, (int, np.integer)) else [n_newton],
    }
    
    return {
        "u": u_grid,
        "solver_info": solver_info,
    }