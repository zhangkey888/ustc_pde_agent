import numpy as np
from dolfinx import mesh, fem, nls, geometry
from dolfinx.fem import petsc
from mpi4py import MPI
import ufl
from petsc4py import PETSc


def solve(case_spec: dict) -> dict:
    # 1. Parse case_spec
    pde = case_spec.get("pde", case_spec.get("oracle_config", {}).get("pde", {}))
    nu_val = float(pde.get("viscosity", 0.22))
    f_expr = pde.get("source_term", ["0.0", "0.0"])
    bc_specs = pde.get("boundary_conditions", [])
    
    # 2. Create mesh - use a fine mesh for accuracy
    N = 80
    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
    tdim = domain.topology.dim
    fdim = tdim - 1
    
    # 3. Mixed function space (Taylor-Hood P2/P1)
    degree_u = 2
    degree_p = 1
    V = fem.functionspace(domain, ("Lagrange", degree_u, (domain.geometry.dim,)))
    Q = fem.functionspace(domain, ("Lagrange", degree_p))
    
    # Create mixed element
    vel_el = ufl.VectorElement("Lagrange", domain.ufl_cell(), degree_u)
    pres_el = ufl.FiniteElement("Lagrange", domain.ufl_cell(), degree_p)
    mixed_el = ufl.MixedElement([vel_el, pres_el])
    W = fem.functionspace(domain, mixed_el)
    
    # 4. Define unknown and test functions
    w = fem.Function(W)
    (u, p) = ufl.split(w)
    (v, q) = ufl.TestFunctions(W)
    
    x = ufl.SpatialCoordinate(domain)
    
    # Source term
    f0 = float(f_expr[0]) if isinstance(f_expr[0], (int, float, str)) else 0.0
    f1 = float(f_expr[1]) if isinstance(f_expr[1], (int, float, str)) else 0.0
    f = ufl.as_vector([f0, f1])
    
    nu = fem.Constant(domain, PETSc.ScalarType(nu_val))
    
    # Residual form
    F = (
        nu * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
        + ufl.inner(ufl.grad(u) * u, v) * ufl.dx
        - p * ufl.div(v) * ufl.dx
        + q * ufl.div(u) * ufl.dx
        - ufl.inner(f, v) * ufl.dx
    )
    
    # 5. Boundary conditions
    # Parse boundary conditions from case_spec
    bcs = []
    
    def parse_bc_value(bc_info):
        """Parse boundary condition value expressions."""
        val = bc_info.get("value", None)
        bc_type = bc_info.get("type", "dirichlet")
        if val is None:
            return None, bc_type
        return val, bc_type
    
    def make_bc_func(val_expr, V_sub, W_sub):
        """Create a boundary condition function from expression."""
        u_bc = fem.Function(V_sub)
        if isinstance(val_expr, (list, tuple)):
            # Vector BC
            expr_strs = [str(v) for v in val_expr]
            def bc_interp(X):
                vals = np.zeros((domain.geometry.dim, X.shape[1]))
                xv = X[0]
                yv = X[1]
                for i, es in enumerate(expr_strs):
                    if i < domain.geometry.dim:
                        vals[i] = eval(es, {"np": np, "x": xv, "y": yv, 
                                           "sin": np.sin, "cos": np.cos, "pi": np.pi,
                                           "exp": np.exp, "sqrt": np.sqrt,
                                           "__builtins__": {}})
                return vals
            u_bc.interpolate(bc_interp)
        elif isinstance(val_expr, str):
            def bc_interp(X):
                xv = X[0]
                yv = X[1]
                result = eval(val_expr, {"np": np, "x": xv, "y": yv,
                                        "sin": np.sin, "cos": np.cos, "pi": np.pi,
                                        "exp": np.exp, "sqrt": np.sqrt,
                                        "__builtins__": {}})
                return np.full(X.shape[1], float(result)) if np.isscalar(result) else result
            u_bc.interpolate(bc_interp)
        else:
            val_f = float(val_expr)
            u_bc.interpolate(lambda X: np.full((domain.geometry.dim, X.shape[1]), val_f) if V_sub.element.basix_element.value_shape == (domain.geometry.dim,) else np.full(X.shape[1], val_f))
        return u_bc
    
    # Map boundary location strings to marker functions
    def get_boundary_marker(location):
        loc = location.lower().strip()
        if loc == "left" or "x[0] < " in loc or "near(x[0], 0" in loc or "x=0" in loc:
            return lambda X: np.isclose(X[0], 0.0)
        elif loc == "right" or "x[0] > " in loc or "near(x[0], 1" in loc or "x=1" in loc:
            return lambda X: np.isclose(X[0], 1.0)
        elif loc == "bottom" or "x[1] < " in loc or "near(x[1], 0" in loc or "y=0" in loc:
            return lambda X: np.isclose(X[1], 0.0)
        elif loc == "top" or "x[1] > " in loc or "near(x[1], 1" in loc or "y=1" in loc:
            return lambda X: np.isclose(X[1], 1.0)
        elif loc == "all" or loc == "entire boundary" or loc == "boundary":
            return lambda X: np.ones(X.shape[1], dtype=bool)
        else:
            # Try to detect combined boundaries
            return lambda X: np.ones(X.shape[1], dtype=bool)
    
    has_velocity_bcs = False
    
    for bc_info in bc_specs:
        val_expr, bc_type = parse_bc_value(bc_info)
        if val_expr is None:
            continue
        
        location = bc_info.get("location", "all")
        variable = bc_info.get("variable", "velocity")
        
        if bc_type.lower() == "dirichlet":
            marker = get_boundary_marker(location)
            facets = mesh.locate_entities_boundary(domain, fdim, marker)
            
            if variable.lower() in ["velocity", "u", "v"]:
                has_velocity_bcs = True
                u_bc = fem.Function(V)
                if isinstance(val_expr, (list, tuple)):
                    expr_strs = [str(vv) for vv in val_expr]
                    def make_interp(es_list):
                        def bc_interp(X):
                            vals = np.zeros((domain.geometry.dim, X.shape[1]))
                            xv = X[0]
                            yv = X[1]
                            for i, es in enumerate(es_list):
                                if i < domain.geometry.dim:
                                    try:
                                        vals[i] = eval(es, {"np": np, "x": xv, "y": yv,
                                                           "sin": np.sin, "cos": np.cos, "pi": np.pi,
                                                           "exp": np.exp, "sqrt": np.sqrt, "abs": np.abs,
                                                           "__builtins__": {}})
                                    except:
                                        vals[i] = 0.0
                            return vals
                        return bc_interp
                    u_bc.interpolate(make_interp(expr_strs))
                else:
                    val_f = float(val_expr)
                    u_bc.interpolate(lambda X: np.full((domain.geometry.dim, X.shape[1]), val_f))
                
                dofs = fem.locate_dofs_topological((W.sub(0), V), fdim, facets)
                bcs.append(fem.dirichletbc(u_bc, dofs, W.sub(0)))
            
            elif variable.lower() in ["pressure", "p"]:
                p_bc = fem.Function(Q)
                if isinstance(val_expr, str):
                    def make_p_interp(es):
                        def p_interp(X):
                            xv = X[0]
                            yv = X[1]
                            try:
                                result = eval(es, {"np": np, "x": xv, "y": yv,
                                                  "sin": np.sin, "cos": np.cos, "pi": np.pi,
                                                  "__builtins__": {}})
                            except:
                                result = 0.0
                            return np.full(X.shape[1], float(result)) if np.isscalar(result) else result
                        return p_interp
                    p_bc.interpolate(make_p_interp(str(val_expr)))
                else:
                    val_f = float(val_expr)
                    p_bc.interpolate(lambda X: np.full(X.shape[1], val_f))
                
                dofs_p = fem.locate_dofs_topological((W.sub(1), Q), fdim, facets)
                bcs.append(fem.dirichletbc(p_bc, dofs_p, W.sub(1)))
    
    # If no BCs were parsed, apply default no-slip on all boundaries
    if not has_velocity_bcs:
        all_facets = mesh.locate_entities_boundary(domain, fdim, lambda X: np.ones(X.shape[1], dtype=bool))
        u_bc = fem.Function(V)
        u_bc.interpolate(lambda X: np.zeros((domain.geometry.dim, X.shape[1])))
        dofs = fem.locate_dofs_topological((W.sub(0), V), fdim, all_facets)
        bcs.append(fem.dirichletbc(u_bc, dofs, W.sub(0)))
    
    # 6. Solve using Newton
    # First do a Stokes solve for initial guess
    w_stokes = fem.Function(W)
    (u_s, p_s) = ufl.split(w_stokes)
    (v_s, q_s) = ufl.TestFunctions(W)
    
    # Stokes residual (linear)
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
    solver_stokes.max_it = 5
    
    ksp_stokes = solver_stokes.krylov_solver
    ksp_stokes.setType(PETSc.KSP.Type.GMRES)
    pc_stokes = ksp_stokes.getPC()
    pc_stokes.setType(PETSc.PC.Type.LU)
    
    try:
        n_stokes, conv_stokes = solver_stokes.solve(w_stokes)
        w_stokes.x.scatter_forward()
        # Copy Stokes solution as initial guess
        w.x.array[:] = w_stokes.x.array[:]
    except:
        w.x.array[:] = 0.0
    
    # Now solve full Navier-Stokes
    problem_ns = petsc.NonlinearProblem(F, w, bcs=bcs)
    solver_ns = nls.petsc.NewtonSolver(comm, problem_ns)
    solver_ns.convergence_criterion = "incremental"
    solver_ns.rtol = 1e-10
    solver_ns.atol = 1e-12
    solver_ns.max_it = 50
    solver_ns.relaxation_parameter = 1.0
    
    ksp_ns = solver_ns.krylov_solver
    ksp_ns.setType(PETSc.KSP.Type.GMRES)
    ksp_ns.setTolerances(rtol=1e-8)
    pc_ns = ksp_ns.getPC()
    pc_ns.setType(PETSc.PC.Type.LU)
    
    newton_iters = 0
    try:
        n_ns, converged_ns = solver_ns.solve(w)
        w.x.scatter_forward()
        newton_iters = n_ns
    except Exception as e:
        # Try with relaxation
        solver_ns.relaxation_parameter = 0.5
        solver_ns.max_it = 100
        w.x.array[:] = w_stokes.x.array[:] if conv_stokes else 0.0
        try:
            n_ns, converged_ns = solver_ns.solve(w)
            w.x.scatter_forward()
            newton_iters = n_ns
        except:
            pass
    
    # 7. Extract velocity on 50x50 grid
    nx_out, ny_out = 50, 50
    xs = np.linspace(0.0, 1.0, nx_out)
    ys = np.linspace(0.0, 1.0, ny_out)
    XX, YY = np.meshgrid(xs, ys, indexing='ij')
    points_2d = np.column_stack([XX.ravel(), YY.ravel()])
    points_3d = np.zeros((points_2d.shape[0], 3))
    points_3d[:, :2] = points_2d
    
    # Get velocity sub-function
    u_sol = w.sub(0).collapse()
    
    # Point evaluation
    bb_tree = geometry.bb_tree(domain, tdim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, points_3d)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points_3d)
    
    vel_mag = np.full(points_3d.shape[0], np.nan)
    
    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    for i in range(points_3d.shape[0]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(points_3d[i])
            cells_on_proc.append(links[0])
            eval_map.append(i)
    
    if len(points_on_proc) > 0:
        pts_arr = np.array(points_on_proc)
        cells_arr = np.array(cells_on_proc, dtype=np.int32)
        vals = u_sol.eval(pts_arr, cells_arr)
        # vals shape: (n_points, gdim)
        for idx, global_idx in enumerate(eval_map):
            ux = vals[idx, 0]
            uy = vals[idx, 1] if vals.shape[1] > 1 else 0.0
            vel_mag[global_idx] = np.sqrt(ux**2 + uy**2)
    
    # Replace any NaN with nearest valid value or 0
    nan_mask = np.isnan(vel_mag)
    if np.any(nan_mask):
        vel_mag[nan_mask] = 0.0
    
    u_grid = vel_mag.reshape((nx_out, ny_out))
    
    return {
        "u": u_grid,
        "solver_info": {
            "mesh_resolution": N,
            "element_degree": degree_u,
            "ksp_type": "gmres",
            "pc_type": "lu",
            "rtol": 1e-8,
            "nonlinear_iterations": [int(newton_iters)],
        }
    }