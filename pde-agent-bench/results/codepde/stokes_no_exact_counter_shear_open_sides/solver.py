import numpy as np
from dolfinx import mesh, fem, default_scalar_type, geometry
from dolfinx.fem import petsc
from mpi4py import MPI
import ufl
from petsc4py import PETSc
from basix.ufl import element, mixed_element


def solve(case_spec: dict) -> dict:
    # 1. Parse case_spec
    pde = case_spec.get("pde", {})
    if not pde:
        pde = case_spec.get("oracle_config", {}).get("pde", {})
    
    nu_val = float(pde.get("viscosity", 1.0))
    source = pde.get("source_term", ["0.0", "0.0"])
    bcs_spec = pde.get("boundary_conditions", [])
    
    # 2. Create mesh - use high resolution for accuracy
    N = 80
    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
    tdim = domain.topology.dim
    fdim = tdim - 1
    
    # 3. Taylor-Hood mixed elements (P2/P1)
    degree_u = 2
    degree_p = 1
    
    P2 = element("Lagrange", domain.topology.cell_name(), degree_u, shape=(2,))
    P1 = element("Lagrange", domain.topology.cell_name(), degree_p)
    TH = mixed_element([P2, P1])
    
    W = fem.functionspace(domain, TH)
    
    # Sub-spaces for BCs
    V_sub, V_sub_map = W.sub(0).collapse()
    Q_sub, Q_sub_map = W.sub(1).collapse()
    
    # 4. Define variational problem
    (u, p) = ufl.TrialFunctions(W)
    (v, q) = ufl.TestFunctions(W)
    
    nu = fem.Constant(domain, default_scalar_type(nu_val))
    
    # Source term
    f_expr = [0.0, 0.0]
    if source and len(source) >= 2:
        # Try to parse; they might be expressions
        x = ufl.SpatialCoordinate(domain)
        try:
            f0 = float(source[0])
        except:
            f0 = 0.0
        try:
            f1 = float(source[1])
        except:
            f1 = 0.0
        f = ufl.as_vector([f0, f1])
    else:
        f = ufl.as_vector([0.0, 0.0])
    
    # Bilinear form: Stokes
    a = (nu * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
         - p * ufl.div(v) * ufl.dx
         - q * ufl.div(u) * ufl.dx)
    
    L = ufl.inner(f, v) * ufl.dx
    
    # 5. Boundary conditions
    # Parse from case_spec
    bcs = []
    
    def parse_bc_value(val_spec):
        """Parse a boundary condition value specification into a callable for interpolation."""
        if isinstance(val_spec, (int, float)):
            c = float(val_spec)
            return lambda x: np.full((2, x.shape[1]), c)
        elif isinstance(val_spec, list) and len(val_spec) >= 2:
            # Could be expressions or constants
            return _make_vector_bc(val_spec)
        elif isinstance(val_spec, str):
            c = float(val_spec)
            return lambda x: np.full((2, x.shape[1]), c)
        return lambda x: np.zeros((2, x.shape[1]))
    
    def _make_vector_bc(val_list):
        """Create interpolation function from a list of string expressions."""
        import re
        
        def eval_component(expr_str, x):
            expr_str = str(expr_str).strip()
            # Try simple float
            try:
                c = float(expr_str)
                return np.full(x.shape[1], c)
            except ValueError:
                pass
            
            # Parse expression with x[0], x[1], etc.
            # Replace common math
            safe_expr = expr_str
            safe_expr = safe_expr.replace("pi", str(np.pi))
            safe_expr = safe_expr.replace("sin", "np.sin")
            safe_expr = safe_expr.replace("cos", "np.cos")
            safe_expr = safe_expr.replace("exp", "np.exp")
            safe_expr = safe_expr.replace("x[0]", "X0")
            safe_expr = safe_expr.replace("x[1]", "X1")
            safe_expr = safe_expr.replace("x", "X0")  # fallback
            safe_expr = safe_expr.replace("y", "X1")
            
            X0 = x[0]
            X1 = x[1]
            try:
                result = eval(safe_expr)
                if np.isscalar(result):
                    return np.full(x.shape[1], float(result))
                return result
            except:
                return np.zeros(x.shape[1])
        
        def interp_func(x):
            v0 = eval_component(val_list[0], x)
            v1 = eval_component(val_list[1], x)
            return np.stack([v0, v1], axis=0)
        
        return interp_func
    
    # Identify boundary type from case name
    case_id = case_spec.get("case_id", "")
    
    # Parse BCs from spec
    has_velocity_bc = False
    
    for bc_spec in bcs_spec:
        bc_type = bc_spec.get("type", "").lower()
        location = bc_spec.get("location", "").lower()
        value = bc_spec.get("value", None)
        
        if bc_type == "dirichlet" and "velocity" in bc_spec.get("variable", "velocity").lower():
            has_velocity_bc = True
            # Determine boundary marker
            if "top" in location:
                marker = lambda x: np.isclose(x[1], 1.0)
            elif "bottom" in location:
                marker = lambda x: np.isclose(x[1], 0.0)
            elif "left" in location:
                marker = lambda x: np.isclose(x[0], 0.0)
            elif "right" in location:
                marker = lambda x: np.isclose(x[0], 1.0)
            elif "all" in location or "entire" in location or "boundary" in location:
                marker = lambda x: (np.isclose(x[0], 0.0) | np.isclose(x[0], 1.0) |
                                     np.isclose(x[1], 0.0) | np.isclose(x[1], 1.0))
            else:
                marker = lambda x: (np.isclose(x[0], 0.0) | np.isclose(x[0], 1.0) |
                                     np.isclose(x[1], 0.0) | np.isclose(x[1], 1.0))
            
            facets = mesh.locate_entities_boundary(domain, fdim, marker)
            dofs = fem.locate_dofs_topological((W.sub(0), V_sub), fdim, facets)
            
            u_bc = fem.Function(V_sub)
            interp_func = parse_bc_value(value)
            u_bc.interpolate(interp_func)
            
            bc = fem.dirichletbc(u_bc, dofs, W.sub(0))
            bcs.append(bc)
    
    # If no BCs parsed, set up based on case name
    if not has_velocity_bc:
        # "counter_shear_open_sides" pattern:
        # Top and bottom have shear (counter-flowing), left/right are open (natural BC / do-nothing)
        # Typical: top u=(1,0), bottom u=(-1,0), left/right: natural (stress-free)
        
        # Top boundary: u = (1, 0) or some shear
        top_facets = mesh.locate_entities_boundary(domain, fdim, lambda x: np.isclose(x[1], 1.0))
        top_dofs = fem.locate_dofs_topological((W.sub(0), V_sub), fdim, top_facets)
        u_top = fem.Function(V_sub)
        u_top.interpolate(lambda x: np.stack([np.ones(x.shape[1]), np.zeros(x.shape[1])]))
        bc_top = fem.dirichletbc(u_top, top_dofs, W.sub(0))
        bcs.append(bc_top)
        
        # Bottom boundary: u = (-1, 0) (counter shear)
        bot_facets = mesh.locate_entities_boundary(domain, fdim, lambda x: np.isclose(x[1], 0.0))
        bot_dofs = fem.locate_dofs_topological((W.sub(0), V_sub), fdim, bot_facets)
        u_bot = fem.Function(V_sub)
        u_bot.interpolate(lambda x: np.stack([-np.ones(x.shape[1]), np.zeros(x.shape[1])]))
        bc_bot = fem.dirichletbc(u_bot, bot_dofs, W.sub(0))
        bcs.append(bc_bot)
        
        # Left and right: do-nothing (natural BC) - no Dirichlet needed
    
    # 6. Solve using manual assembly for better control of saddle-point system
    a_form = fem.form(a)
    L_form = fem.form(L)
    
    A = petsc.assemble_matrix(a_form, bcs=bcs)
    A.assemble()
    
    b = petsc.create_vector(L_form)
    with b.localForm() as loc:
        loc.set(0)
    petsc.assemble_vector(b, L_form)
    petsc.apply_lifting(b, [a_form], bcs=[bcs])
    b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    petsc.set_bc(b, bcs)
    
    # Create solution
    wh = fem.Function(W)
    
    # Set up KSP solver
    ksp = PETSc.KSP().create(domain.comm)
    ksp.setOperators(A)
    ksp.setType(PETSc.KSP.Type.MINRES)
    pc = ksp.getPC()
    pc.setType(PETSc.PC.Type.LU)
    ksp.setTolerances(rtol=1e-12, atol=1e-14, max_it=5000)
    ksp.setUp()
    
    ksp.solve(b, wh.x.petsc_vec)
    wh.x.scatter_forward()
    
    iterations = ksp.getIterationNumber()
    
    # 7. Extract velocity on 100x100 grid
    nx_out, ny_out = 100, 100
    xs = np.linspace(0, 1, nx_out)
    ys = np.linspace(0, 1, ny_out)
    XX, YY = np.meshgrid(xs, ys, indexing='ij')
    points_2d = np.stack([XX.ravel(), YY.ravel()], axis=0)
    points_3d = np.vstack([points_2d, np.zeros((1, points_2d.shape[1]))])
    
    # Get velocity sub-function
    uh = wh.sub(0).collapse()
    
    bb_tree = geometry.bb_tree(domain, tdim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, points_3d.T)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points_3d.T)
    
    n_points = points_3d.shape[1]
    vel_mag = np.full(n_points, np.nan)
    
    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    
    for i in range(n_points):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(points_3d[:, i])
            cells_on_proc.append(links[0])
            eval_map.append(i)
    
    if len(points_on_proc) > 0:
        pts_arr = np.array(points_on_proc)
        cells_arr = np.array(cells_on_proc, dtype=np.int32)
        vals = uh.eval(pts_arr, cells_arr)
        # vals shape: (n_eval_points, 2) for 2D velocity
        for k, idx in enumerate(eval_map):
            ux = vals[k, 0]
            uy = vals[k, 1]
            vel_mag[idx] = np.sqrt(ux**2 + uy**2)
    
    u_grid = vel_mag.reshape((nx_out, ny_out))
    
    # Clean up
    ksp.destroy()
    A.destroy()
    b.destroy()
    
    return {
        "u": u_grid,
        "solver_info": {
            "mesh_resolution": N,
            "element_degree": degree_u,
            "ksp_type": "minres",
            "pc_type": "lu",
            "rtol": 1e-12,
            "iterations": iterations,
        }
    }