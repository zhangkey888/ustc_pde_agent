import numpy as np
from dolfinx import mesh, fem, default_scalar_type, geometry
from dolfinx.fem import petsc
from dolfinx import nls
from mpi4py import MPI
import ufl
from petsc4py import PETSc
import basix


def solve(case_spec: dict) -> dict:
    # 1. Parse case_spec
    pde = case_spec.get("pde", {})
    if not pde:
        pde = case_spec.get("oracle_config", {}).get("pde", {})
    
    nu_val = float(pde.get("viscosity", 0.5))
    source = pde.get("source_term", ["0.0", "0.0"])
    bcs_spec = pde.get("boundary_conditions", [])
    
    # 2. Create mesh - use fine mesh for accuracy
    N = 80
    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
    
    tdim = domain.topology.dim
    fdim = tdim - 1
    
    # 3. Mixed function space (Taylor-Hood P2/P1)
    degree_u = 2
    degree_p = 1
    
    V_el = basix.ufl.element("Lagrange", domain.basix_cell(), degree_u, shape=(domain.geometry.dim,))
    Q_el = basix.ufl.element("Lagrange", domain.basix_cell(), degree_p)
    ME = basix.ufl.mixed_element([V_el, Q_el])
    W = fem.functionspace(domain, ME)
    
    # Also create sub-spaces for BC interpolation
    V, V_to_W = W.sub(0).collapse()
    Q, Q_to_W = W.sub(1).collapse()
    
    # 4. Define trial/test functions and variational form
    # For Stokes (linear), use TrialFunction/TestFunction
    (u, p) = ufl.TrialFunctions(W)
    (v, q) = ufl.TestFunctions(W)
    
    x = ufl.SpatialCoordinate(domain)
    
    # Source term
    f_x_str = source[0] if len(source) > 0 else "0.0"
    f_y_str = source[1] if len(source) > 1 else "0.0"
    
    # Parse source terms
    def parse_expr(s):
        s = s.strip()
        try:
            val = float(s)
            return val
        except:
            pass
        # Try to parse symbolic expressions
        expr_str = s.replace("sin", "ufl.sin").replace("cos", "ufl.cos").replace("pi", "ufl.pi")
        expr_str = expr_str.replace("x", "x[0]").replace("y", "x[1]")
        try:
            return eval(expr_str)
        except:
            return 0.0
    
    f_x = parse_expr(f_x_str)
    f_y = parse_expr(f_y_str)
    
    if isinstance(f_x, (int, float)) and isinstance(f_y, (int, float)):
        f = fem.Constant(domain, np.array([float(f_x), float(f_y)], dtype=PETSc.ScalarType))
    else:
        f = ufl.as_vector([f_x, f_y])
    
    nu = fem.Constant(domain, PETSc.ScalarType(nu_val))
    
    # Bilinear form: Stokes
    a_form = (
        nu * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
        - p * ufl.div(v) * ufl.dx
        - q * ufl.div(u) * ufl.dx
    )
    
    L_form = ufl.inner(f, v) * ufl.dx
    
    # 5. Boundary conditions
    # Parse boundary conditions from case_spec
    bcs = []
    
    # Helper to parse BC value expressions
    def parse_bc_value(val_str, component=None):
        """Parse a BC value string into a callable for interpolation."""
        val_str = str(val_str).strip()
        try:
            val = float(val_str)
            return lambda x_arr: np.full(x_arr.shape[1], val)
        except:
            pass
        return None
    
    def make_vector_bc_callable(expr_list):
        """Create a callable that returns (2, N) array from list of expression strings."""
        def bc_func(x_arr):
            result = np.zeros((2, x_arr.shape[1]))
            for i, expr_str in enumerate(expr_list):
                expr_str = str(expr_str).strip()
                try:
                    val = float(expr_str)
                    result[i, :] = val
                    continue
                except:
                    pass
                # Parse symbolic expression
                safe_expr = expr_str
                safe_expr = safe_expr.replace("^", "**")
                safe_expr = safe_expr.replace("sin", "np.sin")
                safe_expr = safe_expr.replace("cos", "np.cos")
                safe_expr = safe_expr.replace("exp", "np.exp")
                safe_expr = safe_expr.replace("pi", "np.pi")
                safe_expr = safe_expr.replace("sqrt", "np.sqrt")
                # Replace x and y with array references
                # Be careful with order of replacement
                import re
                safe_expr = re.sub(r'\by\b', 'x_arr[1]', safe_expr)
                safe_expr = re.sub(r'\bx\b', 'x_arr[0]', safe_expr)
                try:
                    result[i, :] = eval(safe_expr)
                except Exception as e:
                    result[i, :] = 0.0
            return result
        return bc_func
    
    # Process boundary conditions
    domain.topology.create_connectivity(fdim, tdim)
    
    for bc_spec in bcs_spec:
        bc_type = bc_spec.get("type", "dirichlet")
        location = bc_spec.get("location", "")
        value = bc_spec.get("value", None)
        component = bc_spec.get("component", None)
        
        if bc_type != "dirichlet":
            continue
        
        # Determine boundary marker function
        if location == "left":
            marker = lambda x: np.isclose(x[0], 0.0)
        elif location == "right":
            marker = lambda x: np.isclose(x[0], 1.0)
        elif location == "bottom":
            marker = lambda x: np.isclose(x[1], 0.0)
        elif location == "top":
            marker = lambda x: np.isclose(x[1], 1.0)
        elif location == "all":
            marker = lambda x: (np.isclose(x[0], 0.0) | np.isclose(x[0], 1.0) |
                               np.isclose(x[1], 0.0) | np.isclose(x[1], 1.0))
        else:
            marker = lambda x: (np.isclose(x[0], 0.0) | np.isclose(x[0], 1.0) |
                               np.isclose(x[1], 0.0) | np.isclose(x[1], 1.0))
        
        facets = mesh.locate_entities_boundary(domain, fdim, marker)
        
        if component is not None:
            # Scalar BC on a component of velocity
            comp_idx = int(component)
            W_sub_comp = W.sub(0).sub(comp_idx)
            V_comp, V_comp_to_W = W_sub_comp.collapse()
            
            u_bc_func = fem.Function(V_comp)
            val_str = str(value) if not isinstance(value, list) else str(value[0])
            bc_callable = parse_bc_value(val_str)
            if bc_callable is not None:
                u_bc_func.interpolate(lambda x_arr: bc_callable(x_arr))
            else:
                u_bc_func.interpolate(lambda x_arr: np.zeros(x_arr.shape[1]))
            
            dofs = fem.locate_dofs_topological((W_sub_comp, V_comp), fdim, facets)
            bc = fem.dirichletbc(u_bc_func, dofs, W_sub_comp)
            bcs.append(bc)
        else:
            # Vector BC on velocity
            if isinstance(value, list) and len(value) >= 2:
                bc_callable = make_vector_bc_callable(value)
            elif isinstance(value, str):
                bc_callable = make_vector_bc_callable([value, "0.0"])
            else:
                bc_callable = lambda x_arr: np.zeros((2, x_arr.shape[1]))
            
            u_bc_func = fem.Function(V)
            u_bc_func.interpolate(bc_callable)
            
            dofs = fem.locate_dofs_topological((W.sub(0), V), fdim, facets)
            bc = fem.dirichletbc(u_bc_func, dofs, W.sub(0))
            bcs.append(bc)
    
    # If no BCs were specified, apply zero velocity on all boundaries
    if len(bcs) == 0:
        all_facets = mesh.locate_entities_boundary(
            domain, fdim,
            lambda x: np.ones(x.shape[1], dtype=bool)
        )
        u_bc_func = fem.Function(V)
        u_bc_func.interpolate(lambda x_arr: np.zeros((2, x_arr.shape[1])))
        dofs = fem.locate_dofs_topological((W.sub(0), V), fdim, all_facets)
        bc = fem.dirichletbc(u_bc_func, dofs, W.sub(0))
        bcs.append(bc)
    
    # 6. Solve using PETSc KSP (manual assembly for better control)
    a_compiled = fem.form(a_form)
    L_compiled = fem.form(L_form)
    
    A = petsc.assemble_matrix(a_compiled, bcs=bcs)
    A.assemble()
    
    b = petsc.create_vector(L_compiled)
    with b.localForm() as loc:
        loc.set(0)
    petsc.assemble_vector(b, L_compiled)
    petsc.apply_lifting(b, [a_compiled], bcs=[bcs])
    b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    petsc.set_bc(b, bcs)
    
    # Create solution function
    wh = fem.Function(W)
    
    # Set up KSP solver
    ksp = PETSc.KSP().create(domain.comm)
    ksp.setOperators(A)
    
    # Use MINRES with appropriate preconditioner for saddle-point system
    ksp_type = "minres"
    pc_type = "lu"
    rtol = 1e-10
    
    ksp.setType(ksp_type)
    ksp.getPC().setType(pc_type)
    if pc_type == "lu":
        ksp.getPC().setFactorSolverType("mumps")
    ksp.setTolerances(rtol=rtol, atol=1e-12, max_it=5000)
    
    # Handle pressure nullspace if needed (pure Dirichlet velocity BCs)
    # Check if we need nullspace
    # For Stokes with all Dirichlet velocity BCs, pressure is determined up to a constant
    # We'll create a nullspace for the pressure block
    
    # Create nullspace vector: zero velocity, constant pressure
    null_vec = A.createVecLeft()
    null_vec.set(0.0)
    
    # Get pressure DOFs in the mixed space
    p_dofs = W.sub(1).dofmap.list.array
    # Map to mixed space DOFs
    offset = W.sub(1).dofmap.index_map.size_local
    bs = W.sub(1).dofmap.index_map_bs
    
    # Actually, let's use the simpler approach with index maps
    # Get the offset for pressure block
    map0 = W.sub(0).dofmap.index_map
    offset_p = map0.size_local * W.sub(0).dofmap.index_map_bs
    
    # Set pressure DOFs to 1 in nullspace vector
    map1 = W.sub(1).dofmap.index_map
    p_size = map1.size_local
    
    # Get the actual DOF indices for pressure in the mixed space
    # In dolfinx mixed spaces, DOFs are interleaved or blocked
    # We need to figure out the layout
    
    # Use a different approach: create a Function and set pressure to 1
    null_func = fem.Function(W)
    null_func.x.array[:] = 0.0
    
    # Get pressure sub-space DOFs
    p_sub_dofs = np.array(W.sub(1).dofmap.list.array, dtype=np.int32)
    # These are local to the sub-dofmap; we need the mixed space indices
    # Actually in dolfinx 0.10, W.sub(1).dofmap gives DOFs in the parent space
    
    # Simpler: use collapse mapping
    _, p_to_W_map = W.sub(1).collapse()
    null_func.x.array[p_to_W_map] = 1.0
    
    # Normalize
    null_func.x.petsc_vec.normalize()
    
    nsp = PETSc.NullSpace().create(vectors=[null_func.x.petsc_vec], comm=domain.comm)
    A.setNullSpace(nsp)
    nsp.remove(b)
    
    ksp.setUp()
    ksp.solve(b, wh.x.petsc_vec)
    wh.x.scatter_forward()
    
    iterations = ksp.getIterationNumber()
    
    # 7. Extract velocity on uniform grid
    nx_out = 100
    ny_out = 100
    xs = np.linspace(0, 1, nx_out)
    ys = np.linspace(0, 1, ny_out)
    XX, YY = np.meshgrid(xs, ys, indexing='ij')
    points = np.zeros((3, nx_out * ny_out))
    points[0, :] = XX.ravel()
    points[1, :] = YY.ravel()
    
    # Extract velocity sub-function
    uh = wh.sub(0).collapse()
    
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
    
    # Velocity magnitude
    vel_mag = np.full(nx_out * ny_out, np.nan)
    if len(points_on_proc) > 0:
        pts_arr = np.array(points_on_proc)
        cells_arr = np.array(cells_on_proc, dtype=np.int32)
        vals = uh.eval(pts_arr, cells_arr)
        # vals shape: (n_points, 2) for 2D velocity
        mag = np.sqrt(vals[:, 0]**2 + vals[:, 1]**2)
        for idx, global_idx in enumerate(eval_map):
            vel_mag[global_idx] = mag[idx]
    
    u_grid = vel_mag.reshape((nx_out, ny_out))
    
    # Clean up PETSc objects
    ksp.destroy()
    A.destroy()
    b.destroy()
    
    return {
        "u": u_grid,
        "solver_info": {
            "mesh_resolution": N,
            "element_degree": degree_u,
            "ksp_type": ksp_type,
            "pc_type": pc_type,
            "rtol": rtol,
            "iterations": iterations,
            "pressure_fixing": "nullspace"
        }
    }