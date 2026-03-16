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
    
    nu_val = float(pde.get("viscosity", 0.3))
    source = pde.get("source_term", ["0.0", "0.0"])
    bcs_spec = pde.get("boundary_conditions", [])
    
    # 2. Create mesh
    N = 80  # mesh resolution
    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
    tdim = domain.topology.dim
    fdim = tdim - 1
    
    # 3. Taylor-Hood mixed elements: P2 velocity, P1 pressure
    P2 = fem.functionspace(domain, ("Lagrange", 2, (domain.geometry.dim,)))
    P1 = fem.functionspace(domain, ("Lagrange", 1))
    
    # Mixed function space via ufl.MixedElement approach
    # In dolfinx 0.10.0, we use separate spaces and block assembly
    # Or we can use the mixed element approach
    
    # Use block (monolithic) approach with mixed element
    el_v = basix.ufl.element("Lagrange", domain.topology.cell_name(), 2, shape=(domain.geometry.dim,))
    el_p = basix.ufl.element("Lagrange", domain.topology.cell_name(), 1)
    el_mixed = basix.ufl.mixed_element([el_v, el_p])
    
    W = fem.functionspace(domain, el_mixed)
    
    # 4. Define variational problem
    (u, p) = ufl.TrialFunctions(W)
    (v, q) = ufl.TestFunctions(W)
    
    nu = fem.Constant(domain, PETSc.ScalarType(nu_val))
    
    # Parse source term
    f_expr = [0.0, 0.0]
    if source:
        for i, s in enumerate(source):
            try:
                f_expr[i] = float(s)
            except:
                f_expr[i] = 0.0
    
    f = fem.Constant(domain, PETSc.ScalarType(np.array(f_expr, dtype=np.float64)))
    
    # Stokes weak form:
    # nu * inner(grad(u), grad(v)) * dx - inner(p, div(v)) * dx + inner(div(u), q) * dx = inner(f, v) * dx
    a = (nu * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
         - ufl.inner(p, ufl.div(v)) * ufl.dx
         + ufl.inner(ufl.div(u), q) * ufl.dx)
    
    L = ufl.inner(f, v) * ufl.dx
    
    # 5. Boundary conditions
    # Parse BCs from case_spec
    bcs = []
    
    # Get sub spaces
    V_sub, V_sub_map = W.sub(0).collapse()
    
    # Parse boundary conditions
    def parse_bc_value(val_str):
        """Parse a boundary condition value string."""
        if isinstance(val_str, (int, float)):
            return float(val_str)
        if isinstance(val_str, str):
            try:
                return float(val_str)
            except:
                return 0.0
        return 0.0
    
    def make_bc_function(bc_info, V_collapsed):
        """Create a function for the boundary condition."""
        bc_type = bc_info.get("type", "dirichlet")
        value = bc_info.get("value", None)
        
        u_bc = fem.Function(V_collapsed)
        
        if value is not None:
            if isinstance(value, list):
                vals = [parse_bc_value(v) for v in value]
                u_bc.interpolate(lambda x: np.array([np.full_like(x[0], vals[i]) for i in range(len(vals))]))
            elif isinstance(value, str):
                # Try to parse expressions involving x, y
                val = parse_bc_value(value)
                u_bc.interpolate(lambda x: np.full_like(x[0], val))
            else:
                val = float(value)
                u_bc.interpolate(lambda x: np.full_like(x[0], val))
        else:
            u_bc.interpolate(lambda x: np.zeros((domain.geometry.dim, x.shape[1])))
        
        return u_bc
    
    # For double lid cavity: identify boundary conditions
    # Typically: top and bottom walls move (lids), left and right are no-slip
    # Parse from case_spec
    
    if bcs_spec:
        for bc_info in bcs_spec:
            location = bc_info.get("location", "")
            bc_type = bc_info.get("type", "dirichlet")
            value = bc_info.get("value", None)
            
            if bc_type.lower() != "dirichlet":
                continue
            
            # Parse velocity values
            if isinstance(value, list):
                vals = [parse_bc_value(v) for v in value]
            elif isinstance(value, str):
                # Could be expression
                vals = [parse_bc_value(value), 0.0]
            else:
                vals = [0.0, 0.0]
            
            u_bc = fem.Function(V_sub)
            vx, vy = vals[0], vals[1]
            u_bc.interpolate(lambda x, vx=vx, vy=vy: np.vstack([
                np.full_like(x[0], vx),
                np.full_like(x[0], vy)
            ]))
            
            # Determine boundary facets
            if "top" in location.lower():
                facets = mesh.locate_entities_boundary(domain, fdim, lambda x: np.isclose(x[1], 1.0))
            elif "bottom" in location.lower():
                facets = mesh.locate_entities_boundary(domain, fdim, lambda x: np.isclose(x[1], 0.0))
            elif "left" in location.lower():
                facets = mesh.locate_entities_boundary(domain, fdim, lambda x: np.isclose(x[0], 0.0))
            elif "right" in location.lower():
                facets = mesh.locate_entities_boundary(domain, fdim, lambda x: np.isclose(x[0], 1.0))
            elif "all" in location.lower() or "entire" in location.lower() or "boundary" in location.lower():
                facets = mesh.locate_entities_boundary(domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool))
            else:
                facets = mesh.locate_entities_boundary(domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool))
            
            dofs = fem.locate_dofs_topological((W.sub(0), V_sub), fdim, facets)
            bc = fem.dirichletbc(u_bc, dofs, W.sub(0))
            bcs.append(bc)
    else:
        # Default double lid cavity: top lid moves right, bottom lid moves left (or similar)
        # Top: u = (1, 0)
        u_top = fem.Function(V_sub)
        u_top.interpolate(lambda x: np.vstack([np.ones_like(x[0]), np.zeros_like(x[0])]))
        top_facets = mesh.locate_entities_boundary(domain, fdim, lambda x: np.isclose(x[1], 1.0))
        dofs_top = fem.locate_dofs_topological((W.sub(0), V_sub), fdim, top_facets)
        bc_top = fem.dirichletbc(u_top, dofs_top, W.sub(0))
        bcs.append(bc_top)
        
        # Bottom: u = (-1, 0) for double lid
        u_bottom = fem.Function(V_sub)
        u_bottom.interpolate(lambda x: np.vstack([-np.ones_like(x[0]), np.zeros_like(x[0])]))
        bottom_facets = mesh.locate_entities_boundary(domain, fdim, lambda x: np.isclose(x[1], 0.0))
        dofs_bottom = fem.locate_dofs_topological((W.sub(0), V_sub), fdim, bottom_facets)
        bc_bottom = fem.dirichletbc(u_bottom, dofs_bottom, W.sub(0))
        bcs.append(bc_bottom)
        
        # Left and right: no-slip u = (0, 0)
        u_noslip = fem.Function(V_sub)
        u_noslip.interpolate(lambda x: np.vstack([np.zeros_like(x[0]), np.zeros_like(x[0])]))
        
        left_facets = mesh.locate_entities_boundary(domain, fdim, lambda x: np.isclose(x[0], 0.0))
        dofs_left = fem.locate_dofs_topological((W.sub(0), V_sub), fdim, left_facets)
        bc_left = fem.dirichletbc(u_noslip, dofs_left, W.sub(0))
        bcs.append(bc_left)
        
        right_facets = mesh.locate_entities_boundary(domain, fdim, lambda x: np.isclose(x[0], 1.0))
        u_noslip2 = fem.Function(V_sub)
        u_noslip2.interpolate(lambda x: np.vstack([np.zeros_like(x[0]), np.zeros_like(x[0])]))
        dofs_right = fem.locate_dofs_topological((W.sub(0), V_sub), fdim, right_facets)
        bc_right = fem.dirichletbc(u_noslip2, dofs_right, W.sub(0))
        bcs.append(bc_right)
    
    # 6. Solve using LinearProblem with appropriate solver for saddle-point system
    # Pin pressure at one point to remove nullspace
    # Find a DOF near origin for pressure
    Q_sub, Q_sub_map = W.sub(1).collapse()
    
    # Pin pressure at a point
    p_zero = fem.Function(Q_sub)
    p_zero.interpolate(lambda x: np.zeros_like(x[0]))
    
    # Find pressure DOF at (0, 0)
    pressure_dofs = fem.locate_dofs_geometrical((W.sub(1), Q_sub), lambda x: np.isclose(x[0], 0.0) & np.isclose(x[1], 0.0))
    if len(pressure_dofs[0]) > 0:
        bc_p = fem.dirichletbc(p_zero, pressure_dofs, W.sub(1))
        bcs.append(bc_p)
    
    # Assemble and solve
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
    
    # Create solution function
    wh = fem.Function(W)
    
    # Setup KSP solver
    ksp = PETSc.KSP().create(domain.comm)
    ksp.setOperators(A)
    ksp.setType(PETSc.KSP.Type.GMRES)
    ksp.setTolerances(rtol=1e-10, atol=1e-12, max_it=5000)
    
    pc = ksp.getPC()
    pc.setType(PETSc.PC.Type.LU)
    pc.setFactorSolverType("mumps")
    
    ksp.solve(b, wh.x.petsc_vec)
    wh.x.scatter_forward()
    
    iterations = ksp.getIterationNumber()
    
    # 7. Extract velocity on 100x100 grid
    nx_out, ny_out = 100, 100
    xs = np.linspace(0, 1, nx_out)
    ys = np.linspace(0, 1, ny_out)
    XX, YY = np.meshgrid(xs, ys, indexing='ij')
    
    points = np.zeros((3, nx_out * ny_out))
    points[0, :] = XX.ravel()
    points[1, :] = YY.ravel()
    
    # Get velocity subfunction
    uh = wh.sub(0).collapse()
    
    # Probe points
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
    
    # Evaluate velocity (2 components)
    u_values = np.full((points.shape[1], 2), np.nan)
    if len(points_on_proc) > 0:
        pts_arr = np.array(points_on_proc)
        cells_arr = np.array(cells_on_proc, dtype=np.int32)
        vals = uh.eval(pts_arr, cells_arr)
        for idx_local, idx_global in enumerate(eval_map):
            u_values[idx_global, :] = vals[idx_local, :]
    
    # Compute velocity magnitude
    vel_mag = np.sqrt(u_values[:, 0]**2 + u_values[:, 1]**2)
    u_grid = vel_mag.reshape(nx_out, ny_out)
    
    # Handle any NaN values (boundary points that might not be found)
    if np.any(np.isnan(u_grid)):
        from scipy.ndimage import generic_filter
        # Simple fill: replace NaN with nearest non-NaN
        mask = np.isnan(u_grid)
        if np.sum(~mask) > 0:
            from scipy.interpolate import NearestNDInterpolator
            valid_coords = np.array(np.where(~mask)).T
            valid_vals = u_grid[~mask]
            interp = NearestNDInterpolator(valid_coords, valid_vals)
            nan_coords = np.array(np.where(mask)).T
            if len(nan_coords) > 0:
                u_grid[mask] = interp(nan_coords)
    
    ksp.destroy()
    A.destroy()
    b.destroy()
    
    return {
        "u": u_grid,
        "solver_info": {
            "mesh_resolution": N,
            "element_degree": 2,
            "ksp_type": "gmres",
            "pc_type": "lu",
            "rtol": 1e-10,
            "iterations": iterations if iterations > 0 else 1,
        }
    }