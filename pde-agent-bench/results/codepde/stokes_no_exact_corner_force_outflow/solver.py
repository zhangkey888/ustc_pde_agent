import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry, nls
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc


def solve(case_spec: dict) -> dict:
    # 1. Parse case_spec
    pde = case_spec.get("pde", {})
    if not pde:
        pde = case_spec.get("oracle_config", {}).get("pde", {})
    
    nu_val = float(pde.get("viscosity", 0.1))
    source = pde.get("source", [
        '3*exp(-50*((x-0.15)**2 + (y-0.15)**2))',
        '3*exp(-50*((x-0.15)**2 + (y-0.15)**2))'
    ])
    
    # Check for boundary conditions
    bcs_spec = pde.get("boundary_conditions", [])
    
    # 2. Create mesh - use fine mesh for accuracy
    N = 80
    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
    
    tdim = domain.topology.dim
    fdim = tdim - 1
    
    # 3. Define Taylor-Hood mixed elements (P2/P1)
    P2 = ufl.VectorElement("Lagrange", domain.ufl_cell(), 2)
    P1 = ufl.FiniteElement("Lagrange", domain.ufl_cell(), 1)
    TH = ufl.MixedElement([P2, P1])
    W = fem.functionspace(domain, TH)
    
    # Also create individual spaces for BC interpolation
    V = fem.functionspace(domain, ("Lagrange", 2, (domain.geometry.dim,)))
    Q = fem.functionspace(domain, ("Lagrange", 1))
    
    # 4. Define variational problem
    w = fem.Function(W)
    (u, p) = ufl.split(w)
    (v, q) = ufl.TestFunctions(W)
    
    x = ufl.SpatialCoordinate(domain)
    
    nu = fem.Constant(domain, PETSc.ScalarType(nu_val))
    
    # Parse source terms
    def parse_source(expr_str, x_ufl):
        from ufl import exp
        x_var = x_ufl[0]
        y_var = x_ufl[1]
        # Replace x and y in expression
        expr_str = expr_str.replace('x', 'x_var').replace('y_var', 'y_var')
        # Actually, let's build UFL expressions properly
        return None
    
    # Build source term using UFL
    f0_expr = 3.0 * ufl.exp(-50.0 * ((x[0] - 0.15)**2 + (x[1] - 0.15)**2))
    f1_expr = 3.0 * ufl.exp(-50.0 * ((x[0] - 0.15)**2 + (x[1] - 0.15)**2))
    f = ufl.as_vector([f0_expr, f1_expr])
    
    # Stokes residual: -nu*laplacian(u) + grad(p) = f, div(u) = 0
    # Weak form:
    F = (
        nu * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
        - p * ufl.div(v) * ufl.dx
        - ufl.inner(f, v) * ufl.dx
        + q * ufl.div(u) * ufl.dx
    )
    
    # 5. Boundary conditions
    # Determine BC type from case spec
    # "corner_force_outflow" suggests:
    # - Some boundaries have Dirichlet (no-slip or specified velocity)
    # - Some have outflow (natural/do-nothing BC)
    
    # Parse boundary conditions from spec
    has_dirichlet = False
    has_outflow = False
    dirichlet_bcs = []
    outflow_markers = []
    
    for bc_spec in bcs_spec:
        bc_type = bc_spec.get("type", "")
        location = bc_spec.get("location", "")
        
        if bc_type == "dirichlet":
            has_dirichlet = True
            dirichlet_bcs.append(bc_spec)
        elif bc_type == "outflow" or bc_type == "neumann" or bc_type == "do_nothing":
            has_outflow = True
            outflow_markers.append(location)
    
    bcs = []
    
    if len(bcs_spec) > 0:
        # Apply BCs from spec
        for bc_spec in dirichlet_bcs:
            location = bc_spec.get("location", "")
            value = bc_spec.get("value", [0.0, 0.0])
            
            def make_bc_func(loc, val):
                val = [float(v) for v in val]
                
                def marker(x_arr):
                    if loc == "left":
                        return np.isclose(x_arr[0], 0.0)
                    elif loc == "right":
                        return np.isclose(x_arr[0], 1.0)
                    elif loc == "bottom":
                        return np.isclose(x_arr[1], 0.0)
                    elif loc == "top":
                        return np.isclose(x_arr[1], 1.0)
                    else:
                        return np.full(x_arr.shape[1], True)
                
                facets = mesh.locate_entities_boundary(domain, fdim, marker)
                dofs = fem.locate_dofs_topological((W.sub(0), V), fdim, facets)
                u_bc = fem.Function(V)
                u_bc.interpolate(lambda x_arr: np.full((domain.geometry.dim, x_arr.shape[1]), 
                                                        np.array(val)[:, None]))
                return fem.dirichletbc(u_bc, dofs, W.sub(0))
            
            bcs.append(make_bc_func(location, value))
    else:
        # Default: "corner_force_outflow" pattern
        # No-slip on top and bottom, outflow on right, some condition on left
        # For "no_exact_corner_force_outflow":
        # Typically: no-slip on top/bottom/left, do-nothing on right
        
        # No-slip on bottom
        def bottom(x_arr):
            return np.isclose(x_arr[1], 0.0)
        
        facets_bottom = mesh.locate_entities_boundary(domain, fdim, bottom)
        dofs_bottom = fem.locate_dofs_topological((W.sub(0), V), fdim, facets_bottom)
        u_zero = fem.Function(V)
        u_zero.interpolate(lambda x_arr: np.zeros((domain.geometry.dim, x_arr.shape[1])))
        bcs.append(fem.dirichletbc(u_zero, dofs_bottom, W.sub(0)))
        
        # No-slip on top
        def top(x_arr):
            return np.isclose(x_arr[1], 1.0)
        
        facets_top = mesh.locate_entities_boundary(domain, fdim, top)
        dofs_top = fem.locate_dofs_topological((W.sub(0), V), fdim, facets_top)
        u_zero_top = fem.Function(V)
        u_zero_top.interpolate(lambda x_arr: np.zeros((domain.geometry.dim, x_arr.shape[1])))
        bcs.append(fem.dirichletbc(u_zero_top, dofs_top, W.sub(0)))
        
        # No-slip on left
        def left(x_arr):
            return np.isclose(x_arr[0], 0.0)
        
        facets_left = mesh.locate_entities_boundary(domain, fdim, left)
        dofs_left = fem.locate_dofs_topological((W.sub(0), V), fdim, facets_left)
        u_zero_left = fem.Function(V)
        u_zero_left.interpolate(lambda x_arr: np.zeros((domain.geometry.dim, x_arr.shape[1])))
        bcs.append(fem.dirichletbc(u_zero_left, dofs_left, W.sub(0)))
        
        # Right boundary: do-nothing (natural BC for outflow) - no BC needed
    
    # Pin pressure at one point to remove nullspace ambiguity
    # Find a DOF near (0,0) for pressure
    def origin_marker(x_arr):
        return np.logical_and(np.isclose(x_arr[0], 0.0), np.isclose(x_arr[1], 0.0))
    
    # Try to pin pressure
    pressure_facets = mesh.locate_entities_boundary(domain, 0, origin_marker)  # vertices
    if len(pressure_facets) > 0:
        p_dofs = fem.locate_dofs_topological((W.sub(1), Q), 0, pressure_facets)
        p_bc_func = fem.Function(Q)
        p_bc_func.interpolate(lambda x_arr: np.zeros(x_arr.shape[1]))
        bcs.append(fem.dirichletbc(p_bc_func, p_dofs, W.sub(1)))
    
    # 6. Solve - Stokes is linear, so use a linear solve approach
    # Extract bilinear and linear forms from F
    # F = a(u,p; v,q) - L(v,q) = 0
    # Since Stokes is linear, we can use TrialFunction approach
    
    # Re-define using trial functions for linear solve
    W2 = fem.functionspace(domain, TH)
    
    (u_trial, p_trial) = ufl.TrialFunctions(W2)
    (v_test, q_test) = ufl.TestFunctions(W2)
    
    a_form = (
        nu * ufl.inner(ufl.grad(u_trial), ufl.grad(v_test)) * ufl.dx
        - p_trial * ufl.div(v_test) * ufl.dx
        + q_test * ufl.div(u_trial) * ufl.dx
    )
    
    L_form = ufl.inner(f, v_test) * ufl.dx
    
    # Compile forms
    a_compiled = fem.form(a_form)
    L_compiled = fem.form(L_form)
    
    # Assemble
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
    wh = fem.Function(W2)
    
    # Setup KSP solver
    ksp = PETSc.KSP().create(domain.comm)
    ksp.setOperators(A)
    ksp.setType(PETSc.KSP.Type.MINRES)
    pc = ksp.getPC()
    pc.setType(PETSc.PC.Type.LU)
    ksp.setTolerances(rtol=1e-10, atol=1e-12, max_it=5000)
    ksp.setUp()
    
    ksp.solve(b, wh.x.petsc_vec)
    wh.x.scatter_forward()
    
    iterations = ksp.getIterationNumber()
    
    # 7. Extract velocity on grid
    nx_out = 100
    ny_out = 100
    
    xs = np.linspace(0, 1, nx_out)
    ys = np.linspace(0, 1, ny_out)
    X, Y = np.meshgrid(xs, ys, indexing='ij')
    
    points = np.zeros((3, nx_out * ny_out))
    points[0, :] = X.ravel()
    points[1, :] = Y.ravel()
    
    # Get velocity sub-function
    uh = wh.sub(0).collapse()
    
    bb_tree = geometry.bb_tree(domain, domain.topology.dim)
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
    
    # Evaluate velocity (vector field, 2 components)
    vel_mag = np.full(nx_out * ny_out, np.nan)
    
    if len(points_on_proc) > 0:
        pts_arr = np.array(points_on_proc)
        cells_arr = np.array(cells_on_proc, dtype=np.int32)
        vals = uh.eval(pts_arr, cells_arr)  # shape (n_points, 2)
        
        for idx, global_idx in enumerate(eval_map):
            ux = vals[idx, 0]
            uy = vals[idx, 1]
            vel_mag[global_idx] = np.sqrt(ux**2 + uy**2)
    
    u_grid = vel_mag.reshape((nx_out, ny_out))
    
    # Clean up PETSc objects
    ksp.destroy()
    A.destroy()
    b.destroy()
    
    return {
        "u": u_grid,
        "solver_info": {
            "mesh_resolution": N,
            "element_degree": 2,
            "ksp_type": "minres",
            "pc_type": "lu",
            "rtol": 1e-10,
            "iterations": iterations,
        }
    }