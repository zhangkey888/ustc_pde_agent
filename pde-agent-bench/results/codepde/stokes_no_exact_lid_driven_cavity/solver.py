import numpy as np
from dolfinx import mesh, fem, default_scalar_type, geometry, nls
from dolfinx.fem import petsc
from mpi4py import MPI
import ufl
from petsc4py import PETSc
import basix


def solve(case_spec: dict) -> dict:
    # 1. Parse case_spec
    pde = case_spec.get("pde", {})
    nu_val = float(pde.get("viscosity", 0.2))
    source = pde.get("source_term", ["0.0", "0.0"])
    bcs_spec = pde.get("boundary_conditions", [])
    
    # Mesh parameters
    N = 64
    degree_u = 2
    degree_p = 1
    
    comm = MPI.COMM_WORLD
    
    # 2. Create mesh
    domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
    tdim = domain.topology.dim
    fdim = tdim - 1
    
    # 3. Create Taylor-Hood mixed function spaces
    V_el = basix.ufl.element("Lagrange", domain.basix_cell(), degree_u, shape=(domain.geometry.dim,))
    Q_el = basix.ufl.element("Lagrange", domain.basix_cell(), degree_p)
    ME = basix.ufl.mixed_element([V_el, Q_el])
    W = fem.functionspace(domain, ME)
    
    # Also create individual spaces for BC interpolation
    V, V_to_W = W.sub(0).collapse()
    Q, Q_to_W = W.sub(1).collapse()
    
    # 4. Define variational problem
    # For Stokes: -nu * laplacian(u) + grad(p) = f, div(u) = 0
    # Weak form: nu * inner(grad(u), grad(v)) * dx - p * div(v) * dx - q * div(u) * dx = inner(f, v) * dx
    
    (u, p) = ufl.TrialFunctions(W)
    (v, q) = ufl.TestFunctions(W)
    
    nu = fem.Constant(domain, default_scalar_type(nu_val))
    
    x = ufl.SpatialCoordinate(domain)
    
    # Source term
    f_val = [0.0, 0.0]
    if source is not None and len(source) >= 2:
        try:
            f_val = [float(source[0]), float(source[1])]
        except:
            f_val = [0.0, 0.0]
    
    f = fem.Constant(domain, default_scalar_type((f_val[0], f_val[1])))
    
    a = (
        nu * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
        - p * ufl.div(v) * ufl.dx
        - q * ufl.div(u) * ufl.dx
    )
    L = ufl.inner(f, v) * ufl.dx
    
    # 5. Boundary conditions - Lid-driven cavity
    # Top wall: u = (1, 0)
    # Other walls: u = (0, 0)
    
    bcs = []
    
    # Parse boundary conditions from case_spec
    bc_dict = {}
    for bc_item in bcs_spec:
        loc = bc_item.get("location", "")
        bc_type = bc_item.get("type", "")
        value = bc_item.get("value", None)
        bc_dict[loc] = {"type": bc_type, "value": value}
    
    # If no BCs specified, use default lid-driven cavity
    if not bc_dict:
        bc_dict = {
            "top": {"type": "dirichlet", "value": ["1.0", "0.0"]},
            "bottom": {"type": "dirichlet", "value": ["0.0", "0.0"]},
            "left": {"type": "dirichlet", "value": ["0.0", "0.0"]},
            "right": {"type": "dirichlet", "value": ["0.0", "0.0"]},
        }
    
    def parse_bc_value(val):
        if val is None:
            return [0.0, 0.0]
        if isinstance(val, (list, tuple)):
            return [float(v) for v in val]
        return [float(val), 0.0]
    
    # Boundary markers
    def top_boundary(x):
        return np.isclose(x[1], 1.0)
    
    def bottom_boundary(x):
        return np.isclose(x[1], 0.0)
    
    def left_boundary(x):
        return np.isclose(x[0], 0.0)
    
    def right_boundary(x):
        return np.isclose(x[0], 1.0)
    
    boundary_markers = {
        "top": top_boundary,
        "bottom": bottom_boundary,
        "left": left_boundary,
        "right": right_boundary,
    }
    
    for loc, bc_info in bc_dict.items():
        if bc_info["type"].lower() == "dirichlet" and loc in boundary_markers:
            val = parse_bc_value(bc_info["value"])
            marker_fn = boundary_markers[loc]
            
            facets = mesh.locate_entities_boundary(domain, fdim, marker_fn)
            
            u_bc = fem.Function(V)
            u_bc.interpolate(lambda x, v=val: np.full((domain.geometry.dim, x.shape[1]), 
                                                        np.array(v)[:, None]))
            
            dofs = fem.locate_dofs_topological((W.sub(0), V), fdim, facets)
            bc = fem.dirichletbc(u_bc, dofs, W.sub(0))
            bcs.append(bc)
    
    # If no BCs were created from spec, create default lid-driven cavity
    if len(bcs) == 0:
        # No-slip on bottom, left, right
        for marker_fn in [bottom_boundary, left_boundary, right_boundary]:
            facets = mesh.locate_entities_boundary(domain, fdim, marker_fn)
            u_bc = fem.Function(V)
            u_bc.interpolate(lambda x: np.zeros((domain.geometry.dim, x.shape[1])))
            dofs = fem.locate_dofs_topological((W.sub(0), V), fdim, facets)
            bc = fem.dirichletbc(u_bc, dofs, W.sub(0))
            bcs.append(bc)
        
        # Lid velocity on top
        facets_top = mesh.locate_entities_boundary(domain, fdim, top_boundary)
        u_lid = fem.Function(V)
        u_lid.interpolate(lambda x: np.vstack([np.ones(x.shape[1]), np.zeros(x.shape[1])]))
        dofs_top = fem.locate_dofs_topological((W.sub(0), V), fdim, facets_top)
        bc_top = fem.dirichletbc(u_lid, dofs_top, W.sub(0))
        bcs.append(bc_top)
    
    # Pin pressure at one point to remove nullspace
    # Find a point, e.g. bottom-left corner
    pressure_dofs = fem.locate_dofs_geometrical(
        (W.sub(1), Q), lambda x: np.isclose(x[0], 0.0) & np.isclose(x[1], 0.0)
    )
    p_zero = fem.Function(Q)
    p_zero.x.array[:] = 0.0
    bc_p = fem.dirichletbc(p_zero, pressure_dofs, W.sub(1))
    bcs.append(bc_p)
    
    # 6. Solve
    ksp_type = "gmres"
    pc_type = "lu"
    rtol = 1e-8
    
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
    
    wh = fem.Function(W)
    
    solver = PETSc.KSP().create(domain.comm)
    solver.setOperators(A)
    solver.setType(PETSc.KSP.Type.PREONLY)
    pc = solver.getPC()
    pc.setType(PETSc.PC.Type.LU)
    pc.setFactorSolverType("mumps")
    solver.setUp()
    
    solver.solve(b, wh.x.petsc_vec)
    wh.x.scatter_forward()
    
    iterations = solver.getIterationNumber()
    
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
        for idx, global_idx in enumerate(eval_map):
            ux = vals[idx, 0]
            uy = vals[idx, 1]
            vel_mag[global_idx] = np.sqrt(ux**2 + uy**2)
    
    u_grid = vel_mag.reshape((nx_out, ny_out))
    
    # Clean up PETSc objects
    solver.destroy()
    A.destroy()
    b.destroy()
    
    return {
        "u": u_grid,
        "solver_info": {
            "mesh_resolution": N,
            "element_degree": degree_u,
            "ksp_type": "preonly",
            "pc_type": "lu",
            "rtol": rtol,
            "iterations": max(iterations, 1),
        }
    }