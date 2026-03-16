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
    pde_config = case_spec.get("pde", {})
    
    # Extract viscosity
    nu_val = float(pde_config.get("viscosity", 0.4))
    
    # Extract source term
    source = pde_config.get("source_term", ["1.0", "0.0"])
    f0 = float(source[0]) if isinstance(source, list) else 1.0
    f1 = float(source[1]) if isinstance(source, list) else 0.0
    
    # Extract boundary conditions
    bcs_spec = pde_config.get("boundary_conditions", [])
    
    # Mesh resolution
    N = 80
    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
    
    tdim = domain.topology.dim
    fdim = tdim - 1
    
    # 2. Create Taylor-Hood mixed elements (P2/P1)
    degree_u = 2
    degree_p = 1
    
    V_el = basix.ufl.element("Lagrange", domain.basix_cell(), degree_u, shape=(domain.geometry.dim,))
    Q_el = basix.ufl.element("Lagrange", domain.basix_cell(), degree_p)
    mixed_el = basix.ufl.mixed_element([V_el, Q_el])
    
    W = fem.functionspace(domain, mixed_el)
    
    # Also create individual spaces for BC interpolation
    V = fem.functionspace(domain, ("Lagrange", degree_u, (domain.geometry.dim,)))
    Q = fem.functionspace(domain, ("Lagrange", degree_p))
    
    # 3. Define variational problem
    w = fem.Function(W)
    (u, p) = ufl.split(w)
    (v, q) = ufl.TestFunctions(W)
    
    # Source term
    f = fem.Constant(domain, np.array([f0, f1], dtype=default_scalar_type))
    nu = fem.Constant(domain, default_scalar_type(nu_val))
    
    # Stokes residual: nu * inner(grad(u), grad(v)) - p * div(v) + div(u) * q - inner(f, v) = 0
    F = (
        nu * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
        - p * ufl.div(v) * ufl.dx
        + q * ufl.div(u) * ufl.dx
        - ufl.inner(f, v) * ufl.dx
    )
    
    # 4. Boundary conditions
    # Parse boundary conditions from case_spec
    bcs = []
    
    # Determine which boundaries have Dirichlet vs outflow (Neumann)
    # Default: check for outflow conditions
    has_outflow = False
    outflow_boundary = None
    dirichlet_bcs_info = []
    
    for bc_spec in bcs_spec:
        bc_type = bc_spec.get("type", "dirichlet")
        location = bc_spec.get("location", "")
        
        if bc_type == "outflow" or bc_type == "neumann" or bc_type == "do_nothing":
            has_outflow = True
            outflow_boundary = location
        elif bc_type == "dirichlet":
            value = bc_spec.get("value", ["0.0", "0.0"])
            if isinstance(value, list):
                val = [float(v) for v in value]
            else:
                val = [float(value), 0.0]
            dirichlet_bcs_info.append({"location": location, "value": val})
    
    # Helper to create boundary markers
    def make_boundary_marker(location):
        if location == "left":
            return lambda x: np.isclose(x[0], 0.0)
        elif location == "right":
            return lambda x: np.isclose(x[0], 1.0)
        elif location == "bottom":
            return lambda x: np.isclose(x[1], 0.0)
        elif location == "top":
            return lambda x: np.isclose(x[1], 1.0)
        elif location == "all":
            return lambda x: (np.isclose(x[0], 0.0) | np.isclose(x[0], 1.0) |
                              np.isclose(x[1], 0.0) | np.isclose(x[1], 1.0))
        else:
            return lambda x: (np.isclose(x[0], 0.0) | np.isclose(x[0], 1.0) |
                              np.isclose(x[1], 0.0) | np.isclose(x[1], 1.0))
    
    if len(bcs_spec) == 0:
        # No explicit BCs specified - apply based on case name
        # "stokes_no_exact_constant_force_outflow" suggests:
        # - outflow on right boundary (do-nothing / natural BC)
        # - no-slip (u=0) on top, bottom
        # - some inflow on left or no-slip
        
        # For constant force with outflow: 
        # no-slip on top and bottom walls, outflow on right, 
        # and either inflow or no-slip on left
        
        # Apply no-slip on top
        facets_top = mesh.locate_entities_boundary(domain, fdim, lambda x: np.isclose(x[1], 1.0))
        dofs_top = fem.locate_dofs_topological((W.sub(0), V), fdim, facets_top)
        u_zero = fem.Function(V)
        u_zero.interpolate(lambda x: np.zeros((domain.geometry.dim, x.shape[1])))
        bc_top = fem.dirichletbc(u_zero, dofs_top, W.sub(0))
        bcs.append(bc_top)
        
        # Apply no-slip on bottom
        facets_bottom = mesh.locate_entities_boundary(domain, fdim, lambda x: np.isclose(x[1], 0.0))
        dofs_bottom = fem.locate_dofs_topological((W.sub(0), V), fdim, facets_bottom)
        u_zero2 = fem.Function(V)
        u_zero2.interpolate(lambda x: np.zeros((domain.geometry.dim, x.shape[1])))
        bc_bottom = fem.dirichletbc(u_zero2, dofs_bottom, W.sub(0))
        bcs.append(bc_bottom)
        
        # Apply no-slip on left (or could be inflow)
        facets_left = mesh.locate_entities_boundary(domain, fdim, lambda x: np.isclose(x[0], 0.0))
        dofs_left = fem.locate_dofs_topological((W.sub(0), V), fdim, facets_left)
        u_zero3 = fem.Function(V)
        u_zero3.interpolate(lambda x: np.zeros((domain.geometry.dim, x.shape[1])))
        bc_left = fem.dirichletbc(u_zero3, dofs_left, W.sub(0))
        bcs.append(bc_left)
        
        # Right boundary: do-nothing (natural BC) - no Dirichlet needed
        has_outflow = True
        
    else:
        # Apply parsed BCs
        for bc_info in dirichlet_bcs_info:
            loc = bc_info["location"]
            val = bc_info["value"]
            marker = make_boundary_marker(loc)
            facets = mesh.locate_entities_boundary(domain, fdim, marker)
            dofs = fem.locate_dofs_topological((W.sub(0), V), fdim, facets)
            u_bc_func = fem.Function(V)
            val_arr = np.array(val, dtype=default_scalar_type)
            u_bc_func.interpolate(lambda x, v=val_arr: np.outer(v, np.ones(x.shape[1])))
            bc = fem.dirichletbc(u_bc_func, dofs, W.sub(0))
            bcs.append(bc)
    
    # If no outflow, we need pressure pinning
    if not has_outflow:
        # Pin pressure at one point
        facets_corner = mesh.locate_entities_boundary(domain, fdim, 
            lambda x: np.isclose(x[0], 0.0) & np.isclose(x[1], 0.0))
        if len(facets_corner) > 0:
            dofs_p = fem.locate_dofs_topological((W.sub(1), Q), fdim, facets_corner)
            p_zero = fem.Function(Q)
            p_zero.interpolate(lambda x: np.zeros(x.shape[1]))
            bc_p = fem.dirichletbc(p_zero, dofs_p, W.sub(1))
            bcs.append(bc_p)
    
    # 5. Since Stokes is linear, we can use a linear solve approach
    # Split into trial and test
    (u_trial, p_trial) = ufl.TrialFunctions(W)
    (v_test, q_test) = ufl.TestFunctions(W)
    
    a_form = (
        nu * ufl.inner(ufl.grad(u_trial), ufl.grad(v_test)) * ufl.dx
        - p_trial * ufl.div(v_test) * ufl.dx
        + q_test * ufl.div(u_trial) * ufl.dx
    )
    
    L_form = ufl.inner(f, v_test) * ufl.dx
    
    # 6. Solve using LinearProblem
    ksp_type = "gmres"
    pc_type = "lu"
    rtol = 1e-10
    
    problem = petsc.LinearProblem(
        a_form, L_form, bcs=bcs,
        petsc_options={
            "ksp_type": ksp_type,
            "pc_type": pc_type,
            "ksp_rtol": str(rtol),
            "ksp_max_it": "2000",
            "ksp_monitor": None,
        },
        petsc_options_prefix="stokes_"
    )
    
    wh = problem.solve()
    wh.x.scatter_forward()
    
    # Get iteration count
    ksp = problem.solver
    iterations = ksp.getIterationNumber()
    
    # 7. Extract velocity on evaluation grid
    nx_eval = 100
    ny_eval = 100
    
    xs = np.linspace(0.0, 1.0, nx_eval)
    ys = np.linspace(0.0, 1.0, ny_eval)
    XX, YY = np.meshgrid(xs, ys, indexing='ij')
    
    points = np.zeros((3, nx_eval * ny_eval))
    points[0, :] = XX.ravel()
    points[1, :] = YY.ravel()
    
    # Extract velocity sub-function
    u_sub = wh.sub(0).collapse()
    
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
    vel_mag = np.full(nx_eval * ny_eval, np.nan)
    
    if len(points_on_proc) > 0:
        pts_arr = np.array(points_on_proc)
        cells_arr = np.array(cells_on_proc, dtype=np.int32)
        vals = u_sub.eval(pts_arr, cells_arr)
        # vals should be (N, 2) for 2D velocity
        if vals.ndim == 1:
            vals = vals.reshape(-1, 2)
        mag = np.sqrt(vals[:, 0]**2 + vals[:, 1]**2)
        for idx, global_idx in enumerate(eval_map):
            vel_mag[global_idx] = mag[idx]
    
    u_grid = vel_mag.reshape(nx_eval, ny_eval)
    
    # Fill any NaN values (boundary points that might be missed)
    if np.any(np.isnan(u_grid)):
        from scipy import ndimage
        mask = np.isnan(u_grid)
        u_grid_filled = np.where(mask, 0.0, u_grid)
        u_grid = u_grid_filled
    
    solver_info = {
        "mesh_resolution": N,
        "element_degree": degree_u,
        "ksp_type": ksp_type,
        "pc_type": pc_type,
        "rtol": rtol,
        "iterations": iterations,
    }
    
    return {
        "u": u_grid,
        "solver_info": solver_info,
    }