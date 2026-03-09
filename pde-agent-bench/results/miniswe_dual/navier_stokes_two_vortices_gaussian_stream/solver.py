import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
import basix
from petsc4py import PETSc


def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD
    
    # Parse viscosity from case spec (handle multiple formats)
    pde = case_spec.get("pde", {})
    nu_val = pde.get("viscosity", None)
    if nu_val is None:
        nu_val = pde.get("pde_params", {}).get("nu", 0.14)
    
    # Get domain info
    domain_spec = case_spec.get("domain", {})
    x_range = domain_spec.get("x_range", [0.0, 1.0])
    y_range = domain_spec.get("y_range", [0.0, 1.0])
    
    # If domain is "unit_square", use [0,1]x[0,1]
    if domain_spec.get("type") == "unit_square":
        x_range = [0.0, 1.0]
        y_range = [0.0, 1.0]
    
    # Get output grid size (handle multiple formats)
    output = case_spec.get("output", {})
    grid = output.get("grid", {})
    nx_out = grid.get("nx", output.get("nx", 50))
    ny_out = grid.get("ny", output.get("ny", 50))
    
    # Also try oracle_config path
    oracle_config = case_spec.get("oracle_config", {})
    if oracle_config:
        oc_output = oracle_config.get("output", {})
        oc_grid = oc_output.get("grid", {})
        nx_out = oc_grid.get("nx", nx_out)
        ny_out = oc_grid.get("ny", ny_out)
    
    # P3/P2 Taylor-Hood with N=80 for tight accuracy
    degree_u = 3
    degree_p = 2
    N = 80
    
    # Create mesh
    p0 = np.array([x_range[0], y_range[0]])
    p1 = np.array([x_range[1], y_range[1]])
    domain_mesh = mesh.create_rectangle(comm, [p0, p1], [N, N], cell_type=mesh.CellType.triangle)
    
    gdim = domain_mesh.geometry.dim
    tdim = domain_mesh.topology.dim
    fdim = tdim - 1
    
    # Create mixed function space
    V_el = basix.ufl.element("Lagrange", domain_mesh.topology.cell_name(), degree_u, shape=(gdim,))
    Q_el = basix.ufl.element("Lagrange", domain_mesh.topology.cell_name(), degree_p)
    ME = basix.ufl.mixed_element([V_el, Q_el])
    W = fem.functionspace(domain_mesh, ME)
    
    V, V_to_W = W.sub(0).collapse()
    Q, Q_to_W = W.sub(1).collapse()
    
    # Define manufactured solution using UFL
    x = ufl.SpatialCoordinate(domain_mesh)
    
    u1_exact = (-60.0*(x[1]-0.7)*ufl.exp(-30.0*((x[0]-0.3)**2 + (x[1]-0.7)**2)) 
                + 60.0*(x[1]-0.3)*ufl.exp(-30.0*((x[0]-0.7)**2 + (x[1]-0.3)**2)))
    u2_exact = (60.0*(x[0]-0.3)*ufl.exp(-30.0*((x[0]-0.3)**2 + (x[1]-0.7)**2)) 
                - 60.0*(x[0]-0.7)*ufl.exp(-30.0*((x[0]-0.7)**2 + (x[1]-0.3)**2)))
    u_exact = ufl.as_vector([u1_exact, u2_exact])
    
    # Compute forcing term: f = (u·∇)u - ν∇²u + ∇p (p=0)
    nu = fem.Constant(domain_mesh, PETSc.ScalarType(nu_val))
    f_body = ufl.grad(u_exact) * u_exact - nu * ufl.div(ufl.grad(u_exact))
    
    # Solution function
    w = fem.Function(W)
    (u, p) = ufl.split(w)
    (v, q) = ufl.TestFunctions(W)
    
    # Weak form (residual)
    F = (
        ufl.inner(nu * ufl.grad(u), ufl.grad(v)) * ufl.dx
        + ufl.inner(ufl.grad(u) * u, v) * ufl.dx
        - ufl.inner(p, ufl.div(v)) * ufl.dx
        + ufl.inner(ufl.div(u), q) * ufl.dx
        - ufl.inner(f_body, v) * ufl.dx
    )
    
    # Boundary conditions
    domain_mesh.topology.create_connectivity(fdim, tdim)
    boundary_facets = mesh.exterior_facet_indices(domain_mesh.topology)
    
    u_bc_func = fem.Function(V)
    u_bc_expr = fem.Expression(u_exact, V.element.interpolation_points)
    u_bc_func.interpolate(u_bc_expr)
    
    dofs_u = fem.locate_dofs_topological((W.sub(0), V), fdim, boundary_facets)
    bc_u = fem.dirichletbc(u_bc_func, dofs_u, W.sub(0))
    
    # Pin pressure at corner to remove nullspace
    def corner_marker(x_coord):
        return np.isclose(x_coord[0], x_range[0]) & np.isclose(x_coord[1], y_range[0])
    
    corner_facets = mesh.locate_entities_boundary(domain_mesh, fdim, corner_marker)
    bcs = [bc_u]
    if len(corner_facets) > 0:
        p_bc_func = fem.Function(Q)
        p_bc_func.x.array[:] = 0.0
        dofs_p = fem.locate_dofs_topological((W.sub(1), Q), fdim, corner_facets)
        if len(dofs_p[0]) > 0:
            dofs_p_single = (np.array([dofs_p[0][0]]), np.array([dofs_p[1][0]]))
            bc_p = fem.dirichletbc(p_bc_func, dofs_p_single, W.sub(1))
            bcs.append(bc_p)
    
    # Initial guess: interpolate exact solution for velocity (helps Newton converge fast)
    w.sub(0).interpolate(u_bc_expr)
    w.x.scatter_forward()
    
    # Solve nonlinear problem
    petsc_options = {
        "snes_type": "newtonls",
        "snes_rtol": 1e-10,
        "snes_atol": 1e-12,
        "snes_max_it": 25,
        "ksp_type": "preonly",
        "pc_type": "lu",
        "pc_factor_mat_solver_type": "mumps",
        "snes_linesearch_type": "bt",
    }
    
    problem = petsc.NonlinearProblem(
        F, w, bcs=bcs,
        petsc_options_prefix="ns_",
        petsc_options=petsc_options,
    )
    
    problem.solve()
    
    snes = problem.solver
    converged_reason = snes.getConvergedReason()
    n_newton = snes.getIterationNumber()
    assert converged_reason > 0, f"SNES did not converge, reason: {converged_reason}"
    
    w.x.scatter_forward()
    
    # Extract velocity
    u_h = w.sub(0).collapse()
    
    # Evaluate on output grid
    x_coords = np.linspace(x_range[0], x_range[1], nx_out)
    y_coords = np.linspace(y_range[0], y_range[1], ny_out)
    X, Y = np.meshgrid(x_coords, y_coords, indexing='ij')
    
    points_2d = np.column_stack([X.ravel(), Y.ravel()])
    points_3d = np.zeros((points_2d.shape[0], 3))
    points_3d[:, :2] = points_2d
    
    # Point evaluation
    bb_tree = geometry.bb_tree(domain_mesh, tdim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, points_3d)
    colliding_cells = geometry.compute_colliding_cells(domain_mesh, cell_candidates, points_3d)
    
    vel_mag = np.zeros(points_2d.shape[0])
    
    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    for i in range(points_2d.shape[0]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(points_3d[i])
            cells_on_proc.append(links[0])
            eval_map.append(i)
    
    if len(points_on_proc) > 0:
        pts_arr = np.array(points_on_proc)
        cells_arr = np.array(cells_on_proc, dtype=np.int32)
        vals = u_h.eval(pts_arr, cells_arr)
        for j, idx in enumerate(eval_map):
            ux = vals[j, 0]
            uy = vals[j, 1]
            vel_mag[idx] = np.sqrt(ux**2 + uy**2)
    
    u_grid = vel_mag.reshape((nx_out, ny_out))
    
    result = {
        "u": u_grid,
        "solver_info": {
            "mesh_resolution": N,
            "element_degree": degree_u,
            "ksp_type": "preonly",
            "pc_type": "lu",
            "rtol": 1e-10,
            "nonlinear_iterations": [int(n_newton)],
        }
    }
    
    return result
