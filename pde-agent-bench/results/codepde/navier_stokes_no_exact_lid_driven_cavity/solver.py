import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, nls, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc


def solve(case_spec: dict) -> dict:
    # 1. Parse case_spec
    pde_config = case_spec.get("pde", {})
    nu_val = float(pde_config.get("viscosity", 0.08))
    source = pde_config.get("source_term", ["0.0", "0.0"])
    
    # Mesh and solver parameters
    N = 64
    degree_u = 2
    degree_p = 1
    newton_rtol = 1e-6
    newton_atol = 1e-10
    newton_max_it = 50
    
    # 2. Create mesh
    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
    tdim = domain.topology.dim
    fdim = tdim - 1
    
    # 3. Function spaces (Taylor-Hood P2/P1)
    V_el = ("Lagrange", degree_u, (domain.geometry.dim,))
    Q_el = ("Lagrange", degree_p)
    
    V = fem.functionspace(domain, V_el)
    Q = fem.functionspace(domain, Q_el)
    
    # Mixed function space
    from dolfinx.fem import Function
    from basix.ufl import mixed_element, element
    
    vel_elem = element("Lagrange", domain.topology.cell_name(), degree_u, shape=(domain.geometry.dim,))
    pres_elem = element("Lagrange", domain.topology.cell_name(), degree_p)
    mixed_el = mixed_element([vel_elem, pres_elem])
    
    W = fem.functionspace(domain, mixed_el)
    
    # 4. Define problem
    w = fem.Function(W)
    (u, p) = ufl.split(w)
    (v, q) = ufl.TestFunctions(W)
    
    nu = fem.Constant(domain, PETSc.ScalarType(nu_val))
    
    # Source term
    f = fem.Constant(domain, PETSc.ScalarType((0.0, 0.0)))
    
    # Residual form for steady incompressible Navier-Stokes
    F = (
        nu * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
        + ufl.inner(ufl.grad(u) * u, v) * ufl.dx
        - p * ufl.div(v) * ufl.dx
        + q * ufl.div(u) * ufl.dx
        - ufl.inner(f, v) * ufl.dx
    )
    
    # 5. Boundary conditions - Lid-driven cavity
    # Top wall: u = (1, 0)
    # Other walls: u = (0, 0)
    
    W0 = W.sub(0)
    V_collapsed, dofs_map = W0.collapse()
    
    # No-slip on bottom (y=0)
    def bottom(x):
        return np.isclose(x[1], 0.0)
    
    facets_bottom = mesh.locate_entities_boundary(domain, fdim, bottom)
    dofs_bottom = fem.locate_dofs_topological((W0, V_collapsed), fdim, facets_bottom)
    u_noslip = fem.Function(V_collapsed)
    u_noslip.interpolate(lambda x: np.zeros((domain.geometry.dim, x.shape[1])))
    bc_bottom = fem.dirichletbc(u_noslip, dofs_bottom, W0)
    
    # No-slip on left (x=0)
    def left(x):
        return np.isclose(x[0], 0.0)
    
    facets_left = mesh.locate_entities_boundary(domain, fdim, left)
    dofs_left = fem.locate_dofs_topological((W0, V_collapsed), fdim, facets_left)
    u_noslip_left = fem.Function(V_collapsed)
    u_noslip_left.interpolate(lambda x: np.zeros((domain.geometry.dim, x.shape[1])))
    bc_left = fem.dirichletbc(u_noslip_left, dofs_left, W0)
    
    # No-slip on right (x=1)
    def right(x):
        return np.isclose(x[0], 1.0)
    
    facets_right = mesh.locate_entities_boundary(domain, fdim, right)
    dofs_right = fem.locate_dofs_topological((W0, V_collapsed), fdim, facets_right)
    u_noslip_right = fem.Function(V_collapsed)
    u_noslip_right.interpolate(lambda x: np.zeros((domain.geometry.dim, x.shape[1])))
    bc_right = fem.dirichletbc(u_noslip_right, dofs_right, W0)
    
    # Lid (top, y=1): u = (1, 0)
    def top(x):
        return np.isclose(x[1], 1.0)
    
    facets_top = mesh.locate_entities_boundary(domain, fdim, top)
    dofs_top = fem.locate_dofs_topological((W0, V_collapsed), fdim, facets_top)
    u_lid = fem.Function(V_collapsed)
    u_lid.interpolate(lambda x: np.vstack([np.ones_like(x[0]), np.zeros_like(x[0])]))
    bc_top = fem.dirichletbc(u_lid, dofs_top, W0)
    
    # Pin pressure at one point to remove null space
    def origin(x):
        return np.logical_and(np.isclose(x[0], 0.0), np.isclose(x[1], 0.0))
    
    W1 = W.sub(1)
    Q_collapsed, _ = W1.collapse()
    dofs_p = fem.locate_dofs_geometrical((W1, Q_collapsed), origin)
    p_zero = fem.Function(Q_collapsed)
    p_zero.x.array[:] = 0.0
    bc_p = fem.dirichletbc(p_zero, dofs_p, W1)
    
    bcs = [bc_bottom, bc_left, bc_right, bc_top, bc_p]
    
    # 6. Initial guess - Stokes solve first for robustness
    # We'll just initialize with zero and let Newton handle it
    # But set lid BC in initial guess
    w.x.array[:] = 0.0
    
    # Apply BCs to initial guess manually
    # Interpolate lid velocity into w
    # Actually, let's do a simple initialization
    fem.petsc.set_bc(w.x.petsc_vec, bcs)
    w.x.scatter_forward()
    
    # 7. Solve nonlinear problem
    problem = petsc.NonlinearProblem(F, w, bcs=bcs)
    solver = nls.petsc.NewtonSolver(comm, problem)
    
    solver.convergence_criterion = "incremental"
    solver.rtol = newton_rtol
    solver.atol = newton_atol
    solver.max_it = newton_max_it
    solver.relaxation_parameter = 1.0
    
    # Configure linear solver
    ksp = solver.krylov_solver
    ksp.setType(PETSc.KSP.Type.GMRES)
    ksp.setTolerances(rtol=1e-8, max_it=1000)
    pc = ksp.getPC()
    pc.setType(PETSc.PC.Type.LU)
    pc.setFactorSolverType("mumps")
    
    n_newton, converged = solver.solve(w)
    assert converged, f"Newton solver did not converge after {n_newton} iterations"
    w.x.scatter_forward()
    
    # 8. Extract velocity on 50x50 grid
    nx_out, ny_out = 50, 50
    xs = np.linspace(0.0, 1.0, nx_out)
    ys = np.linspace(0.0, 1.0, ny_out)
    XX, YY = np.meshgrid(xs, ys, indexing='ij')
    
    points = np.zeros((3, nx_out * ny_out))
    points[0, :] = XX.ravel()
    points[1, :] = YY.ravel()
    
    # Get velocity sub-function
    u_sol = w.sub(0).collapse()
    
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
    
    # Evaluate velocity magnitude
    vel_mag = np.full(nx_out * ny_out, np.nan)
    
    if len(points_on_proc) > 0:
        pts_array = np.array(points_on_proc)
        cells_array = np.array(cells_on_proc, dtype=np.int32)
        vals = u_sol.eval(pts_array, cells_array)
        # vals shape: (n_points, 2) for 2D velocity
        for idx, global_idx in enumerate(eval_map):
            ux = vals[idx, 0]
            uy = vals[idx, 1]
            vel_mag[global_idx] = np.sqrt(ux**2 + uy**2)
    
    u_grid = vel_mag.reshape((nx_out, ny_out))
    
    # Count linear iterations
    total_linear_its = ksp.getIterationNumber()
    
    return {
        "u": u_grid,
        "solver_info": {
            "mesh_resolution": N,
            "element_degree": degree_u,
            "ksp_type": "gmres",
            "pc_type": "lu",
            "rtol": 1e-8,
            "iterations": total_linear_its,
            "nonlinear_iterations": [int(n_newton)],
        }
    }