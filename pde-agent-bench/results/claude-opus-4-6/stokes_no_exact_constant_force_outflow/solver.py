import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc

ScalarType = PETSc.ScalarType


def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD
    
    # Parameters
    nx = ny = 64
    degree_u = 2
    degree_p = 1
    nu_val = 0.4
    
    # Create mesh
    domain = mesh.create_unit_square(comm, nx, ny, cell_type=mesh.CellType.triangle)
    tdim = domain.topology.dim
    fdim = tdim - 1
    
    # Function spaces (Taylor-Hood P2/P1)
    V = fem.functionspace(domain, ("Lagrange", degree_u, (domain.geometry.dim,)))
    Q = fem.functionspace(domain, ("Lagrange", degree_p))
    
    # Mixed function space
    mel = ufl.MixedElement([V.ufl_element(), Q.ufl_element()])
    W = fem.functionspace(domain, mel)
    
    # Trial and test functions
    (u, p) = ufl.TrialFunctions(W)
    (v, q) = ufl.TestFunctions(W)
    
    # Source term
    f = fem.Constant(domain, np.array([1.0, 0.0], dtype=ScalarType))
    nu = fem.Constant(domain, ScalarType(nu_val))
    
    # Bilinear form
    a = (nu * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
         - p * ufl.div(v) * ufl.dx
         + q * ufl.div(u) * ufl.dx)
    
    # Linear form
    L = ufl.inner(f, v) * ufl.dx
    
    # Boundary conditions
    # Identify boundary facets
    # No-slip on top and bottom (y=0, y=1)
    # No-slip on left (x=0)
    # Outflow on right (x=1) - natural BC (do-nothing)
    
    def left_boundary(x):
        return np.isclose(x[0], 0.0)
    
    def bottom_boundary(x):
        return np.isclose(x[1], 0.0)
    
    def top_boundary(x):
        return np.isclose(x[1], 1.0)
    
    bcs = []
    
    # No-slip on left
    facets_left = mesh.locate_entities_boundary(domain, fdim, left_boundary)
    V_sub, _ = W.sub(0).collapse()
    u_zero = fem.Function(V_sub)
    u_zero.interpolate(lambda x: np.zeros((domain.geometry.dim, x.shape[1])))
    dofs_left = fem.locate_dofs_topological((W.sub(0), V_sub), fdim, facets_left)
    bcs.append(fem.dirichletbc(u_zero, dofs_left, W.sub(0)))
    
    # No-slip on bottom
    facets_bottom = mesh.locate_entities_boundary(domain, fdim, bottom_boundary)
    u_zero_bot = fem.Function(V_sub)
    u_zero_bot.interpolate(lambda x: np.zeros((domain.geometry.dim, x.shape[1])))
    dofs_bottom = fem.locate_dofs_topological((W.sub(0), V_sub), fdim, facets_bottom)
    bcs.append(fem.dirichletbc(u_zero_bot, dofs_bottom, W.sub(0)))
    
    # No-slip on top
    facets_top = mesh.locate_entities_boundary(domain, fdim, top_boundary)
    u_zero_top = fem.Function(V_sub)
    u_zero_top.interpolate(lambda x: np.zeros((domain.geometry.dim, x.shape[1])))
    dofs_top = fem.locate_dofs_topological((W.sub(0), V_sub), fdim, facets_top)
    bcs.append(fem.dirichletbc(u_zero_top, dofs_top, W.sub(0)))
    
    # Pin pressure at one point to remove nullspace
    # Find a DOF near (1.0, 0.0) for pressure
    def corner_point(x):
        return np.logical_and(np.isclose(x[0], 1.0), np.isclose(x[1], 0.0))
    
    Q_sub, _ = W.sub(1).collapse()
    p_zero = fem.Function(Q_sub)
    p_zero.interpolate(lambda x: np.zeros(x.shape[1]))
    
    # Try to pin pressure at a point
    facets_corner = mesh.locate_entities_boundary(domain, fdim, lambda x: np.isclose(x[0], 1.0))
    if len(facets_corner) > 0:
        dofs_p = fem.locate_dofs_topological((W.sub(1), Q_sub), fdim, facets_corner[:1])
        if len(dofs_p[0]) > 0:
            bcs.append(fem.dirichletbc(p_zero, dofs_p, W.sub(1)))
    
    # Solve
    ksp_type = "gmres"  # was minres but gmres is more robust
    pc_type = "ilu"
    rtol = 1e-10
    
    problem = petsc.LinearProblem(
        a, L, bcs=bcs,
        petsc_options={
            "ksp_type": ksp_type,
            "pc_type": pc_type,
            "ksp_rtol": str(rtol),
            "ksp_max_it": "5000",
            "ksp_monitor": None,
        },
        petsc_options_prefix="stokes_"
    )
    wh = problem.solve()
    wh.x.scatter_forward()
    
    # Get iteration count
    ksp = problem.solver
    iterations = ksp.getIterationNumber()
    
    # Extract velocity
    u_sol = wh.sub(0).collapse()
    
    # Evaluate on 100x100 grid
    ngrid = 100
    xs = np.linspace(0, 1, ngrid)
    ys = np.linspace(0, 1, ngrid)
    XX, YY = np.meshgrid(xs, ys, indexing='ij')
    points = np.zeros((3, ngrid * ngrid))
    points[0, :] = XX.ravel()
    points[1, :] = YY.ravel()
    
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
    
    vel_mag = np.full(ngrid * ngrid, np.nan)
    if len(points_on_proc) > 0:
        pts_arr = np.array(points_on_proc)
        cells_arr = np.array(cells_on_proc, dtype=np.int32)
        vals = u_sol.eval(pts_arr, cells_arr)
        # vals shape: (n_points, 2) for 2D velocity
        mag = np.sqrt(vals[:, 0]**2 + vals[:, 1]**2)
        for idx, global_idx in enumerate(eval_map):
            vel_mag[global_idx] = mag[idx]
    
    u_grid = vel_mag.reshape((ngrid, ngrid))
    
    solver_info = {
        "mesh_resolution": nx,
        "element_degree": degree_u,
        "ksp_type": ksp_type,
        "pc_type": pc_type,
        "rtol": rtol,
        "iterations": int(iterations),
    }
    
    return {
        "u": u_grid,
        "solver_info": solver_info,
    }