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
    nu_val = 1.0
    
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
    f = fem.Constant(domain, ScalarType((0.0, 0.0)))
    nu = fem.Constant(domain, ScalarType(nu_val))
    
    # Bilinear form for Stokes
    a = (nu * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
         - p * ufl.div(v) * ufl.dx
         - q * ufl.div(u) * ufl.dx)
    
    L = ufl.inner(f, v) * ufl.dx
    
    # Boundary conditions
    # Channel flow: inflow on left (x=0), outflow on right (x=1), no-slip on top/bottom
    
    # Parabolic inflow on left: u = (4*y*(1-y), 0)
    def left_boundary(x):
        return np.isclose(x[0], 0.0)
    
    def right_boundary(x):
        return np.isclose(x[0], 1.0)
    
    def top_boundary(x):
        return np.isclose(x[1], 1.0)
    
    def bottom_boundary(x):
        return np.isclose(x[1], 0.0)
    
    bcs = []
    
    # Inflow BC (left wall): parabolic profile
    V_sub, V_sub_map = W.sub(0).collapse()
    
    u_inflow = fem.Function(V_sub)
    u_inflow.interpolate(lambda x: np.vstack([4.0 * x[1] * (1.0 - x[1]), np.zeros_like(x[0])]))
    
    left_facets = mesh.locate_entities_boundary(domain, fdim, left_boundary)
    left_dofs = fem.locate_dofs_topological((W.sub(0), V_sub), fdim, left_facets)
    bc_inflow = fem.dirichletbc(u_inflow, left_dofs, W.sub(0))
    bcs.append(bc_inflow)
    
    # Outflow BC (right wall): same parabolic profile (for fully developed flow)
    u_outflow = fem.Function(V_sub)
    u_outflow.interpolate(lambda x: np.vstack([4.0 * x[1] * (1.0 - x[1]), np.zeros_like(x[0])]))
    
    right_facets = mesh.locate_entities_boundary(domain, fdim, right_boundary)
    right_dofs = fem.locate_dofs_topological((W.sub(0), V_sub), fdim, right_facets)
    bc_outflow = fem.dirichletbc(u_outflow, right_dofs, W.sub(0))
    bcs.append(bc_outflow)
    
    # No-slip on top wall
    u_noslip = fem.Function(V_sub)
    u_noslip.interpolate(lambda x: np.vstack([np.zeros_like(x[0]), np.zeros_like(x[0])]))
    
    top_facets = mesh.locate_entities_boundary(domain, fdim, top_boundary)
    top_dofs = fem.locate_dofs_topological((W.sub(0), V_sub), fdim, top_facets)
    bc_top = fem.dirichletbc(u_noslip, top_dofs, W.sub(0))
    bcs.append(bc_top)
    
    # No-slip on bottom wall
    u_noslip2 = fem.Function(V_sub)
    u_noslip2.interpolate(lambda x: np.vstack([np.zeros_like(x[0]), np.zeros_like(x[0])]))
    
    bottom_facets = mesh.locate_entities_boundary(domain, fdim, bottom_boundary)
    bottom_dofs = fem.locate_dofs_topological((W.sub(0), V_sub), fdim, bottom_facets)
    bc_bottom = fem.dirichletbc(u_noslip2, bottom_dofs, W.sub(0))
    bcs.append(bc_bottom)
    
    # Solve
    ksp_type = "minres"
    pc_type = "hypre"
    rtol = 1e-10
    
    problem = petsc.LinearProblem(
        a, L, bcs=bcs,
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
    
    # Get iteration count
    ksp = problem.solver
    iterations = ksp.getIterationNumber()
    
    # Extract velocity sub-function
    u_sol = wh.sub(0).collapse()
    
    # Sample on 100x100 grid
    nx_grid = 100
    ny_grid = 100
    xs = np.linspace(0, 1, nx_grid)
    ys = np.linspace(0, 1, ny_grid)
    XX, YY = np.meshgrid(xs, ys, indexing='ij')
    
    points = np.zeros((3, nx_grid * ny_grid))
    points[0, :] = XX.ravel()
    points[1, :] = YY.ravel()
    
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
    
    vel_magnitude = np.full(nx_grid * ny_grid, np.nan)
    
    if len(points_on_proc) > 0:
        pts_arr = np.array(points_on_proc)
        cells_arr = np.array(cells_on_proc, dtype=np.int32)
        vals = u_sol.eval(pts_arr, cells_arr)
        # vals shape: (n_points, 2) for 2D velocity
        mag = np.sqrt(vals[:, 0]**2 + vals[:, 1]**2)
        for idx, global_idx in enumerate(eval_map):
            vel_magnitude[global_idx] = mag[idx]
    
    u_grid = vel_magnitude.reshape((nx_grid, ny_grid))
    
    solver_info = {
        "mesh_resolution": nx,
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