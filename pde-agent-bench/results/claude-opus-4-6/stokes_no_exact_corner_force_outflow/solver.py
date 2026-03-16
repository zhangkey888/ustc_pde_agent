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
    nu_val = 0.1
    mesh_res = 80
    degree_u = 2
    degree_p = 1
    
    # Create mesh
    domain = mesh.create_unit_square(comm, mesh_res, mesh_res, cell_type=mesh.CellType.triangle)
    tdim = domain.topology.dim
    fdim = tdim - 1
    
    # Function spaces (Taylor-Hood)
    V = fem.functionspace(domain, ("Lagrange", degree_u, (domain.geometry.dim,)))
    Q = fem.functionspace(domain, ("Lagrange", degree_p))
    
    # Mixed function space
    mel = ufl.MixedElement([V.ufl_element(), Q.ufl_element()])
    W = fem.functionspace(domain, mel)
    
    # Trial and test functions
    (u, p) = ufl.TrialFunctions(W)
    (v, q) = ufl.TestFunctions(W)
    
    # Source term
    x = ufl.SpatialCoordinate(domain)
    f = ufl.as_vector([
        3.0 * ufl.exp(-50.0 * ((x[0] - 0.15)**2 + (x[1] - 0.15)**2)),
        3.0 * ufl.exp(-50.0 * ((x[0] - 0.15)**2 + (x[1] - 0.15)**2))
    ])
    
    nu = fem.Constant(domain, ScalarType(nu_val))
    
    # Bilinear form
    a = (nu * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
         - p * ufl.div(v) * ufl.dx
         + q * ufl.div(u) * ufl.dx)
    
    # Linear form
    L = ufl.inner(f, v) * ufl.dx
    
    # Boundary conditions
    # Identify boundary facets
    # "corner_force_outflow" case: we need to figure out BCs
    # Based on the case name, let's apply no-slip (u=0) on walls except outflow
    # For "outflow" typically one boundary has natural BC (do-nothing)
    # Let's apply u=0 on left, bottom, top; outflow (natural) on right
    
    def left_boundary(x):
        return np.isclose(x[0], 0.0)
    
    def bottom_boundary(x):
        return np.isclose(x[1], 0.0)
    
    def top_boundary(x):
        return np.isclose(x[1], 1.0)
    
    def right_boundary(x):
        return np.isclose(x[0], 1.0)
    
    # For Stokes with "no_exact" and "outflow", apply u=0 on left, bottom, top
    # Right boundary: do-nothing (natural outflow BC)
    
    bcs = []
    
    # Zero velocity function
    u0 = fem.Function(V)
    u0.interpolate(lambda x: np.zeros((domain.geometry.dim, x.shape[1])))
    
    V_sub, _ = W.sub(0).collapse()
    
    for marker in [left_boundary, bottom_boundary, top_boundary]:
        facets = mesh.locate_entities_boundary(domain, fdim, marker)
        dofs = fem.locate_dofs_topological((W.sub(0), V), fdim, facets)
        bc = fem.dirichletbc(u0, dofs, W.sub(0))
        bcs.append(bc)
    
    # Pin pressure at one point to remove nullspace ambiguity
    # Since we have outflow on right, pressure should be determined, but let's be safe
    # Actually with do-nothing outflow, pressure is determined. Skip pinning.
    
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
    wh.x.scatter_forward()
    
    # Get iteration count
    ksp = problem.solver
    iterations = ksp.getIterationNumber()
    
    # Extract velocity sub-function
    u_sol = wh.sub(0).collapse()
    
    # Evaluate on 100x100 grid
    nx_out, ny_out = 100, 100
    xs = np.linspace(0, 1, nx_out)
    ys = np.linspace(0, 1, ny_out)
    XX, YY = np.meshgrid(xs, ys, indexing='ij')
    
    points = np.zeros((3, nx_out * ny_out))
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
    
    vel_mag = np.full(nx_out * ny_out, np.nan)
    
    if len(points_on_proc) > 0:
        pts_arr = np.array(points_on_proc)
        cells_arr = np.array(cells_on_proc, dtype=np.int32)
        vals = u_sol.eval(pts_arr, cells_arr)
        # vals shape: (n_points, 2) for 2D velocity
        mag = np.sqrt(vals[:, 0]**2 + vals[:, 1]**2)
        for idx, global_idx in enumerate(eval_map):
            vel_mag[global_idx] = mag[idx]
    
    u_grid = vel_mag.reshape((nx_out, ny_out))
    
    solver_info = {
        "mesh_resolution": mesh_res,
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