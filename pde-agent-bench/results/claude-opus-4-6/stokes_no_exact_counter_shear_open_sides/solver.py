import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc

ScalarType = PETSc.ScalarType


def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD
    
    # Parse case spec
    nu_val = case_spec.get("pde", {}).get("viscosity", 1.0)
    
    # Mesh resolution - use high resolution for accuracy
    N = 80
    
    domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
    tdim = domain.topology.dim
    fdim = tdim - 1
    
    # Taylor-Hood elements: P2/P1
    degree_u = 2
    degree_p = 1
    
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
    # "counter_shear_open_sides" pattern:
    # Top wall: u = (1, 0) moving right
    # Bottom wall: u = (-1, 0) moving left (counter-shear)
    # Left and right: open (no BC, natural/do-nothing)
    
    # Top boundary: y = 1
    def top_boundary(x):
        return np.isclose(x[1], 1.0)
    
    # Bottom boundary: y = 0
    def bottom_boundary(x):
        return np.isclose(x[1], 0.0)
    
    # Top: u = (1, 0)
    top_facets = mesh.locate_entities_boundary(domain, fdim, top_boundary)
    u_top = fem.Function(V)
    u_top.interpolate(lambda x: np.vstack([np.ones_like(x[0]), np.zeros_like(x[0])]))
    dofs_top = fem.locate_dofs_topological((W.sub(0), V), fdim, top_facets)
    bc_top = fem.dirichletbc(u_top, dofs_top, W.sub(0))
    
    # Bottom: u = (-1, 0)
    bottom_facets = mesh.locate_entities_boundary(domain, fdim, bottom_boundary)
    u_bottom = fem.Function(V)
    u_bottom.interpolate(lambda x: np.vstack([-np.ones_like(x[0]), np.zeros_like(x[0])]))
    dofs_bottom = fem.locate_dofs_topological((W.sub(0), V), fdim, bottom_facets)
    bc_bottom = fem.dirichletbc(u_bottom, dofs_bottom, W.sub(0))
    
    bcs = [bc_top, bc_bottom]
    
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
    
    # Sample on 100x100 grid
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
    
    vel_magnitude = np.full(nx_out * ny_out, np.nan)
    
    if len(points_on_proc) > 0:
        pts_arr = np.array(points_on_proc)
        cells_arr = np.array(cells_on_proc, dtype=np.int32)
        vals = u_sol.eval(pts_arr, cells_arr)
        # vals shape: (n_points, 2) for 2D velocity
        mag = np.sqrt(vals[:, 0]**2 + vals[:, 1]**2)
        for idx, global_idx in enumerate(eval_map):
            vel_magnitude[global_idx] = mag[idx]
    
    u_grid = vel_magnitude.reshape((nx_out, ny_out))
    
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