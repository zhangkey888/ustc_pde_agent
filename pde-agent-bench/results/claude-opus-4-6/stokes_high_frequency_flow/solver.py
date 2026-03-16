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
    nu_val = 1.0
    N = 32  # mesh resolution - P2/P1 Taylor-Hood on 32x32 should be sufficient for 2*pi frequency
    degree_u = 2
    degree_p = 1
    
    # Create mesh
    domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
    tdim = domain.topology.dim
    fdim = tdim - 1
    
    # Function spaces - Taylor-Hood P2/P1
    V = fem.functionspace(domain, ("Lagrange", degree_u, (domain.geometry.dim,)))
    Q = fem.functionspace(domain, ("Lagrange", degree_p))
    
    # Mixed function space
    mel = ufl.MixedElement([V.ufl_element(), Q.ufl_element()])
    W = fem.functionspace(domain, mel)
    
    # Trial and test functions
    (u, p) = ufl.TrialFunctions(W)
    (v, q) = ufl.TestFunctions(W)
    
    # Spatial coordinates
    x = ufl.SpatialCoordinate(domain)
    
    # Exact solution
    u_exact = ufl.as_vector([
        2 * ufl.pi * ufl.cos(2 * ufl.pi * x[1]) * ufl.sin(2 * ufl.pi * x[0]),
        -2 * ufl.pi * ufl.cos(2 * ufl.pi * x[0]) * ufl.sin(2 * ufl.pi * x[1])
    ])
    p_exact = ufl.sin(2 * ufl.pi * x[0]) * ufl.cos(2 * ufl.pi * x[1])
    
    # Compute source term: f = -nu * laplacian(u_exact) + grad(p_exact)
    # For u1 = 2*pi*cos(2*pi*y)*sin(2*pi*x):
    #   laplacian(u1) = d^2u1/dx^2 + d^2u1/dy^2 = -8*pi^3*cos(2*pi*y)*sin(2*pi*x) + (-8*pi^3*cos(2*pi*y)*sin(2*pi*x))
    #                 = -2*(2*pi)^2 * u1 = -(2*pi)^2 * u1 * 2 ... let me just use UFL
    # Actually let's just derive f symbolically via UFL
    f = -nu_val * ufl.div(ufl.grad(u_exact)) + ufl.grad(p_exact)
    
    # Bilinear form
    nu = fem.Constant(domain, ScalarType(nu_val))
    a = (nu * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
         - p * ufl.div(v) * ufl.dx
         - q * ufl.div(u) * ufl.dx)
    
    # Linear form
    L = ufl.inner(f, v) * ufl.dx
    
    # Boundary conditions - all boundaries
    def boundary_all(x):
        return (np.isclose(x[0], 0.0) | np.isclose(x[0], 1.0) |
                np.isclose(x[1], 0.0) | np.isclose(x[1], 1.0))
    
    boundary_facets = mesh.locate_entities_boundary(domain, fdim, boundary_all)
    
    # Velocity BC
    u_bc_func = fem.Function(V)
    u_bc_func.interpolate(lambda x: np.stack([
        2 * np.pi * np.cos(2 * np.pi * x[1]) * np.sin(2 * np.pi * x[0]),
        -2 * np.pi * np.cos(2 * np.pi * x[0]) * np.sin(2 * np.pi * x[1])
    ]))
    
    V_sub, _ = W.sub(0).collapse()
    dofs_u = fem.locate_dofs_topological((W.sub(0), V), fdim, boundary_facets)
    bc_u = fem.dirichletbc(u_bc_func, dofs_u, W.sub(0))
    
    bcs = [bc_u]
    
    # Solve
    problem = petsc.LinearProblem(
        a, L, bcs=bcs,
        petsc_options={
            "ksp_type": "gmres",
            "pc_type": "lu",
            "pc_factor_mat_solver_type": "mumps",
            "ksp_rtol": 1e-12,
            "ksp_max_it": 1000,
        },
        petsc_options_prefix="stokes_"
    )
    wh = problem.solve()
    
    # Get iteration count
    ksp = problem.solver
    iterations = ksp.getIterationNumber()
    
    # Extract velocity sub-function
    uh = wh.sub(0).collapse()
    
    # Evaluate on 100x100 grid
    nx_out, ny_out = 100, 100
    xs = np.linspace(0, 1, nx_out)
    ys = np.linspace(0, 1, ny_out)
    XX, YY = np.meshgrid(xs, ys, indexing='ij')
    
    points = np.zeros((3, nx_out * ny_out))
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
    
    u_values = np.full((points.shape[1], 2), np.nan)
    if len(points_on_proc) > 0:
        pts_arr = np.array(points_on_proc)
        cells_arr = np.array(cells_on_proc, dtype=np.int32)
        vals = uh.eval(pts_arr, cells_arr)
        for idx, global_idx in enumerate(eval_map):
            u_values[global_idx, :] = vals[idx, :]
    
    # Compute velocity magnitude
    vel_mag = np.sqrt(u_values[:, 0]**2 + u_values[:, 1]**2)
    u_grid = vel_mag.reshape((nx_out, ny_out))
    
    return {
        "u": u_grid,
        "solver_info": {
            "mesh_resolution": N,
            "element_degree": degree_u,
            "ksp_type": "gmres",
            "pc_type": "lu",
            "rtol": 1e-12,
            "iterations": iterations,
        }
    }