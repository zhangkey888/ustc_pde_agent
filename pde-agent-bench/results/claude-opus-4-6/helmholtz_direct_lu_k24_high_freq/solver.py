import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc


def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD
    
    k_val = 24.0
    
    # For k=24, we need sufficient resolution. Rule of thumb: ~10 points per wavelength
    # wavelength = 2*pi/k ≈ 0.26, so need mesh size h ≈ 0.026 or smaller
    # With P2 elements we can use coarser mesh
    mesh_resolution = 80
    element_degree = 2
    
    domain = mesh.create_unit_square(comm, mesh_resolution, mesh_resolution, cell_type=mesh.CellType.triangle)
    
    V = fem.functionspace(domain, ("Lagrange", element_degree))
    
    x = ufl.SpatialCoordinate(domain)
    
    # Exact solution
    u_exact_expr = ufl.sin(5 * ufl.pi * x[0]) * ufl.sin(4 * ufl.pi * x[1])
    
    # Source term: -∇²u - k²u = f
    # u = sin(5πx)sin(4πy)
    # ∇²u = -(25π² + 16π²)sin(5πx)sin(4πy) = -41π²u
    # -∇²u - k²u = 41π²u - k²u = (41π² - k²)u
    f_expr = (41.0 * ufl.pi**2 - k_val**2) * ufl.sin(5 * ufl.pi * x[0]) * ufl.sin(4 * ufl.pi * x[1])
    
    # Boundary condition
    u_bc_func = fem.Function(V)
    u_bc_func.interpolate(lambda x: np.sin(5 * np.pi * x[0]) * np.sin(4 * np.pi * x[1]))
    
    tdim = domain.topology.dim
    fdim = tdim - 1
    
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
    )
    dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc = fem.dirichletbc(u_bc_func, dofs)
    
    # Variational form: ∇u·∇v - k²uv = fv
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    
    a = ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx - k_val**2 * ufl.inner(u, v) * ufl.dx
    L = ufl.inner(f_expr, v) * ufl.dx
    
    # Use direct LU solver for indefinite Helmholtz
    problem = petsc.LinearProblem(
        a, L, bcs=[bc],
        petsc_options={
            "ksp_type": "preonly",
            "pc_type": "lu",
        },
        petsc_options_prefix="helmholtz_"
    )
    u_sol = problem.solve()
    
    # Evaluate on 50x50 grid
    nx_out, ny_out = 50, 50
    xs = np.linspace(0, 1, nx_out)
    ys = np.linspace(0, 1, ny_out)
    XX, YY = np.meshgrid(xs, ys, indexing='ij')
    
    points = np.zeros((3, nx_out * ny_out))
    points[0, :] = XX.ravel()
    points[1, :] = YY.ravel()
    
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
    
    u_values = np.full(nx_out * ny_out, np.nan)
    if len(points_on_proc) > 0:
        pts_arr = np.array(points_on_proc)
        cells_arr = np.array(cells_on_proc, dtype=np.int32)
        vals = u_sol.eval(pts_arr, cells_arr)
        u_values[eval_map] = vals.flatten()
    
    u_grid = u_values.reshape((nx_out, ny_out))
    
    solver_info = {
        "mesh_resolution": mesh_resolution,
        "element_degree": element_degree,
        "ksp_type": "preonly",
        "pc_type": "lu",
        "rtol": 1e-10,
        "iterations": 1,
    }
    
    return {
        "u": u_grid,
        "solver_info": solver_info,
    }