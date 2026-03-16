import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc

ScalarType = PETSc.ScalarType


def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD
    
    k_val = 20.0
    
    # For k=20, we need sufficient resolution. Rule of thumb: ~10 points per wavelength
    # wavelength = 2*pi/k ≈ 0.314, so need mesh size h ≈ 0.03 or smaller
    # With P2 elements we can use coarser mesh
    # The manufactured solution has sin(6*pi*x)*sin(5*pi*y), highest frequency ~ 6*pi ≈ 18.85
    # Combined with k=20, we need good resolution
    
    mesh_resolution = 100
    element_degree = 2
    
    domain = mesh.create_unit_square(comm, mesh_resolution, mesh_resolution, 
                                      cell_type=mesh.CellType.triangle)
    
    V = fem.functionspace(domain, ("Lagrange", element_degree))
    
    x = ufl.SpatialCoordinate(domain)
    
    # Exact solution
    u_exact_ufl = ufl.sin(6 * ufl.pi * x[0]) * ufl.sin(5 * ufl.pi * x[1])
    
    # Source term: -∇²u - k²u = f
    # u = sin(6πx)sin(5πy)
    # ∇²u = -(36π² + 25π²) sin(6πx)sin(5πy) = -61π² u
    # -∇²u = 61π² u
    # f = -∇²u - k²u = 61π² u - k² u = (61π² - k²) u
    f_expr = (61.0 * ufl.pi**2 - k_val**2) * u_exact_ufl
    
    # Boundary condition
    u_bc_func = fem.Function(V)
    u_exact_expr = fem.Expression(u_exact_ufl, V.element.interpolation_points)
    u_bc_func.interpolate(u_exact_expr)
    
    # Mark all boundary facets
    tdim = domain.topology.dim
    fdim = tdim - 1
    domain.topology.create_connectivity(fdim, tdim)
    boundary_facets = mesh.exterior_facet_indices(domain.topology)
    dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc = fem.dirichletbc(u_bc_func, dofs)
    
    # Variational form: -∇²u - k²u = f
    # Weak form: ∫∇u·∇v dx - k²∫u·v dx = ∫f·v dx
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    
    k_const = fem.Constant(domain, ScalarType(k_val))
    
    a = ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx - k_const**2 * ufl.inner(u, v) * ufl.dx
    L = ufl.inner(f_expr, v) * ufl.dx
    
    # Solve using direct solver (LU) for robustness with indefinite system
    ksp_type = "preonly"
    pc_type = "lu"
    rtol = 1e-10
    
    problem = petsc.LinearProblem(
        a, L, bcs=[bc],
        petsc_options={
            "ksp_type": ksp_type,
            "pc_type": pc_type,
            "pc_factor_mat_solver_type": "mumps",
        },
        petsc_options_prefix="helmholtz_"
    )
    u_sol = problem.solve()
    
    # Get iteration count
    iterations = problem.solver.getIterationNumber()
    
    # Evaluate on 50x50 grid
    nx_eval, ny_eval = 50, 50
    xs = np.linspace(0, 1, nx_eval)
    ys = np.linspace(0, 1, ny_eval)
    XX, YY = np.meshgrid(xs, ys, indexing='ij')
    
    points = np.zeros((3, nx_eval * ny_eval))
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
    
    u_values = np.full(nx_eval * ny_eval, np.nan)
    if len(points_on_proc) > 0:
        pts_arr = np.array(points_on_proc)
        cells_arr = np.array(cells_on_proc, dtype=np.int32)
        vals = u_sol.eval(pts_arr, cells_arr)
        u_values[eval_map] = vals.flatten()
    
    u_grid = u_values.reshape((nx_eval, ny_eval))
    
    solver_info = {
        "mesh_resolution": mesh_resolution,
        "element_degree": element_degree,
        "ksp_type": ksp_type,
        "pc_type": pc_type,
        "rtol": rtol,
        "iterations": iterations,
    }
    
    return {
        "u": u_grid,
        "solver_info": solver_info,
    }