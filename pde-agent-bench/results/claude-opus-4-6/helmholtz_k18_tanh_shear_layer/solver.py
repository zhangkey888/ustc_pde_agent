import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc

ScalarType = PETSc.ScalarType


def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD
    
    k_val = 18.0
    
    # For k=18, we need sufficient resolution. Rule of thumb: ~10 points per wavelength
    # wavelength = 2*pi/k ≈ 0.349, so need mesh size h ≈ 0.035 or smaller
    # With degree 2, we can use fewer elements
    mesh_resolution = 80
    element_degree = 2
    
    domain = mesh.create_unit_square(comm, mesh_resolution, mesh_resolution, 
                                      cell_type=mesh.CellType.triangle)
    
    V = fem.functionspace(domain, ("Lagrange", element_degree))
    
    # Spatial coordinates
    x = ufl.SpatialCoordinate(domain)
    
    # Exact solution
    u_exact_ufl = ufl.tanh(6.0 * (x[0] - 0.5)) * ufl.sin(ufl.pi * x[1])
    
    # Compute source term: f = -∇²u - k²u
    # We need -∇²u_exact - k²*u_exact
    # Let's compute ∇²u symbolically via UFL
    grad_u_exact = ufl.grad(u_exact_ufl)
    laplacian_u_exact = ufl.div(grad_u_exact)
    
    f_expr = -laplacian_u_exact - k_val**2 * u_exact_ufl
    
    # Trial and test functions
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    
    # Weak form: ∫ ∇u·∇v dx - k² ∫ u·v dx = ∫ f·v dx
    a = ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx - k_val**2 * ufl.inner(u, v) * ufl.dx
    L = ufl.inner(f_expr, v) * ufl.dx
    
    # Boundary conditions
    u_bc = fem.Function(V)
    u_bc_expr = fem.Expression(u_exact_ufl, V.element.interpolation_points)
    u_bc.interpolate(u_bc_expr)
    
    tdim = domain.topology.dim
    fdim = tdim - 1
    
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
    )
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc = fem.dirichletbc(u_bc, boundary_dofs)
    
    # Solve using direct solver (LU) for robustness with indefinite Helmholtz
    ksp_type = "preonly"
    pc_type = "lu"
    rtol = 1e-10
    
    problem = petsc.LinearProblem(
        a, L, bcs=[bc],
        petsc_options={
            "ksp_type": ksp_type,
            "pc_type": pc_type,
            "ksp_rtol": str(rtol),
        },
        petsc_options_prefix="helmholtz_"
    )
    u_sol = problem.solve()
    
    # Get iteration count
    iterations = problem.solver.getIterationNumber()
    
    # Evaluate on 50x50 grid
    nx_out, ny_out = 50, 50
    xs = np.linspace(0.0, 1.0, nx_out)
    ys = np.linspace(0.0, 1.0, ny_out)
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
    
    return {
        "u": u_grid,
        "solver_info": {
            "mesh_resolution": mesh_resolution,
            "element_degree": element_degree,
            "ksp_type": ksp_type,
            "pc_type": pc_type,
            "rtol": rtol,
            "iterations": iterations,
        }
    }