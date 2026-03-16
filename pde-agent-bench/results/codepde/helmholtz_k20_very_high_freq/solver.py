import numpy as np
from dolfinx import mesh, fem, default_scalar_type, geometry
from dolfinx.fem import petsc
from mpi4py import MPI
import ufl
from petsc4py import PETSc


def solve(case_spec: dict) -> dict:
    # 1. Parse case_spec
    pde_config = case_spec.get("pde", {})
    k_val = float(pde_config.get("helmholtz_k", 20.0))
    
    # For k=20, we need sufficient resolution. Rule of thumb: ~10 points per wavelength
    # Wavelength = 2*pi/k ≈ 0.314 for k=20
    # The manufactured solution has sin(6*pi*x)*sin(5*pi*y), so effective wavenumbers
    # are 6*pi ≈ 18.85 and 5*pi ≈ 15.71, plus k=20
    # We need high resolution. Use degree 2 elements with fine mesh.
    
    mesh_resolution = 120
    element_degree = 3
    
    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_square(comm, mesh_resolution, mesh_resolution, cell_type=mesh.CellType.triangle)
    
    # 3. Function space
    V = fem.functionspace(domain, ("Lagrange", element_degree))
    
    # 4. Define variational problem
    # Manufactured solution: u_exact = sin(6*pi*x)*sin(5*pi*y)
    # -∇²u - k²u = f
    # ∇²u_exact = -(6π)²sin(6πx)sin(5πy) - (5π)²sin(6πx)sin(5πy) = -(36π²+25π²)sin(6πx)sin(5πy) = -61π²u_exact
    # So -∇²u_exact = 61π²u_exact
    # f = -∇²u_exact - k²u_exact = 61π²u_exact - k²u_exact = (61π²-k²)u_exact
    
    x = ufl.SpatialCoordinate(domain)
    pi = np.pi
    
    u_exact_expr = ufl.sin(6 * pi * x[0]) * ufl.sin(5 * pi * x[1])
    f_coeff = 61.0 * pi**2 - k_val**2
    f_expr = f_coeff * u_exact_expr
    
    u_trial = ufl.TrialFunction(V)
    v_test = ufl.TestFunction(V)
    
    # Weak form: ∫ ∇u·∇v dx - k²∫ u*v dx = ∫ f*v dx
    k_const = fem.Constant(domain, PETSc.ScalarType(k_val))
    a = ufl.inner(ufl.grad(u_trial), ufl.grad(v_test)) * ufl.dx - k_const**2 * ufl.inner(u_trial, v_test) * ufl.dx
    L = ufl.inner(f_expr, v_test) * ufl.dx
    
    # 5. Boundary conditions: u = g = u_exact on ∂Ω
    u_bc_func = fem.Function(V)
    u_exact_fem_expr = fem.Expression(u_exact_expr, V.element.interpolation_points)
    u_bc_func.interpolate(u_exact_fem_expr)
    
    tdim = domain.topology.dim
    fdim = tdim - 1
    
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
    )
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc = fem.dirichletbc(u_bc_func, boundary_dofs)
    
    # 6. Solve - use direct solver for robustness with indefinite Helmholtz
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
    uh = problem.solve()
    
    # Get iteration count
    iterations = problem.solver.getIterationNumber()
    
    # 7. Extract solution on 50x50 uniform grid
    nx_out, ny_out = 50, 50
    xs = np.linspace(0.0, 1.0, nx_out)
    ys = np.linspace(0.0, 1.0, ny_out)
    XX, YY = np.meshgrid(xs, ys, indexing='ij')
    
    points_2d = np.stack([XX.ravel(), YY.ravel()], axis=0)
    points_3d = np.vstack([points_2d, np.zeros((1, points_2d.shape[1]))])
    
    # Point evaluation
    bb_tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, points_3d.T)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points_3d.T)
    
    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    
    for i in range(points_3d.shape[1]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(points_3d[:, i])
            cells_on_proc.append(links[0])
            eval_map.append(i)
    
    u_values = np.full(points_3d.shape[1], np.nan)
    if len(points_on_proc) > 0:
        pts_arr = np.array(points_on_proc)
        cells_arr = np.array(cells_on_proc, dtype=np.int32)
        vals = uh.eval(pts_arr, cells_arr)
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