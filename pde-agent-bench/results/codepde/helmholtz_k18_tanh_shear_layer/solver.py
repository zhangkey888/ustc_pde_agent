import numpy as np
from dolfinx import mesh, fem, default_scalar_type, geometry
from dolfinx.fem import petsc
from mpi4py import MPI
import ufl
from petsc4py import PETSc


def solve(case_spec: dict) -> dict:
    # 1. Parse case_spec
    pde_config = case_spec.get("pde", {})
    k_val = float(pde_config.get("helmholtz_k", 18.0))
    
    # For k=18, we need sufficient resolution. Rule of thumb: ~10 points per wavelength
    # wavelength = 2*pi/k ≈ 0.349, so on [0,1] we need ~10/0.349 ≈ 29 points per unit
    # But with higher order elements we can use fewer. Use degree 3 with moderate mesh.
    # Also the tanh shear layer at x=0.5 with steepness 6 needs resolution.
    
    degree = 3
    N = 80  # mesh resolution
    
    # 2. Create mesh
    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
    
    # 3. Function space
    V = fem.functionspace(domain, ("Lagrange", degree))
    
    # 4. Define variational problem
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    
    x = ufl.SpatialCoordinate(domain)
    pi = ufl.pi
    
    # Manufactured solution: u_exact = tanh(6*(x-0.5))*sin(pi*y)
    # Let alpha = 6
    # u_exact = tanh(alpha*(x[0]-0.5)) * sin(pi*x[1])
    # 
    # Derivatives:
    # du/dx = alpha * sech^2(alpha*(x-0.5)) * sin(pi*y)
    #       = alpha * (1 - tanh^2(alpha*(x-0.5))) * sin(pi*y)
    # d2u/dx2 = -2*alpha^2 * tanh(alpha*(x-0.5)) * (1-tanh^2(alpha*(x-0.5))) * sin(pi*y)
    #
    # du/dy = tanh(alpha*(x-0.5)) * pi * cos(pi*y)
    # d2u/dy2 = -pi^2 * tanh(alpha*(x-0.5)) * sin(pi*y)
    #
    # laplacian = d2u/dx2 + d2u/dy2
    # f = -laplacian - k^2 * u_exact
    
    alpha = 6.0
    
    # Use UFL expressions
    th = ufl.tanh(alpha * (x[0] - 0.5))
    sy = ufl.sin(pi * x[1])
    
    u_exact = th * sy
    
    # Compute Laplacian symbolically
    # d2u/dx2 = -2*alpha^2 * th * (1 - th^2) * sy
    d2u_dx2 = -2.0 * alpha**2 * th * (1.0 - th**2) * sy
    # d2u/dy2 = -pi^2 * th * sy
    d2u_dy2 = -pi**2 * th * sy
    
    laplacian_u = d2u_dx2 + d2u_dy2
    
    k2 = fem.Constant(domain, PETSc.ScalarType(k_val**2))
    
    # f = -laplacian_u - k^2 * u_exact
    f_expr = -laplacian_u - k2 * u_exact
    
    # Weak form: integral(grad(u).grad(v)) - k^2 * integral(u*v) = integral(f*v)
    a = ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx - k2 * ufl.inner(u, v) * ufl.dx
    L = ufl.inner(f_expr, v) * ufl.dx
    
    # 5. Boundary conditions
    tdim = domain.topology.dim
    fdim = tdim - 1
    
    # Find all boundary facets
    domain.topology.create_connectivity(fdim, tdim)
    boundary_facets = mesh.exterior_facets(domain.topology)
    
    # Create BC function
    u_bc = fem.Function(V)
    u_bc.interpolate(lambda X: np.tanh(6.0 * (X[0] - 0.5)) * np.sin(np.pi * X[1]))
    
    dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc = fem.dirichletbc(u_bc, dofs)
    
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
    
    # 7. Extract solution on 50x50 uniform grid
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
    for i in range(nx_out * ny_out):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(points[:, i])
            cells_on_proc.append(links[0])
            eval_map.append(i)
    
    u_values = np.full(nx_out * ny_out, np.nan)
    if len(points_on_proc) > 0:
        pts_arr = np.array(points_on_proc)
        cells_arr = np.array(cells_on_proc, dtype=np.int32)
        vals = uh.eval(pts_arr, cells_arr)
        u_values[eval_map] = vals.flatten()
    
    u_grid = u_values.reshape((nx_out, ny_out))
    
    return {
        "u": u_grid,
        "solver_info": {
            "mesh_resolution": N,
            "element_degree": degree,
            "ksp_type": ksp_type,
            "pc_type": pc_type,
            "rtol": rtol,
            "iterations": 1,
        }
    }