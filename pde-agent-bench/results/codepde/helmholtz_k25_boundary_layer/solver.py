import numpy as np
from dolfinx import mesh, fem, default_scalar_type, geometry
from dolfinx.fem import petsc
from mpi4py import MPI
import ufl
from petsc4py import PETSc


def solve(case_spec: dict) -> dict:
    # 1. Parse case_spec
    pde_config = case_spec.get("pde", {})
    k_val = float(pde_config.get("wavenumber", 25.0))
    
    # For k=25, we need sufficient resolution. Rule of thumb: ~10 points per wavelength
    # Wavelength = 2*pi/k ≈ 0.25 for k=25. On [0,1], need ~40 points per direction minimum.
    # Use higher order elements and fine mesh for accuracy
    mesh_resolution = 128
    element_degree = 2
    
    # 2. Create mesh
    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_square(comm, mesh_resolution, mesh_resolution, cell_type=mesh.CellType.triangle)
    
    # 3. Function space
    V = fem.functionspace(domain, ("Lagrange", element_degree))
    
    # 4. Define exact solution and source term using UFL
    x = ufl.SpatialCoordinate(domain)
    
    # Exact solution: u = exp(4*x)*sin(pi*y)
    u_exact_ufl = ufl.exp(4.0 * x[0]) * ufl.sin(ufl.pi * x[1])
    
    # Compute source term: f = -∇²u - k²u
    # ∇²u = d²u/dx² + d²u/dy²
    # u = exp(4x)*sin(pi*y)
    # du/dx = 4*exp(4x)*sin(pi*y), d²u/dx² = 16*exp(4x)*sin(pi*y)
    # du/dy = pi*exp(4x)*cos(pi*y), d²u/dy² = -pi²*exp(4x)*sin(pi*y)
    # ∇²u = (16 - pi²)*exp(4x)*sin(pi*y)
    # f = -∇²u - k²u = -(16 - pi²)*exp(4x)*sin(pi*y) - k²*exp(4x)*sin(pi*y)
    #   = (-16 + pi² - k²)*exp(4x)*sin(pi*y)
    
    k2 = k_val * k_val
    pi_val = ufl.pi
    f_expr = (-16.0 + pi_val**2 - k2) * ufl.exp(4.0 * x[0]) * ufl.sin(pi_val * x[1])
    
    # 5. Variational problem
    # -∇²u - k²u = f  =>  ∫∇u·∇v dx - k²∫u·v dx = ∫f·v dx
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    
    k_const = fem.Constant(domain, PETSc.ScalarType(k_val))
    
    a = ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx - k_const**2 * ufl.inner(u, v) * ufl.dx
    L = ufl.inner(f_expr, v) * ufl.dx
    
    # 6. Boundary conditions
    tdim = domain.topology.dim
    fdim = tdim - 1
    
    # Create BC function by interpolating exact solution
    u_bc = fem.Function(V)
    u_bc.interpolate(lambda x: np.exp(4.0 * x[0]) * np.sin(np.pi * x[1]))
    
    # All boundary facets
    domain.topology.create_connectivity(fdim, tdim)
    boundary_facets = mesh.exterior_facets(domain.topology)
    
    dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc = fem.dirichletbc(u_bc, dofs)
    
    # 7. Solve - use direct solver for indefinite Helmholtz
    ksp_type = "gmres"
    pc_type = "lu"
    rtol = 1e-10
    
    problem = petsc.LinearProblem(
        a, L, bcs=[bc],
        petsc_options={
            "ksp_type": ksp_type,
            "pc_type": pc_type,
            "ksp_rtol": str(rtol),
            "ksp_max_it": "2000",
            "ksp_monitor": None,
        },
        petsc_options_prefix="helmholtz_"
    )
    uh = problem.solve()
    
    # Get iteration count
    ksp = problem.solver
    iterations = ksp.getIterationNumber()
    
    # 8. Extract solution on 50x50 uniform grid
    nx_out, ny_out = 50, 50
    xs = np.linspace(0.0, 1.0, nx_out)
    ys = np.linspace(0.0, 1.0, ny_out)
    XX, YY = np.meshgrid(xs, ys, indexing='ij')
    
    points_2d = np.zeros((3, nx_out * ny_out))
    points_2d[0, :] = XX.ravel()
    points_2d[1, :] = YY.ravel()
    
    # Point evaluation
    bb_tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, points_2d.T)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points_2d.T)
    
    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    for i in range(points_2d.shape[1]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(points_2d[:, i])
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
            "mesh_resolution": mesh_resolution,
            "element_degree": element_degree,
            "ksp_type": ksp_type,
            "pc_type": pc_type,
            "rtol": rtol,
            "iterations": int(iterations),
        }
    }