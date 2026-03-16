import numpy as np
from dolfinx import mesh, fem, default_scalar_type, geometry
from dolfinx.fem.petsc import LinearProblem
from mpi4py import MPI
import ufl
from petsc4py import PETSc


def solve(case_spec: dict) -> dict:
    # 1. Parse case_spec
    pde_config = case_spec.get("pde", case_spec.get("oracle_config", {}).get("pde", {}))
    k_val = float(pde_config.get("wavenumber", 10.0))
    
    # 2. Create mesh - use quadrilateral as specified in case ID
    nx, ny = 80, 80
    degree = 2
    domain = mesh.create_unit_square(MPI.COMM_WORLD, nx, ny, cell_type=mesh.CellType.quadrilateral)
    
    # 3. Function space
    V = fem.functionspace(domain, ("Lagrange", degree))
    
    # 4. Define exact solution and source term using UFL
    x = ufl.SpatialCoordinate(domain)
    pi = ufl.pi
    
    # Exact solution: u = sin(2*pi*x)*cos(3*pi*y)
    u_exact_ufl = ufl.sin(2 * pi * x[0]) * ufl.cos(3 * pi * x[1])
    
    # Compute -laplacian(u_exact): 
    # u = sin(2*pi*x)*cos(3*pi*y)
    # u_xx = -(2*pi)^2 * sin(2*pi*x)*cos(3*pi*y)
    # u_yy = -(3*pi)^2 * sin(2*pi*x)*cos(3*pi*y)
    # laplacian(u) = -(4*pi^2 + 9*pi^2) * sin(2*pi*x)*cos(3*pi*y) = -13*pi^2 * u
    # -laplacian(u) = 13*pi^2 * u
    # f = -laplacian(u) - k^2 * u = (13*pi^2 - k^2) * u
    
    k2 = fem.Constant(domain, default_scalar_type(k_val**2))
    f_expr = (13.0 * pi**2 - k2) * u_exact_ufl
    
    # 5. Variational problem: -∇²u - k²u = f
    # Weak form: ∫ ∇u·∇v dx - k² ∫ u*v dx = ∫ f*v dx
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    
    a = ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx - k2 * ufl.inner(u, v) * ufl.dx
    L = ufl.inner(f_expr, v) * ufl.dx
    
    # 6. Boundary conditions
    tdim = domain.topology.dim
    fdim = tdim - 1
    
    # Find all boundary facets
    domain.topology.create_connectivity(fdim, tdim)
    boundary_facets = mesh.exterior_facet_indices(domain.topology)
    
    # Create BC function by interpolating exact solution
    u_bc = fem.Function(V)
    u_exact_expr = fem.Expression(u_exact_ufl, V.element.interpolation_points)
    u_bc.interpolate(u_exact_expr)
    
    dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc = fem.dirichletbc(u_bc, dofs)
    
    # 7. Solve - use direct solver for indefinite Helmholtz
    ksp_type = "gmres"
    pc_type = "ilu"
    rtol = 1e-10
    
    problem = LinearProblem(
        a, L, bcs=[bc],
        petsc_options={
            "ksp_type": ksp_type,
            "pc_type": pc_type,
            "ksp_rtol": str(rtol),
            "ksp_max_it": "2000",
            "ksp_gmres_restart": "100",
        },
        petsc_options_prefix="helmholtz_"
    )
    uh = problem.solve()
    
    # Get iteration count
    ksp = problem.solver
    iterations = ksp.getIterationNumber()
    
    # 8. Extract solution on 50x50 uniform grid
    n_eval = 50
    xs = np.linspace(0.0, 1.0, n_eval)
    ys = np.linspace(0.0, 1.0, n_eval)
    XX, YY = np.meshgrid(xs, ys, indexing='ij')
    
    points_2d = np.stack([XX.ravel(), YY.ravel()], axis=0)
    points_3d = np.vstack([points_2d, np.zeros((1, points_2d.shape[1]))])
    
    # Use geometry utilities for point evaluation
    bb_tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, points_3d.T)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points_3d.T)
    
    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    for i in range(points_3d.shape[1]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(points_3d.T[i])
            cells_on_proc.append(links[0])
            eval_map.append(i)
    
    u_values = np.full(points_3d.shape[1], np.nan)
    if len(points_on_proc) > 0:
        pts_arr = np.array(points_on_proc)
        cells_arr = np.array(cells_on_proc, dtype=np.int32)
        vals = uh.eval(pts_arr, cells_arr)
        u_values[eval_map] = vals.flatten()
    
    u_grid = u_values.reshape((n_eval, n_eval))
    
    return {
        "u": u_grid,
        "solver_info": {
            "mesh_resolution": nx,
            "element_degree": degree,
            "ksp_type": ksp_type,
            "pc_type": pc_type,
            "rtol": rtol,
            "iterations": int(iterations),
        }
    }