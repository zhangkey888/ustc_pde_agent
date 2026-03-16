import numpy as np
from dolfinx import mesh, fem, default_scalar_type, geometry
from dolfinx.fem.petsc import LinearProblem
from mpi4py import MPI
import ufl
from petsc4py import PETSc


def solve(case_spec: dict) -> dict:
    # 1. Parse case_spec
    pde_config = case_spec.get("pde", case_spec.get("oracle_config", {}).get("pde", {}))
    kappa_val = 1.0
    
    # Try to extract kappa from case_spec
    coeffs = pde_config.get("coefficients", {})
    if "kappa" in coeffs:
        kappa_val = float(coeffs["kappa"])

    # 2. Create mesh - use higher resolution for the Gaussian bump
    # The Gaussian bump exp(-40*((x-0.5)^2+(y-0.5)^2)) has sharp gradients near center
    nx, ny = 128, 128
    element_degree = 2
    
    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_square(comm, nx, ny, cell_type=mesh.CellType.triangle)
    
    # 3. Function space
    V = fem.functionspace(domain, ("Lagrange", element_degree))
    
    # 4. Define exact solution and source term using UFL
    x = ufl.SpatialCoordinate(domain)
    
    # Exact solution: u_exact = exp(-40*((x-0.5)^2 + (y-0.5)^2))
    r2 = (x[0] - 0.5)**2 + (x[1] - 0.5)**2
    u_exact_ufl = ufl.exp(-40.0 * r2)
    
    # For -div(kappa * grad(u)) = f, compute f = -kappa * div(grad(u_exact))
    # = -kappa * laplacian(u_exact)
    # Let alpha = 40
    # u = exp(-alpha * r2)
    # grad(u) = -2*alpha*(x-0.5, y-0.5) * u
    # div(grad(u)) = (-2*alpha*2 + 4*alpha^2*r2*2 - 4*alpha^2*r2*2) ... let UFL handle it
    kappa = fem.Constant(domain, default_scalar_type(kappa_val))
    f_expr = -kappa * ufl.div(ufl.grad(u_exact_ufl))
    # Since we want -div(kappa*grad(u)) = f, and f = -div(kappa*grad(u_exact)):
    # f = -kappa * laplacian(u_exact)
    # Actually let's be more careful:
    # -div(kappa * grad(u)) = f
    # f = -div(kappa * grad(u_exact)) = -kappa * div(grad(u_exact))  [kappa constant]
    # So f_expr is correct as defined above (note the negative sign is already in f_expr)
    
    # 5. Variational problem
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    
    a = kappa * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = f_expr * v * ufl.dx
    
    # Note: f_expr = -kappa * div(grad(u_exact))
    # The weak form of -div(kappa*grad(u)) = f is:
    # kappa * inner(grad(u), grad(v)) dx = f * v dx
    # But f = -kappa * div(grad(u_exact)), so we need the negative:
    # Actually, -div(kappa*grad(u_exact)) IS our f. So:
    # L = integral of f * v dx = integral of (-kappa * div(grad(u_exact))) * v dx
    # This is correct as stated.
    
    # 6. Boundary conditions - Dirichlet from exact solution
    tdim = domain.topology.dim
    fdim = tdim - 1
    
    # Find all boundary facets
    domain.topology.create_connectivity(fdim, tdim)
    boundary_facets = mesh.exterior_facet_indices(domain.topology)
    
    # Create BC function by interpolating exact solution
    u_bc = fem.Function(V)
    
    # Create expression for exact solution
    u_exact_expr = fem.Expression(u_exact_ufl, V.element.interpolation_points)
    u_bc.interpolate(u_exact_expr)
    
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc = fem.dirichletbc(u_bc, boundary_dofs)
    
    # 7. Solve
    ksp_type = "cg"
    pc_type = "hypre"
    rtol = 1e-10
    
    problem = LinearProblem(
        a, L, bcs=[bc],
        petsc_options={
            "ksp_type": ksp_type,
            "pc_type": pc_type,
            "ksp_rtol": str(rtol),
            "ksp_max_it": "2000",
        },
        petsc_options_prefix="poisson_"
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
    
    points_2d = np.column_stack([XX.ravel(), YY.ravel()])
    # dolfinx expects (N, 3) for 2D meshes embedded in 3D
    points_3d = np.zeros((points_2d.shape[0], 3))
    points_3d[:, 0] = points_2d[:, 0]
    points_3d[:, 1] = points_2d[:, 1]
    
    # Point evaluation
    bb_tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, points_3d)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points_3d)
    
    u_values = np.full(points_3d.shape[0], np.nan)
    
    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    
    for i in range(points_3d.shape[0]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(points_3d[i])
            cells_on_proc.append(links[0])
            eval_map.append(i)
    
    if len(points_on_proc) > 0:
        pts_arr = np.array(points_on_proc)
        cells_arr = np.array(cells_on_proc, dtype=np.int32)
        vals = uh.eval(pts_arr, cells_arr)
        for idx, global_idx in enumerate(eval_map):
            u_values[global_idx] = vals[idx].flat[0]
    
    u_grid = u_values.reshape((nx_out, ny_out))
    
    return {
        "u": u_grid,
        "solver_info": {
            "mesh_resolution": nx,
            "element_degree": element_degree,
            "ksp_type": ksp_type,
            "pc_type": pc_type,
            "rtol": rtol,
            "iterations": int(iterations),
        }
    }