import numpy as np
from dolfinx import mesh, fem, default_scalar_type, geometry
from dolfinx.fem.petsc import LinearProblem
from mpi4py import MPI
import ufl
from petsc4py import PETSc


def solve(case_spec: dict) -> dict:
    # 1. Parse case_spec
    pde_config = case_spec.get("oracle_config", {}).get("pde", {})
    
    # 2. Create mesh - use higher resolution for multi-frequency solution
    nx, ny = 80, 80
    domain = mesh.create_unit_square(MPI.COMM_WORLD, nx, ny, cell_type=mesh.CellType.triangle)
    
    # 3. Function space - P2 elements for better accuracy with multi-frequency
    degree = 2
    V = fem.functionspace(domain, ("Lagrange", degree))
    
    # 4. Spatial coordinates and exact solution
    x = ufl.SpatialCoordinate(domain)
    pi = ufl.pi
    
    # Manufactured solution: u = sin(pi*x)*sin(pi*y) + 0.2*sin(5*pi*x)*sin(4*pi*y)
    u_exact = ufl.sin(pi * x[0]) * ufl.sin(pi * x[1]) + 0.2 * ufl.sin(5 * pi * x[0]) * ufl.sin(4 * pi * x[1])
    
    # kappa = 1.0
    kappa = fem.Constant(domain, default_scalar_type(1.0))
    
    # Source term: f = -div(kappa * grad(u_exact))
    # For kappa=1: f = -laplacian(u_exact)
    # laplacian of sin(pi*x)*sin(pi*y) = -2*pi^2 * sin(pi*x)*sin(pi*y)
    # laplacian of 0.2*sin(5*pi*x)*sin(4*pi*y) = -0.2*(25+16)*pi^2 * sin(5*pi*x)*sin(4*pi*y)
    # So f = 2*pi^2*sin(pi*x)*sin(pi*y) + 0.2*41*pi^2*sin(5*pi*x)*sin(4*pi*y)
    f = -ufl.div(kappa * ufl.grad(u_exact))
    
    # 5. Variational problem
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    
    a = kappa * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = f * v * ufl.dx
    
    # 6. Boundary conditions - u = g on boundary (g = u_exact)
    # Since u_exact = sin(pi*x)*sin(pi*y) + 0.2*sin(5*pi*x)*sin(4*pi*y),
    # on the boundary of [0,1]^2, all sin terms vanish, so g = 0
    tdim = domain.topology.dim
    fdim = tdim - 1
    
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
    )
    
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    
    # g = 0 on boundary
    u_bc = fem.Function(V)
    u_bc.x.array[:] = 0.0
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
            "ksp_max_it": "1000",
        },
        petsc_options_prefix="poisson_"
    )
    uh = problem.solve()
    
    # Get iteration count
    iterations = problem.solver.getIterationNumber()
    
    # 8. Extract solution on 50x50 uniform grid
    n_eval = 50
    xs = np.linspace(0.0, 1.0, n_eval)
    ys = np.linspace(0.0, 1.0, n_eval)
    XX, YY = np.meshgrid(xs, ys, indexing='ij')
    
    points = np.zeros((3, n_eval * n_eval))
    points[0, :] = XX.ravel()
    points[1, :] = YY.ravel()
    points[2, :] = 0.0
    
    # Point evaluation
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
    
    u_values = np.full(n_eval * n_eval, np.nan)
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
            "iterations": iterations,
        }
    }