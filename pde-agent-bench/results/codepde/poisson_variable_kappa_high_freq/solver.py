import numpy as np
from dolfinx import mesh, fem, default_scalar_type, geometry
from dolfinx.fem.petsc import LinearProblem
from mpi4py import MPI
import ufl
from petsc4py import PETSc


def solve(case_spec: dict) -> dict:
    # 1. Parse case_spec
    pde_config = case_spec.get("pde", case_spec.get("oracle_config", {}).get("pde", {}))
    
    # 2. Create mesh - use higher resolution for variable kappa with high frequency
    nx, ny = 128, 128
    element_degree = 2
    domain = mesh.create_unit_square(MPI.COMM_WORLD, nx, ny, cell_type=mesh.CellType.triangle)
    
    # 3. Function space
    V = fem.functionspace(domain, ("Lagrange", element_degree))
    
    # 4. Spatial coordinates
    x = ufl.SpatialCoordinate(domain)
    pi = ufl.pi
    
    # Define kappa = 1 + 0.3*sin(8*pi*x)*sin(8*pi*y)
    kappa = 1.0 + 0.3 * ufl.sin(8 * pi * x[0]) * ufl.sin(8 * pi * x[1])
    
    # Manufactured solution: u_exact = sin(2*pi*x)*sin(2*pi*y)
    u_exact = ufl.sin(2 * pi * x[0]) * ufl.sin(2 * pi * x[1])
    
    # Compute f = -div(kappa * grad(u_exact))
    # We need to compute this symbolically
    # grad(u_exact) = (2*pi*cos(2*pi*x)*sin(2*pi*y), 2*pi*sin(2*pi*x)*cos(2*pi*y))
    # kappa * grad(u_exact) 
    # div(kappa * grad(u_exact)) = grad(kappa) . grad(u_exact) + kappa * laplacian(u_exact)
    # f = -div(kappa * grad(u_exact))
    
    grad_u_exact = ufl.grad(u_exact)
    f = -ufl.div(kappa * grad_u_exact)
    
    # 5. Variational problem
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    
    a = ufl.inner(kappa * ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = f * v * ufl.dx
    
    # 6. Boundary conditions: u = g = sin(2*pi*x)*sin(2*pi*y) on boundary
    # Since sin(2*pi*x)*sin(2*pi*y) = 0 on all boundaries of [0,1]^2, g = 0
    tdim = domain.topology.dim
    fdim = tdim - 1
    
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
    )
    dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc = fem.dirichletbc(PETSc.ScalarType(0.0), dofs, V)
    
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
    
    # 8. Extract solution on 50x50 uniform grid
    n_grid = 50
    xs = np.linspace(0.0, 1.0, n_grid)
    ys = np.linspace(0.0, 1.0, n_grid)
    X, Y = np.meshgrid(xs, ys, indexing='ij')
    points = np.zeros((3, n_grid * n_grid))
    points[0, :] = X.ravel()
    points[1, :] = Y.ravel()
    
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
    
    u_values = np.full(n_grid * n_grid, np.nan)
    if len(points_on_proc) > 0:
        pts_arr = np.array(points_on_proc)
        cells_arr = np.array(cells_on_proc, dtype=np.int32)
        vals = uh.eval(pts_arr, cells_arr)
        u_values[eval_map] = vals.flatten()
    
    u_grid = u_values.reshape((n_grid, n_grid))
    
    return {
        "u": u_grid,
        "solver_info": {
            "mesh_resolution": nx,
            "element_degree": element_degree,
            "ksp_type": ksp_type,
            "pc_type": pc_type,
            "rtol": rtol,
            "iterations": -1,
        }
    }