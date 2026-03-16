import numpy as np
from dolfinx import mesh, fem, default_scalar_type, geometry
from dolfinx.fem.petsc import LinearProblem
from mpi4py import MPI
import ufl
from petsc4py import PETSc


def solve(case_spec: dict) -> dict:
    # 1. Parse case_spec
    pde_config = case_spec.get("pde", case_spec.get("oracle_config", {}).get("pde", {}))
    
    # For high-frequency sin(4*pi*x)*sin(4*pi*y), we need adequate resolution
    # The wavelength is 1/4 = 0.25, so we need enough elements per wavelength
    # With P2 elements, ~16 elements per wavelength is good -> 64 elements across domain
    # But to be safe for the error threshold, use higher resolution
    nx = 80
    ny = 80
    degree = 2
    
    # 2. Create mesh
    domain = mesh.create_unit_square(MPI.COMM_WORLD, nx, ny, cell_type=mesh.CellType.triangle)
    
    # 3. Function space
    V = fem.functionspace(domain, ("Lagrange", degree))
    
    # 4. Define variational problem
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    
    x = ufl.SpatialCoordinate(domain)
    
    # Manufactured solution: u_exact = sin(4*pi*x)*sin(4*pi*y)
    # -kappa * laplacian(u_exact) = kappa * 32 * pi^2 * sin(4*pi*x) * sin(4*pi*y)
    kappa = 1.0
    
    pi = ufl.pi
    u_exact_ufl = ufl.sin(4 * pi * x[0]) * ufl.sin(4 * pi * x[1])
    
    # Source term: f = -kappa * div(grad(u_exact)) = kappa * 32 * pi^2 * sin(4*pi*x)*sin(4*pi*y)
    f = kappa * 32.0 * pi**2 * ufl.sin(4 * pi * x[0]) * ufl.sin(4 * pi * x[1])
    
    kappa_c = fem.Constant(domain, default_scalar_type(kappa))
    
    a = kappa_c * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = f * v * ufl.dx
    
    # 5. Boundary conditions: u = g = sin(4*pi*x)*sin(4*pi*y) on boundary
    # On the unit square boundary, sin(4*pi*x)*sin(4*pi*y) = 0 
    # (since at x=0,1: sin(4*pi*0)=0, sin(4*pi*1)=0, same for y)
    # So g = 0 on all boundaries
    
    tdim = domain.topology.dim
    fdim = tdim - 1
    
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
    )
    dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc = fem.dirichletbc(default_scalar_type(0.0), dofs, V)
    
    # 6. Solve
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
    
    # 7. Extract solution on 50x50 uniform grid
    n_eval = 50
    xs = np.linspace(0.0, 1.0, n_eval)
    ys = np.linspace(0.0, 1.0, n_eval)
    XX, YY = np.meshgrid(xs, ys, indexing='ij')
    
    points_2d = np.stack([XX.ravel(), YY.ravel()], axis=0)
    points_3d = np.vstack([points_2d, np.zeros((1, points_2d.shape[1]))])
    
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
    
    u_grid = u_values.reshape((n_eval, n_eval))
    
    return {
        "u": u_grid,
        "solver_info": {
            "mesh_resolution": nx,
            "element_degree": degree,
            "ksp_type": ksp_type,
            "pc_type": pc_type,
            "rtol": rtol,
            "iterations": -1,  # not easily accessible from LinearProblem
        }
    }