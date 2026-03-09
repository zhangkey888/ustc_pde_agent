import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc

def solve(case_spec: dict = None) -> dict:
    """Solve Poisson equation: -div(kappa * grad(u)) = f with Dirichlet BCs."""
    
    comm = MPI.COMM_WORLD
    
    if case_spec is None:
        case_spec = {}
    
    nx_out = 50
    ny_out = 50
    
    mesh_resolution = 64
    element_degree = 2
    ksp_type_str = "cg"
    pc_type_str = "hypre"
    rtol = 1e-10
    
    domain = mesh.create_unit_square(comm, mesh_resolution, mesh_resolution, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", element_degree))
    
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    x = ufl.SpatialCoordinate(domain)
    
    kappa = fem.Constant(domain, PETSc.ScalarType(1.0))
    f_expr = ufl.sin(8 * ufl.pi * x[0]) * ufl.sin(8 * ufl.pi * x[1])
    
    a = kappa * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = f_expr * v * ufl.dx
    
    tdim = domain.topology.dim
    fdim = tdim - 1
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
    )
    dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc = fem.dirichletbc(PETSc.ScalarType(0.0), dofs, V)
    
    problem = petsc.LinearProblem(
        a, L, bcs=[bc],
        petsc_options={
            "ksp_type": ksp_type_str,
            "pc_type": pc_type_str,
            "ksp_rtol": str(rtol),
            "ksp_max_it": "1000",
        },
        petsc_options_prefix="poisson_"
    )
    u_sol = problem.solve()
    iterations = problem.solver.getIterationNumber()
    
    x_pts = np.linspace(0, 1, nx_out)
    y_pts = np.linspace(0, 1, ny_out)
    xx, yy = np.meshgrid(x_pts, y_pts, indexing='ij')
    
    points_3d = np.zeros((3, nx_out * ny_out))
    points_3d[0, :] = xx.flatten()
    points_3d[1, :] = yy.flatten()
    
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
            "ksp_type": ksp_type_str,
            "pc_type": pc_type_str,
            "rtol": rtol,
            "iterations": int(iterations),
        }
    }


if __name__ == "__main__":
    import time
    t0 = time.time()
    result = solve()
    elapsed = time.time() - t0
    print(f"Solve time: {elapsed:.3f}s")
    print(f"Solution shape: {result['u'].shape}")
    print(f"Solution range: [{np.nanmin(result['u']):.6e}, {np.nanmax(result['u']):.6e}]")
    print(f"NaN count: {np.isnan(result['u']).sum()}")
    print(f"Solver info: {result['solver_info']}")
    
    x_pts = np.linspace(0, 1, 50)
    y_pts = np.linspace(0, 1, 50)
    xx, yy = np.meshgrid(x_pts, y_pts, indexing='ij')
    u_exact = np.sin(8*np.pi*xx) * np.sin(8*np.pi*yy) / (2 * (8*np.pi)**2)
    
    mask = ~np.isnan(result['u'])
    error = np.sqrt(np.mean((result['u'][mask] - u_exact[mask])**2))
    print(f"Approximate RMS grid error: {error:.6e}")
    print(f"Max absolute error: {np.max(np.abs(result['u'][mask] - u_exact[mask])):.6e}")
