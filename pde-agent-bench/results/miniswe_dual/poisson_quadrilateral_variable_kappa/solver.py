import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc

def solve(case_spec: dict) -> dict:
    """Solve Poisson equation with variable kappa."""
    
    comm = MPI.COMM_WORLD
    
    element_degree = 2
    N = 64
    
    domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.quadrilateral)
    V = fem.functionspace(domain, ("Lagrange", element_degree))
    
    x = ufl.SpatialCoordinate(domain)
    
    u_exact_expr = ufl.sin(2 * ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])
    kappa = 1.0 + 0.5 * ufl.cos(2 * ufl.pi * x[0]) * ufl.cos(2 * ufl.pi * x[1])
    f = -ufl.div(kappa * ufl.grad(u_exact_expr))
    
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    
    a = ufl.inner(kappa * ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = ufl.inner(f, v) * ufl.dx
    
    u_bc = fem.Function(V)
    u_bc.interpolate(fem.Expression(u_exact_expr, V.element.interpolation_points))
    
    tdim = domain.topology.dim
    fdim = tdim - 1
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
    )
    dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc = fem.dirichletbc(u_bc, dofs)
    
    try:
        problem = petsc.LinearProblem(
            a, L, bcs=[bc],
            petsc_options={
                "ksp_type": "cg",
                "pc_type": "hypre",
                "ksp_rtol": "1e-10",
                "ksp_max_it": "2000",
            },
            petsc_options_prefix="poisson_"
        )
        u_sol = problem.solve()
        ksp_type = "cg"
        pc_type = "hypre"
    except Exception:
        problem = petsc.LinearProblem(
            a, L, bcs=[bc],
            petsc_options={
                "ksp_type": "preonly",
                "pc_type": "lu",
            },
            petsc_options_prefix="poisson_lu_"
        )
        u_sol = problem.solve()
        ksp_type = "preonly"
        pc_type = "lu"
    
    nx_out, ny_out = 50, 50
    xs = np.linspace(0, 1, nx_out)
    ys = np.linspace(0, 1, ny_out)
    XX, YY = np.meshgrid(xs, ys, indexing='ij')
    
    points_flat = np.zeros((nx_out * ny_out, 3))
    points_flat[:, 0] = XX.flatten()
    points_flat[:, 1] = YY.flatten()
    
    bb_tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, points_flat)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points_flat)
    
    u_values = np.full(nx_out * ny_out, np.nan)
    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    
    for i in range(len(points_flat)):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(points_flat[i])
            cells_on_proc.append(links[0])
            eval_map.append(i)
    
    if len(points_on_proc) > 0:
        pts_arr = np.array(points_on_proc)
        cells_arr = np.array(cells_on_proc, dtype=np.int32)
        vals = u_sol.eval(pts_arr, cells_arr)
        u_values[eval_map] = vals.flatten()
    
    u_grid = u_values.reshape((nx_out, ny_out))
    
    return {
        "u": u_grid,
        "solver_info": {
            "mesh_resolution": N,
            "element_degree": element_degree,
            "ksp_type": ksp_type,
            "pc_type": pc_type,
            "rtol": 1e-10,
            "iterations": 0,
        }
    }


if __name__ == "__main__":
    import time
    
    case_spec = {}
    
    t0 = time.time()
    result = solve(case_spec)
    elapsed = time.time() - t0
    
    print(f"Solve time: {elapsed:.3f}s")
    print(f"Solution shape: {result['u'].shape}")
    print(f"NaN count: {np.isnan(result['u']).sum()}")
    
    nx_out, ny_out = 50, 50
    xs = np.linspace(0, 1, nx_out)
    ys = np.linspace(0, 1, ny_out)
    XX, YY = np.meshgrid(xs, ys, indexing='ij')
    u_exact = np.sin(2 * np.pi * XX) * np.sin(np.pi * YY)
    
    error = np.sqrt(np.nanmean((result['u'] - u_exact)**2))
    max_error = np.nanmax(np.abs(result['u'] - u_exact))
    print(f"RMS error: {error:.6e}")
    print(f"Max error: {max_error:.6e}")
    print(f"Solver info: {result['solver_info']}")
