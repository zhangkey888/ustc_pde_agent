import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc

def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD
    
    # Evaluation grid
    nx_eval = 50
    ny_eval = 50
    
    # Use degree 3 elements for high accuracy on exponential solution
    N = 48
    degree = 3
    
    # Create mesh
    domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
    
    # Function space
    V = fem.functionspace(domain, ("Lagrange", degree))
    
    # Spatial coordinates
    x = ufl.SpatialCoordinate(domain)
    
    # Exact solution as UFL expression
    pi = ufl.pi
    u_exact_ufl = ufl.exp(3.0 * (x[0] + x[1])) * ufl.sin(pi * x[0]) * ufl.sin(pi * x[1])
    
    # kappa
    kappa = fem.Constant(domain, PETSc.ScalarType(1.0))
    
    # Compute source term f = -div(kappa * grad(u_exact))
    f = -ufl.div(kappa * ufl.grad(u_exact_ufl))
    
    # Boundary condition
    u_bc_func = fem.Function(V)
    u_exact_expr = fem.Expression(u_exact_ufl, V.element.interpolation_points)
    u_bc_func.interpolate(u_exact_expr)
    
    # Mark all boundary
    tdim = domain.topology.dim
    fdim = tdim - 1
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
    )
    dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc = fem.dirichletbc(u_bc_func, dofs)
    
    # Variational form
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    
    a = ufl.inner(kappa * ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = ufl.inner(f, v) * ufl.dx
    
    # Solve
    ksp_type = "cg"
    pc_type = "hypre"
    rtol = 1e-12
    
    problem = petsc.LinearProblem(
        a, L, bcs=[bc],
        petsc_options={
            "ksp_type": ksp_type,
            "pc_type": pc_type,
            "ksp_rtol": str(rtol),
            "ksp_max_it": "2000",
        },
        petsc_options_prefix="poisson_"
    )
    u_sol = problem.solve()
    
    # Evaluate on 50x50 grid
    xs = np.linspace(0, 1, nx_eval)
    ys = np.linspace(0, 1, ny_eval)
    XX, YY = np.meshgrid(xs, ys, indexing='ij')
    
    points_2d = np.column_stack([XX.ravel(), YY.ravel()])
    # dolfinx needs 3D points
    points_3d = np.zeros((points_2d.shape[0], 3))
    points_3d[:, :2] = points_2d
    
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
        vals = u_sol.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u_values[eval_map] = vals.flatten()
    
    u_grid = u_values.reshape((nx_eval, ny_eval))
    
    solver_info = {
        "mesh_resolution": N,
        "element_degree": degree,
        "ksp_type": ksp_type,
        "pc_type": pc_type,
        "rtol": rtol,
        "iterations": -1,
    }
    
    return {
        "u": u_grid,
        "solver_info": solver_info,
    }


if __name__ == "__main__":
    import time
    t0 = time.time()
    result = solve({})
    elapsed = time.time() - t0
    
    u_grid = result["u"]
    print(f"Solution shape: {u_grid.shape}")
    print(f"Solution range: [{np.nanmin(u_grid):.6f}, {np.nanmax(u_grid):.6f}]")
    print(f"Any NaN: {np.any(np.isnan(u_grid))}")
    print(f"Time: {elapsed:.3f}s")
    
    # Compute error against exact solution
    xs = np.linspace(0, 1, 50)
    ys = np.linspace(0, 1, 50)
    XX, YY = np.meshgrid(xs, ys, indexing='ij')
    u_exact = np.exp(3*(XX+YY)) * np.sin(np.pi*XX) * np.sin(np.pi*YY)
    
    err = np.abs(u_grid - u_exact)
    l2_err = np.sqrt(np.mean(err**2))
    linf_err = np.max(err)
    rel_l2 = np.sqrt(np.mean(err**2)) / np.sqrt(np.mean(u_exact**2))
    print(f"L2 error (grid): {l2_err:.6e}")
    print(f"Linf error (grid): {linf_err:.6e}")
    print(f"Relative L2 error: {rel_l2:.6e}")
