import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc


def solve(case_spec: dict) -> dict:
    grid = case_spec["output"]["grid"]
    nx_out = grid["nx"]
    ny_out = grid["ny"]
    bbox = grid["bbox"]

    # High resolution for accuracy
    N = 120
    degree = 2

    comm = MPI.COMM_WORLD
    msh = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)

    # Standard Lagrange space
    V = fem.functionspace(msh, ("Lagrange", degree))

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    x = ufl.SpatialCoordinate(msh)

    # Exact solution u_ex = tanh(6*(y-0.5))*sin(pi*x)
    u_ex = ufl.tanh(6.0 * (x[1] - 0.5)) * ufl.sin(ufl.pi * x[0])
    
    # Compute Laplace of u_ex
    grad_u = ufl.grad(u_ex)
    lap_u_ex = ufl.div(grad_u)
    
    # f = Delta^2 u = Delta(Delta u)
    grad_lap_u = ufl.grad(lap_u_ex)
    f = ufl.div(grad_lap_u)

    # Two-step Poisson approach:
    # Step 1: Solve Delta w = f with w = Delta(g) on boundary
    # Step 2: Solve -Delta u = -w with u = g on boundary

    fdim = msh.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(
        msh, fdim, lambda xx: np.ones(xx.shape[1], dtype=bool)
    )

    # --- Step 1: Solve Delta w = f, w = lap_u_ex on boundary ---
    w = ufl.TrialFunction(V)
    q = ufl.TestFunction(V)
    
    a_w = ufl.inner(ufl.grad(w), ufl.grad(q)) * ufl.dx
    L_w = -f * q * ufl.dx

    # BC for w: w = lap_u_ex on boundary
    w_bc_func = fem.Function(V)
    w_bc_expr = fem.Expression(lap_u_ex, V.element.interpolation_points)
    w_bc_func.interpolate(w_bc_expr)
    dofs_w = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc_w = fem.dirichletbc(w_bc_func, dofs_w)

    petsc_options = {
        "ksp_type": "preonly",
        "pc_type": "lu",
        "pc_factor_mat_solver_type": "mumps",
    }
    problem_w = petsc.LinearProblem(a_w, L_w, bcs=[bc_w],
                                     petsc_options=petsc_options,
                                     petsc_options_prefix="w_")
    w_h = problem_w.solve()
    iterations_w = 1
    try:
        iterations_w = int(problem_w.solver.getIterationNumber())
    except Exception:
        pass

    # --- Step 2: Solve -Delta u = -w, u = u_ex on boundary ---
    a_u = ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
    L_u = -ufl.inner(w_h, v) * ufl.dx

    # BC for u: u = u_ex on boundary
    u_bc_func = fem.Function(V)
    u_bc_expr = fem.Expression(u_ex, V.element.interpolation_points)
    u_bc_func.interpolate(u_bc_expr)
    dofs_u = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc_u = fem.dirichletbc(u_bc_func, dofs_u)

    problem_u = petsc.LinearProblem(a_u, L_u, bcs=[bc_u],
                                     petsc_options=petsc_options,
                                     petsc_options_prefix="u_")
    u_h = problem_u.solve()
    iterations_u = 1
    try:
        iterations_u = int(problem_u.solver.getIterationNumber())
    except Exception:
        pass

    # Sample on output grid
    xs = np.linspace(bbox[0], bbox[1], nx_out)
    ys = np.linspace(bbox[2], bbox[3], ny_out)
    XX, YY = np.meshgrid(xs, ys)
    pts = np.column_stack([XX.ravel(), YY.ravel(), np.zeros(nx_out * ny_out)])

    tree = geometry.bb_tree(msh, msh.topology.dim)
    cell_candidates = geometry.compute_collisions_points(tree, pts)
    colliding = geometry.compute_colliding_cells(msh, cell_candidates, pts)

    u_grid = np.zeros(nx_out * ny_out)
    points_on_proc = []
    cells_on_proc = []
    idx_map = []
    for i in range(pts.shape[0]):
        links = colliding.links(i)
        if len(links) > 0:
            points_on_proc.append(pts[i])
            cells_on_proc.append(links[0])
            idx_map.append(i)
    if len(points_on_proc) > 0:
        vals = u_h.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u_grid[idx_map] = vals.flatten()
    u_grid = u_grid.reshape(ny_out, nx_out)

    # Clean up to prevent segfault
    del problem_w, problem_u, w_h, u_h

    return {
        "u": u_grid,
        "solver_info": {
            "mesh_resolution": N,
            "element_degree": degree,
            "ksp_type": "preonly",
            "pc_type": "lu",
            "rtol": 1e-10,
            "iterations": iterations_w + iterations_u,
        },
    }


if __name__ == "__main__":
    import time
    case_spec = {
        "output": {
            "grid": {"nx": 64, "ny": 64, "bbox": [0.0, 1.0, 0.0, 1.0]}
        }
    }
    t0 = time.time()
    result = solve(case_spec)
    elapsed = time.time() - t0

    u_grid = result["u"]
    ny_out, nx_out = u_grid.shape
    xs = np.linspace(0, 1, nx_out)
    ys = np.linspace(0, 1, ny_out)
    XX, YY = np.meshgrid(xs, ys)
    u_exact_grid = np.tanh(6.0 * (YY - 0.5)) * np.sin(np.pi * XX)
    err_l2 = np.sqrt(np.mean((u_grid - u_exact_grid) ** 2))
    err_max = np.max(np.abs(u_grid - u_exact_grid))
    print(f"Elapsed: {elapsed:.3f}s")
    print(f"L2 error: {err_l2:.6e}")
    print(f"Max error: {err_max:.6e}")
