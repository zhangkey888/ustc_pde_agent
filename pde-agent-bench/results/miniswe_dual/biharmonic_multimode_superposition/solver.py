import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc
import time


def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD

    element_degree = 2
    N = 64
    ksp_type_str = "preonly"
    pc_type_str = "lu"
    rtol = 1e-10

    total_iterations = 0

    domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", element_degree))

    x = ufl.SpatialCoordinate(domain)
    pi_val = np.pi

    f_expr = (4.0 * pi_val**4 * ufl.sin(pi_val * x[0]) * ufl.sin(pi_val * x[1])
              + 84.5 * pi_val**4 * ufl.sin(2.0 * pi_val * x[0]) * ufl.sin(3.0 * pi_val * x[1]))

    tdim = domain.topology.dim
    fdim = tdim - 1
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
    )
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc_zero = fem.dirichletbc(PETSc.ScalarType(0.0), boundary_dofs, V)

    v_test = ufl.TestFunction(V)

    # Step 1: Solve -Lap(w) = f, w=0 on boundary
    w_trial = ufl.TrialFunction(V)
    a1 = ufl.inner(ufl.grad(w_trial), ufl.grad(v_test)) * ufl.dx
    L1 = ufl.inner(f_expr, v_test) * ufl.dx

    problem1 = petsc.LinearProblem(
        a1, L1, bcs=[bc_zero],
        petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
        petsc_options_prefix="bih1_"
    )
    w_sol = problem1.solve()

    try:
        total_iterations += problem1.solver.getIterationNumber()
    except Exception:
        pass

    # Step 2: Solve -Lap(u) = w, u=0 on boundary
    u_trial = ufl.TrialFunction(V)
    a2 = ufl.inner(ufl.grad(u_trial), ufl.grad(v_test)) * ufl.dx
    L2 = ufl.inner(w_sol, v_test) * ufl.dx

    problem2 = petsc.LinearProblem(
        a2, L2, bcs=[bc_zero],
        petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
        petsc_options_prefix="bih2_"
    )
    u_sol = problem2.solve()

    try:
        total_iterations += problem2.solver.getIterationNumber()
    except Exception:
        pass

    # Evaluate on 50x50 grid
    nx_out, ny_out = 50, 50
    xs = np.linspace(0.0, 1.0, nx_out)
    ys = np.linspace(0.0, 1.0, ny_out)
    XX, YY = np.meshgrid(xs, ys, indexing="ij")

    points_3d = np.zeros((nx_out * ny_out, 3))
    points_3d[:, 0] = XX.ravel()
    points_3d[:, 1] = YY.ravel()

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
        vals = u_sol.eval(pts_arr, cells_arr)
        u_values[eval_map] = vals.flatten()

    u_grid = u_values.reshape((nx_out, ny_out))

    return {
        "u": u_grid,
        "solver_info": {
            "mesh_resolution": N,
            "element_degree": element_degree,
            "ksp_type": ksp_type_str,
            "pc_type": pc_type_str,
            "rtol": rtol,
            "iterations": max(total_iterations, 2),
        },
    }


if __name__ == "__main__":
    t0 = time.time()
    result = solve({})
    elapsed = time.time() - t0

    u_grid = result["u"]
    print(f"Shape: {u_grid.shape}")
    print(f"Range: [{np.nanmin(u_grid):.6e}, {np.nanmax(u_grid):.6e}]")
    print(f"NaN count: {np.isnan(u_grid).sum()}")
    print(f"Wall time: {elapsed:.3f}s")
    print(f"Solver info: {result['solver_info']}")

    xs = np.linspace(0, 1, 50)
    ys = np.linspace(0, 1, 50)
    XX, YY = np.meshgrid(xs, ys, indexing="ij")
    u_exact = (np.sin(np.pi * XX) * np.sin(np.pi * YY)
               + 0.5 * np.sin(2 * np.pi * XX) * np.sin(3 * np.pi * YY))

    rms = np.sqrt(np.nanmean((u_grid - u_exact) ** 2))
    mx = np.nanmax(np.abs(u_grid - u_exact))
    print(f"RMS error: {rms:.6e}")
    print(f"Max error: {mx:.6e}")
