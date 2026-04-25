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

    N = 128
    degree = 3
    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)

    V = fem.functionspace(domain, ("Lagrange", degree))

    x = ufl.SpatialCoordinate(domain)
    u_exact_ufl = ufl.sin(3 * ufl.pi * x[0]) * ufl.sin(2 * ufl.pi * x[1])
    # f = Δ²u = (9π² + 4π²)² u = (13 π²)² u
    lam = (9.0 + 4.0) * np.pi**2  # = -Δu / u constant factor
    f_ufl = (lam ** 2) * u_exact_ufl  # f = Δ²u
    w_bc_ufl = lam * u_exact_ufl  # w = -Δu on boundary (also zero here)

    # BCs: u = 0 on boundary; w = -Δu = 13π² u = 0 on boundary
    fdim = domain.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda X: np.ones(X.shape[1], dtype=bool)
    )
    dofs_V = fem.locate_dofs_topological(V, fdim, boundary_facets)

    u_bc = fem.Function(V)
    u_bc.x.array[:] = 0.0
    bc_u = fem.dirichletbc(u_bc, dofs_V)

    w_bc = fem.Function(V)
    w_bc.x.array[:] = 0.0
    bc_w = fem.dirichletbc(w_bc, dofs_V)

    total_iters = 0

    # Step 1: solve -Δw = f with w=0 on boundary
    u_tr = ufl.TrialFunction(V)
    v_te = ufl.TestFunction(V)
    a1 = ufl.inner(ufl.grad(u_tr), ufl.grad(v_te)) * ufl.dx
    L1 = ufl.inner(f_ufl, v_te) * ufl.dx

    problem1 = petsc.LinearProblem(
        a1, L1, bcs=[bc_w],
        petsc_options={"ksp_type": "cg", "pc_type": "hypre", "ksp_rtol": 1e-12, "ksp_atol": 1e-14},
        petsc_options_prefix="bih1_"
    )
    w_sol = problem1.solve()
    try:
        total_iters += problem1.solver.getIterationNumber()
    except Exception:
        pass

    # Step 2: solve -Δu = w with u=0 on boundary
    a2 = ufl.inner(ufl.grad(u_tr), ufl.grad(v_te)) * ufl.dx
    L2 = ufl.inner(w_sol, v_te) * ufl.dx

    problem2 = petsc.LinearProblem(
        a2, L2, bcs=[bc_u],
        petsc_options={"ksp_type": "cg", "pc_type": "hypre", "ksp_rtol": 1e-12, "ksp_atol": 1e-14},
        petsc_options_prefix="bih2_"
    )
    u_sol = problem2.solve()
    try:
        total_iters += problem2.solver.getIterationNumber()
    except Exception:
        pass

    # Sample on the output grid
    xs = np.linspace(bbox[0], bbox[1], nx_out)
    ys = np.linspace(bbox[2], bbox[3], ny_out)
    XX, YY = np.meshgrid(xs, ys)
    pts = np.column_stack([XX.ravel(), YY.ravel(), np.zeros(nx_out * ny_out)])

    tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(tree, pts)
    colliding = geometry.compute_colliding_cells(domain, cell_candidates, pts)

    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    for i in range(pts.shape[0]):
        links = colliding.links(i)
        if len(links) > 0:
            points_on_proc.append(pts[i])
            cells_on_proc.append(links[0])
            eval_map.append(i)

    u_values = np.zeros(pts.shape[0])
    if len(points_on_proc) > 0:
        vals = u_sol.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u_values[eval_map] = vals.flatten()

    u_grid = u_values.reshape(ny_out, nx_out)

    # Verify accuracy
    u_ex = np.sin(3 * np.pi * XX) * np.sin(2 * np.pi * YY)
    err = float(np.sqrt(np.mean((u_grid - u_ex) ** 2)))
    print(f"[solver] RMSE vs exact = {err:.3e}, iters = {total_iters}")

    return {
        "u": u_grid,
        "solver_info": {
            "mesh_resolution": N,
            "element_degree": degree,
            "ksp_type": "cg",
            "pc_type": "hypre",
            "rtol": 1e-12,
            "iterations": total_iters,
        },
    }


if __name__ == "__main__":
    import time
    spec = {"output": {"grid": {"nx": 64, "ny": 64, "bbox": [0.0, 1.0, 0.0, 1.0]}}}
    t0 = time.time()
    out = solve(spec)
    print(f"Wall time: {time.time()-t0:.2f}s, shape={out['u'].shape}")
