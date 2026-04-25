import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc


def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD
    N = 48
    degree = 2

    domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", degree))

    x = ufl.SpatialCoordinate(domain)
    u_exact = x[0] * (1 - x[0]) * x[1] * (1 - x[1])
    f_expr = 2 * (x[0] * (1 - x[0]) + x[1] * (1 - x[1]))

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    a = ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = f_expr * v * ufl.dx

    fdim = domain.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
    )
    dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    u_bc = fem.Function(V)
    u_bc.x.array[:] = 0.0
    bc = fem.dirichletbc(u_bc, dofs)

    rtol = 1e-10
    problem = petsc.LinearProblem(
        a, L, bcs=[bc],
        petsc_options={"ksp_type": "cg", "pc_type": "hypre", "ksp_rtol": rtol},
        petsc_options_prefix="poisson_"
    )
    u_sol = problem.solve()
    try:
        iters = problem.solver.getIterationNumber()
    except Exception:
        iters = 0

    # Sample on grid
    grid = case_spec["output"]["grid"]
    nx = grid["nx"]; ny = grid["ny"]
    bbox = grid["bbox"]
    xs = np.linspace(bbox[0], bbox[1], nx)
    ys = np.linspace(bbox[2], bbox[3], ny)
    XX, YY = np.meshgrid(xs, ys)
    pts = np.column_stack([XX.ravel(), YY.ravel(), np.zeros(nx * ny)])

    tree = geometry.bb_tree(domain, domain.topology.dim)
    cand = geometry.compute_collisions_points(tree, pts)
    coll = geometry.compute_colliding_cells(domain, cand, pts)

    points_on_proc = []
    cells_on_proc = []
    idx_map = []
    for i in range(pts.shape[0]):
        links = coll.links(i)
        if len(links) > 0:
            points_on_proc.append(pts[i])
            cells_on_proc.append(links[0])
            idx_map.append(i)

    u_vals = np.zeros(pts.shape[0])
    if points_on_proc:
        vals = u_sol.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u_vals[idx_map] = vals.flatten()

    u_grid = u_vals.reshape(ny, nx)

    return {
        "u": u_grid,
        "solver_info": {
            "mesh_resolution": N,
            "element_degree": degree,
            "ksp_type": "cg",
            "pc_type": "hypre",
            "rtol": rtol,
            "iterations": int(iters),
        }
    }


if __name__ == "__main__":
    import time
    spec = {"output": {"grid": {"nx": 64, "ny": 64, "bbox": [0, 1, 0, 1]}}}
    t0 = time.time()
    res = solve(spec)
    t1 = time.time()
    print(f"Wall time: {t1-t0:.3f}s")
    u = res["u"]
    nx, ny = 64, 64
    xs = np.linspace(0, 1, nx); ys = np.linspace(0, 1, ny)
    XX, YY = np.meshgrid(xs, ys)
    u_ex = XX * (1 - XX) * YY * (1 - YY)
    err = np.sqrt(np.mean((u - u_ex) ** 2))
    print(f"L2 error: {err:.3e}")
    print(f"Max error: {np.max(np.abs(u-u_ex)):.3e}")
    print(f"Iterations: {res['solver_info']['iterations']}")
