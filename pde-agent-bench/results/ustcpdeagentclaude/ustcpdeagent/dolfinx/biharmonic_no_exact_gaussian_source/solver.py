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

    N = 192
    degree = 3
    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)

    V = fem.functionspace(domain, ("Lagrange", degree))

    x = ufl.SpatialCoordinate(domain)
    f_expr = 10.0 * ufl.exp(-80.0 * ((x[0] - 0.35)**2 + (x[1] - 0.55)**2))

    # Boundary
    fdim = domain.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
    )
    bdofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    u_bc = fem.Function(V)
    u_bc.x.array[:] = 0.0
    bc = fem.dirichletbc(u_bc, bdofs)

    # Step 1: solve -Δw = f with w = 0 on ∂Ω (simply supported)
    # Then: -Δu = w with u = 0 on ∂Ω
    # This gives Δ²u = -Δw = f. ✓
    w_tr = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    a1 = ufl.inner(ufl.grad(w_tr), ufl.grad(v)) * ufl.dx
    L1 = ufl.inner(f_expr, v) * ufl.dx

    problem1 = petsc.LinearProblem(
        a1, L1, bcs=[bc],
        petsc_options={"ksp_type": "cg", "pc_type": "hypre", "ksp_rtol": 1e-12, "ksp_atol": 1e-14},
        petsc_options_prefix="bih1_"
    )
    w_sol = problem1.solve()
    its1 = problem1.solver.getIterationNumber()

    # Step 2
    u_tr = ufl.TrialFunction(V)
    a2 = ufl.inner(ufl.grad(u_tr), ufl.grad(v)) * ufl.dx
    L2 = ufl.inner(w_sol, v) * ufl.dx

    problem2 = petsc.LinearProblem(
        a2, L2, bcs=[bc],
        petsc_options={"ksp_type": "cg", "pc_type": "hypre", "ksp_rtol": 1e-12, "ksp_atol": 1e-14},
        petsc_options_prefix="bih2_"
    )
    u_sol = problem2.solve()
    its2 = problem2.solver.getIterationNumber()

    # Sample on grid
    xs = np.linspace(bbox[0], bbox[1], nx_out)
    ys = np.linspace(bbox[2], bbox[3], ny_out)
    XX, YY = np.meshgrid(xs, ys)
    pts = np.c_[XX.ravel(), YY.ravel(), np.zeros(nx_out * ny_out)]

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
    if points_on_proc:
        vals = u_sol.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u_values[eval_map] = vals.flatten()

    u_grid = u_values.reshape(ny_out, nx_out)

    return {
        "u": u_grid,
        "solver_info": {
            "mesh_resolution": N,
            "element_degree": degree,
            "ksp_type": "cg",
            "pc_type": "hypre",
            "rtol": 1e-12,
            "iterations": int(its1 + its2),
        }
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
    t1 = time.time()
    print(f"Wall time: {t1-t0:.3f}s")
    print(f"u shape: {result['u'].shape}")
    print(f"u min/max: {result['u'].min():.6e} / {result['u'].max():.6e}")
    print(f"solver_info: {result['solver_info']}")
