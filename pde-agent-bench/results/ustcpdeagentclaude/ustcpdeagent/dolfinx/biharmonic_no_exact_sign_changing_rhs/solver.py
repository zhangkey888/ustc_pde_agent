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

    comm = MPI.COMM_WORLD
    N = 192
    degree = 2
    domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)

    V = fem.functionspace(domain, ("Lagrange", degree))

    # Mixed formulation: -Δw = f, -Δu = w, with u=0 and w=0 on boundary (Navier BCs)
    fdim = domain.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
    )
    bdofs = fem.locate_dofs_topological(V, fdim, boundary_facets)

    zero = fem.Function(V)
    zero.x.array[:] = 0.0
    bc_w = fem.dirichletbc(zero, bdofs)

    zero_u = fem.Function(V)
    zero_u.x.array[:] = 0.0
    bc_u = fem.dirichletbc(zero_u, bdofs)

    x = ufl.SpatialCoordinate(domain)
    f_expr = ufl.cos(4 * ufl.pi * x[0]) * ufl.sin(3 * ufl.pi * x[1])

    # Solve 1: -Δw = f, w=0 on boundary
    w_tr = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    a1 = ufl.inner(ufl.grad(w_tr), ufl.grad(v)) * ufl.dx
    L1 = f_expr * v * ufl.dx

    total_iters = 0
    problem1 = petsc.LinearProblem(
        a1, L1, bcs=[bc_w],
        petsc_options={"ksp_type": "cg", "pc_type": "hypre",
                       "ksp_rtol": 1e-10, "ksp_atol": 1e-14},
        petsc_options_prefix="bih1_"
    )
    w_sol = problem1.solve()
    try:
        total_iters += problem1.solver.getIterationNumber()
    except Exception:
        pass

    # Solve 2: -Δu = w, u=0 on boundary
    u_tr = ufl.TrialFunction(V)
    a2 = ufl.inner(ufl.grad(u_tr), ufl.grad(v)) * ufl.dx
    L2 = w_sol * v * ufl.dx

    problem2 = petsc.LinearProblem(
        a2, L2, bcs=[bc_u],
        petsc_options={"ksp_type": "cg", "pc_type": "hypre",
                       "ksp_rtol": 1e-10, "ksp_atol": 1e-14},
        petsc_options_prefix="bih2_"
    )
    u_sol = problem2.solve()
    try:
        total_iters += problem2.solver.getIterationNumber()
    except Exception:
        pass

    # Sample on output grid
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

    u_values = np.full(pts.shape[0], np.nan)
    if len(points_on_proc) > 0:
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
            "rtol": 1e-10,
            "iterations": int(total_iters),
        }
    }


if __name__ == "__main__":
    import time
    spec = {"output": {"grid": {"nx": 128, "ny": 128, "bbox": [0, 1, 0, 1]}}}
    t0 = time.time()
    result = solve(spec)
    t1 = time.time()
    print(f"Time: {t1-t0:.2f}s")
    print(f"u shape: {result['u'].shape}")
    print(f"u range: [{result['u'].min():.4e}, {result['u'].max():.4e}]")
    print(f"solver_info: {result['solver_info']}")

    # Analytical solution via Fourier: f = cos(4πx)sin(3πy). But BCs: u=0 on ∂Ω.
    # Use Fourier sine expansion of f on [0,1]² with u=0 boundary.
    # Simpler check: compare with higher resolution

def reference_solution(nx=128, ny=128):
    # f = cos(4πx) sin(3πy), BCs u=0 on boundary with biharmonic equation assuming Navier BC
    # Actually the problem states only u=0 on boundary. The "mixed formulation" with u=0, Δu=0 is Navier.
    # Compute u via Fourier sine series, assuming Navier BCs.
    xs = np.linspace(0, 1, nx)
    ys = np.linspace(0, 1, ny)
    X, Y = np.meshgrid(xs, ys)
    # a_m = 2 int_0^1 cos(4πx) sin(mπx) dx
    # = 2 * m*(1-cos(mπ)cos(4π))/... actually let me compute numerically
    u = np.zeros_like(X)
    M = 200
    for m in range(1, M+1):
        # a_m = 2 ∫ cos(4πx) sin(mπx) dx from 0 to 1
        # using identity: sin(mπx)cos(4πx) = 0.5[sin((m+4)πx) + sin((m-4)πx)]
        # ∫₀¹ sin(kπx) dx = (1-cos(kπ))/(kπ) for k≠0; =0 for k=0
        def integ(k):
            if k == 0: return 0.0
            return (1 - np.cos(k*np.pi)) / (k*np.pi)
        a_m = 2 * 0.5 * (integ(m+4) + integ(m-4))
        # Biharmonic eigenvalue for sin(mπx)sin(3πy): ((mπ)² + (3π)²)²
        lam = ((m*np.pi)**2 + (3*np.pi)**2)**2
        u += (a_m / lam) * np.sin(m*np.pi*X) * np.sin(3*np.pi*Y)
    return u

if __name__ == "__main__":
    ref = reference_solution(128, 128)
    import time
    spec = {"output": {"grid": {"nx": 128, "ny": 128, "bbox": [0, 1, 0, 1]}}}
    res = solve(spec)
    err = np.sqrt(np.mean((res["u"] - ref)**2))
    print(f"RMSE vs reference: {err:.4e}")
    print(f"Max abs error: {np.max(np.abs(res['u'] - ref)):.4e}")
    print(f"Ref range: [{ref.min():.4e}, {ref.max():.4e}]")
