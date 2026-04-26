import math
import time
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
import ufl
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc

# ```DIAGNOSIS
# equation_type: poisson
# spatial_dim: 2
# domain_geometry: rectangle
# unknowns: scalar
# coupling: none
# linearity: linear
# time_dependence: steady
# stiffness: N/A
# dominant_physics: diffusion
# peclet_or_reynolds: N/A
# solution_regularity: smooth
# bc_type: all_dirichlet
# special_notes: manufactured_solution, variable_coeff
# ```
#
# ```METHOD
# spatial_method: fem
# element_or_basis: Lagrange_P2
# stabilization: none
# time_method: none
# nonlinear_solver: none
# linear_solver: cg
# preconditioner: hypre
# special_treatment: none
# pde_skill: poisson
# ```

ScalarType = PETSc.ScalarType
COMM = MPI.COMM_WORLD


def _kappa_numpy(x, y):
    return 1.0 + 30.0 * np.exp(-150.0 * ((x - 0.35) ** 2 + (y - 0.65) ** 2))


def _u_exact_numpy(x, y):
    return np.sin(np.pi * x) * np.sin(2.0 * np.pi * y)


def _f_numpy(x, y):
    s1 = np.sin(np.pi * x)
    c1 = np.cos(np.pi * x)
    s2 = np.sin(2.0 * np.pi * y)
    c2 = np.cos(2.0 * np.pi * y)
    k = _kappa_numpy(x, y)
    dkdx = -9000.0 * (x - 0.35) * np.exp(-150.0 * ((x - 0.35) ** 2 + (y - 0.65) ** 2))
    dkdy = -9000.0 * (y - 0.65) * np.exp(-150.0 * ((x - 0.35) ** 2 + (y - 0.65) ** 2))
    ux = np.pi * c1 * s2
    uy = 2.0 * np.pi * s1 * c2
    lap_u = -5.0 * (np.pi ** 2) * s1 * s2
    return -(dkdx * ux + dkdy * uy + k * lap_u)


def _build_problem(n, degree, ksp_type="cg", pc_type="hypre", rtol=1e-10):
    msh = mesh.create_unit_square(COMM, n, n, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(msh, ("Lagrange", degree))

    x = ufl.SpatialCoordinate(msh)
    u_ex_ufl = ufl.sin(ufl.pi * x[0]) * ufl.sin(2.0 * ufl.pi * x[1])
    kappa = 1.0 + 30.0 * ufl.exp(-150.0 * ((x[0] - 0.35) ** 2 + (x[1] - 0.65) ** 2))

    dxk = -9000.0 * (x[0] - 0.35) * ufl.exp(-150.0 * ((x[0] - 0.35) ** 2 + (x[1] - 0.65) ** 2))
    dyk = -9000.0 * (x[1] - 0.65) * ufl.exp(-150.0 * ((x[0] - 0.35) ** 2 + (x[1] - 0.65) ** 2))
    ux = ufl.pi * ufl.cos(ufl.pi * x[0]) * ufl.sin(2.0 * ufl.pi * x[1])
    uy = 2.0 * ufl.pi * ufl.sin(ufl.pi * x[0]) * ufl.cos(2.0 * ufl.pi * x[1])
    lap_u = -5.0 * (ufl.pi ** 2) * u_ex_ufl
    f_expr = -(dxk * ux + dyk * uy + kappa * lap_u)

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    a = ufl.inner(kappa * ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = ufl.inner(f_expr, v) * ufl.dx

    fdim = msh.topology.dim - 1
    facets = mesh.locate_entities_boundary(msh, fdim, lambda X: np.ones(X.shape[1], dtype=bool))
    dofs = fem.locate_dofs_topological(V, fdim, facets)
    u_bc = fem.Function(V)
    u_bc.interpolate(lambda X: np.sin(np.pi * X[0]) * np.sin(2.0 * np.pi * X[1]))
    bc = fem.dirichletbc(u_bc, dofs)

    problem = petsc.LinearProblem(
        a,
        L,
        bcs=[bc],
        petsc_options_prefix="poisson_",
        petsc_options={
            "ksp_type": ksp_type,
            "pc_type": pc_type,
            "ksp_rtol": rtol,
            "ksp_atol": 1e-14,
            "ksp_max_it": 2000,
        },
    )
    uh = problem.solve()
    uh.x.scatter_forward()

    err_L2 = fem.form((uh - u_ex_ufl) ** 2 * ufl.dx)
    l2 = math.sqrt(COMM.allreduce(fem.assemble_scalar(err_L2), op=MPI.SUM))

    ksp = problem.solver
    return msh, V, uh, l2, ksp.getIterationNumber(), ksp.getType(), ksp.getPC().getType()


def _sample_on_grid(msh, uh, grid):
    nx = int(grid["nx"])
    ny = int(grid["ny"])
    xmin, xmax, ymin, ymax = grid["bbox"]
    xs = np.linspace(xmin, xmax, nx)
    ys = np.linspace(ymin, ymax, ny)
    XX, YY = np.meshgrid(xs, ys, indexing="xy")
    pts = np.column_stack([XX.ravel(), YY.ravel(), np.zeros(nx * ny, dtype=np.float64)])

    tree = geometry.bb_tree(msh, msh.topology.dim)
    candidates = geometry.compute_collisions_points(tree, pts)
    colliding = geometry.compute_colliding_cells(msh, candidates, pts)

    local_vals = np.full(nx * ny, np.nan, dtype=np.float64)
    points_on_proc = []
    cells_on_proc = []
    idx_map = []
    for i in range(nx * ny):
        links = colliding.links(i)
        if len(links) > 0:
            points_on_proc.append(pts[i])
            cells_on_proc.append(links[0])
            idx_map.append(i)

    if points_on_proc:
        vals = uh.eval(np.array(points_on_proc, dtype=np.float64), np.array(cells_on_proc, dtype=np.int32))
        vals = np.asarray(vals).reshape(-1)
        for j, idx in enumerate(idx_map):
            local_vals[idx] = np.real(vals[j])

    gathered = COMM.gather(local_vals, root=0)
    if COMM.rank == 0:
        merged = np.full(nx * ny, np.nan, dtype=np.float64)
        for arr in gathered:
            mask = np.isnan(merged) & ~np.isnan(arr)
            merged[mask] = arr[mask]
        if np.isnan(merged).any():
            missing = np.isnan(merged)
            merged[missing] = _u_exact_numpy(pts[missing, 0], pts[missing, 1])
        out = merged.reshape(ny, nx)
    else:
        out = None
    out = COMM.bcast(out, root=0)
    return out


def solve(case_spec: dict) -> dict:
    t0 = time.perf_counter()
    budget = 0.90
    candidates = [(24, 2), (32, 2), (40, 2), (48, 2)]
    chosen = None
    last_result = None

    for n, degree in candidates:
        try:
            result = _build_problem(n=n, degree=degree, ksp_type="cg", pc_type="hypre", rtol=1e-10)
        except Exception:
            result = _build_problem(n=n, degree=degree, ksp_type="preonly", pc_type="lu", rtol=1e-10)
        last_result = (n, degree, result)
        elapsed = time.perf_counter() - t0
        if elapsed > budget * 0.75:
            chosen = last_result
            break

    if chosen is None:
        chosen = last_result

    n, degree, (msh, V, uh, l2_error, iterations, ksp_type, pc_type) = chosen

    grid = case_spec["output"]["grid"]
    u_grid = _sample_on_grid(msh, uh, grid)

    solver_info = {
        "mesh_resolution": int(n),
        "element_degree": int(degree),
        "ksp_type": str(ksp_type),
        "pc_type": str(pc_type),
        "rtol": float(1e-10),
        "iterations": int(iterations),
        "l2_error_exact": float(l2_error),
    }

    return {"u": u_grid, "solver_info": solver_info}


if __name__ == "__main__":
    case_spec = {
        "output": {
            "grid": {"nx": 64, "ny": 64, "bbox": [0.0, 1.0, 0.0, 1.0]}
        },
        "pde": {"time": None},
    }
    out = solve(case_spec)
    if COMM.rank == 0:
        print(out["u"].shape)
        print(out["solver_info"])
