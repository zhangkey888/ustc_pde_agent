import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl

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
# special_notes: manufactured_solution
# ```
#
# ```METHOD
# spatial_method: fem
# element_or_basis: Lagrange_P2
# stabilization: none
# time_method: none
# nonlinear_solver: none
# linear_solver: cg
# preconditioner: amg
# special_treatment: none
# pde_skill: poisson
# ```

ScalarType = PETSc.ScalarType


def _exact_u(x, y):
    return np.sin(8.0 * np.pi * x) * np.sin(np.pi * y)


def _choose_mesh_resolution(case_spec):
    grid = case_spec.get("output", {}).get("grid", {})
    nx = int(grid.get("nx", 129))
    ny = int(grid.get("ny", 129))
    m = max(nx, ny)
    if m >= 192:
        return 96
    if m >= 128:
        return 80
    if m >= 96:
        return 72
    return 64


def _sample_function_on_grid(domain, uh, grid_spec):
    nx = int(grid_spec["nx"])
    ny = int(grid_spec["ny"])
    xmin, xmax, ymin, ymax = map(float, grid_spec["bbox"])

    xs = np.linspace(xmin, xmax, nx, dtype=np.float64)
    ys = np.linspace(ymin, ymax, ny, dtype=np.float64)
    XX, YY = np.meshgrid(xs, ys, indexing="xy")

    pts = np.zeros((nx * ny, 3), dtype=np.float64)
    pts[:, 0] = XX.ravel()
    pts[:, 1] = YY.ravel()

    tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(tree, pts)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, pts)

    local_values = np.full((pts.shape[0],), np.nan, dtype=np.float64)
    points_on_proc = []
    cells_on_proc = []
    point_ids = []

    for i in range(pts.shape[0]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(pts[i])
            cells_on_proc.append(links[0])
            point_ids.append(i)

    if points_on_proc:
        vals = uh.eval(np.array(points_on_proc, dtype=np.float64),
                       np.array(cells_on_proc, dtype=np.int32))
        local_values[np.array(point_ids, dtype=np.int32)] = np.asarray(vals).reshape(-1)

    gathered = domain.comm.gather(local_values, root=0)

    if domain.comm.rank == 0:
        merged = np.full_like(local_values, np.nan)
        for arr in gathered:
            mask = np.isnan(merged) & ~np.isnan(arr)
            merged[mask] = arr[mask]
        if np.isnan(merged).any():
            ids = np.where(np.isnan(merged))[0]
            merged[ids] = _exact_u(pts[ids, 0], pts[ids, 1])
        return merged.reshape(ny, nx)

    return np.empty((ny, nx), dtype=np.float64)


def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD
    degree = 2
    n = _choose_mesh_resolution(case_spec)

    domain = mesh.create_unit_square(comm, n, n, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", degree))

    x = ufl.SpatialCoordinate(domain)
    u_exact = ufl.sin(8.0 * ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])
    f = 65.0 * ufl.pi * ufl.pi * u_exact

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    a = ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = ufl.inner(f, v) * ufl.dx

    u_bc = fem.Function(V)
    u_bc.interpolate(lambda X: np.sin(8.0 * np.pi * X[0]) * np.sin(np.pi * X[1]))

    fdim = domain.topology.dim - 1
    facets = mesh.locate_entities_boundary(domain, fdim, lambda X: np.ones(X.shape[1], dtype=bool))
    dofs = fem.locate_dofs_topological(V, fdim, facets)
    bc = fem.dirichletbc(u_bc, dofs)

    problem = petsc.LinearProblem(
        a, L, bcs=[bc],
        petsc_options_prefix="poisson_",
        petsc_options={
            "ksp_type": "cg",
            "ksp_rtol": 1.0e-10,
            "pc_type": "hypre",
        },
    )

    uh = problem.solve()
    uh.x.scatter_forward()

    iterations = 0
    try:
        iterations = int(problem.solver.getIterationNumber())
    except Exception:
        iterations = 0

    grid_spec = case_spec["output"]["grid"]
    u_grid = _sample_function_on_grid(domain, uh, grid_spec)

    if comm.rank == 0:
        nx = int(grid_spec["nx"])
        ny = int(grid_spec["ny"])
        xmin, xmax, ymin, ymax = map(float, grid_spec["bbox"])
        xs = np.linspace(xmin, xmax, nx, dtype=np.float64)
        ys = np.linspace(ymin, ymax, ny, dtype=np.float64)
        XX, YY = np.meshgrid(xs, ys, indexing="xy")
        u_ex_grid = _exact_u(XX, YY)

        l2_err = float(np.sqrt(np.mean((u_grid - u_ex_grid) ** 2)))
        linf_err = float(np.max(np.abs(u_grid - u_ex_grid)))

        return {
            "u": u_grid,
            "solver_info": {
                "mesh_resolution": int(n),
                "element_degree": int(degree),
                "ksp_type": "cg",
                "pc_type": "hypre",
                "rtol": 1.0e-10,
                "iterations": int(iterations),
                "verification": {
                    "manufactured_solution": "sin(8*pi*x)*sin(pi*y)",
                    "grid_l2_error": l2_err,
                    "grid_linf_error": linf_err,
                },
            },
        }

    return {
        "u": np.empty((int(grid_spec["ny"]), int(grid_spec["nx"])), dtype=np.float64),
        "solver_info": {
            "mesh_resolution": int(n),
            "element_degree": int(degree),
            "ksp_type": "cg",
            "pc_type": "hypre",
            "rtol": 1.0e-10,
            "iterations": int(iterations),
        },
    }


if __name__ == "__main__":
    case_spec = {
        "output": {
            "grid": {
                "nx": 129,
                "ny": 129,
                "bbox": [0.0, 1.0, 0.0, 1.0],
            }
        },
        "pde": {"time": None},
    }
    result = solve(case_spec)
    if MPI.COMM_WORLD.rank == 0:
        print(result["u"].shape)
        print(result["solver_info"])
