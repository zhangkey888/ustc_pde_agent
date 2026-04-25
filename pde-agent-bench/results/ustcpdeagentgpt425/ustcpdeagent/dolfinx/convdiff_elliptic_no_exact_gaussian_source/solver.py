import time
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl

# ```DIAGNOSIS
# equation_type: convection_diffusion
# spatial_dim: 2
# domain_geometry: rectangle
# unknowns: scalar
# coupling: none
# linearity: linear
# time_dependence: steady
# stiffness: stiff
# dominant_physics: mixed
# peclet_or_reynolds: high
# solution_regularity: smooth
# bc_type: all_dirichlet
# special_notes: none
# ```
#
# ```METHOD
# spatial_method: fem
# element_or_basis: Lagrange_P1
# stabilization: supg
# time_method: none
# nonlinear_solver: none
# linear_solver: gmres
# preconditioner: amg
# special_treatment: none
# pde_skill: convection_diffusion / reaction_diffusion / biharmonic
# ```

ScalarType = PETSc.ScalarType


def _normalize_case_spec(case_spec: dict) -> dict:
    if case_spec is None:
        case_spec = {}
    case_spec.setdefault("output", {})
    case_spec["output"].setdefault("grid", {})
    grid = case_spec["output"]["grid"]
    grid.setdefault("nx", 64)
    grid.setdefault("ny", 64)
    grid.setdefault("bbox", [0.0, 1.0, 0.0, 1.0])
    return case_spec


def _make_forms(domain, V, eps_value=0.05, beta_value=(2.0, 1.0)):
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    x = ufl.SpatialCoordinate(domain)

    eps = ScalarType(eps_value)
    beta_np = np.array(beta_value, dtype=np.float64)
    beta = fem.Constant(domain, beta_np.astype(ScalarType))
    beta_norm = float(np.linalg.norm(beta_np))
    f = ufl.exp(-250.0 * ((x[0] - 0.35) ** 2 + (x[1] - 0.65) ** 2))

    h = ufl.CellDiameter(domain)
    tau = h / (2.0 * beta_norm)

    a = (
        eps * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
        + ufl.inner(beta, ufl.grad(u)) * v * ufl.dx
        + tau * ufl.inner(beta, ufl.grad(u)) * ufl.inner(beta, ufl.grad(v)) * ufl.dx
    )
    L = (
        f * v * ufl.dx
        + tau * f * ufl.inner(beta, ufl.grad(v)) * ufl.dx
    )
    return a, L, beta_norm


def _boundary_condition(domain, V):
    fdim = domain.topology.dim - 1
    facets = mesh.locate_entities_boundary(domain, fdim, lambda X: np.ones(X.shape[1], dtype=bool))
    dofs = fem.locate_dofs_topological(V, fdim, facets)
    return fem.dirichletbc(ScalarType(0.0), dofs, V)


def _solve_on_mesh(mesh_resolution: int, degree: int = 1):
    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_square(comm, mesh_resolution, mesh_resolution, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", degree))
    a, L, beta_norm = _make_forms(domain, V)
    bc = _boundary_condition(domain, V)

    ksp_type = "gmres"
    pc_type = "hypre"
    rtol = 1e-9
    iterations = 0

    try:
        problem = petsc.LinearProblem(
            a,
            L,
            bcs=[bc],
            petsc_options_prefix=f"cd_{mesh_resolution}_",
            petsc_options={
                "ksp_type": ksp_type,
                "pc_type": pc_type,
                "ksp_rtol": rtol,
                "ksp_atol": 1e-12,
                "ksp_max_it": 5000,
            },
        )
        uh = problem.solve()
        uh.x.scatter_forward()
        try:
            iterations = int(problem.solver.getIterationNumber())
            ksp_type = str(problem.solver.getType())
            pc_type = str(problem.solver.getPC().getType())
        except Exception:
            pass
    except Exception:
        ksp_type = "preonly"
        pc_type = "lu"
        problem = petsc.LinearProblem(
            a,
            L,
            bcs=[bc],
            petsc_options_prefix=f"cd_lu_{mesh_resolution}_",
            petsc_options={"ksp_type": ksp_type, "pc_type": pc_type},
        )
        uh = problem.solve()
        uh.x.scatter_forward()
        iterations = 1

    h_est = 1.0 / mesh_resolution
    peclet = beta_norm * h_est / (2.0 * 0.05)

    solver_info = {
        "mesh_resolution": int(mesh_resolution),
        "element_degree": int(degree),
        "ksp_type": str(ksp_type),
        "pc_type": str(pc_type),
        "rtol": float(rtol),
        "iterations": int(iterations),
        "peclet_estimate": float(peclet),
    }
    return domain, uh, solver_info


def _sample_to_grid(domain, uh, grid_spec):
    nx = int(grid_spec["nx"])
    ny = int(grid_spec["ny"])
    xmin, xmax, ymin, ymax = [float(v) for v in grid_spec["bbox"]]

    xs = np.linspace(xmin, xmax, nx)
    ys = np.linspace(ymin, ymax, ny)
    XX, YY = np.meshgrid(xs, ys, indexing="xy")

    pts = np.zeros((nx * ny, 3), dtype=np.float64)
    pts[:, 0] = XX.ravel()
    pts[:, 1] = YY.ravel()

    bb_tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, pts)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, pts)

    values = np.zeros(nx * ny, dtype=np.float64)
    points_on_proc = []
    cells_on_proc = []
    eval_map = []

    for i in range(pts.shape[0]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(pts[i])
            cells_on_proc.append(links[0])
            eval_map.append(i)

    if len(points_on_proc) > 0:
        vals = uh.eval(np.array(points_on_proc, dtype=np.float64), np.array(cells_on_proc, dtype=np.int32))
        vals = np.asarray(vals).reshape(len(eval_map), -1)[:, 0]
        values[np.array(eval_map, dtype=np.int32)] = vals

    return values.reshape(ny, nx)


def _verification(case_spec, mesh_list):
    grid = case_spec["output"]["grid"]
    previous_grid = None
    best = None

    for n in mesh_list:
        domain, uh, solver_info = _solve_on_mesh(n, degree=1)
        u_grid = _sample_to_grid(domain, uh, grid)
        verification = {"mesh_convergence_rel_change": np.nan}
        if previous_grid is not None:
            verification["mesh_convergence_rel_change"] = float(
                np.linalg.norm((u_grid - previous_grid).ravel()) / max(np.linalg.norm(u_grid.ravel()), 1e-14)
            )
        previous_grid = u_grid.copy()
        best = (domain, uh, u_grid, solver_info, verification)

    return best


def solve(case_spec: dict) -> dict:
    case_spec = _normalize_case_spec(case_spec)
    t0 = time.perf_counter()

    mesh_candidates = [96, 144]
    domain, uh, u_grid, solver_info, verification = _verification(case_spec, mesh_candidates)

    elapsed = time.perf_counter() - t0
    if elapsed < 20.0:
        try:
            domain_f, uh_f, solver_info_f = _solve_on_mesh(192, degree=1)
            u_grid_f = _sample_to_grid(domain_f, uh_f, case_spec["output"]["grid"])
            rel = float(
                np.linalg.norm((u_grid_f - u_grid).ravel()) / max(np.linalg.norm(u_grid_f.ravel()), 1e-14)
            )
            if np.isnan(verification["mesh_convergence_rel_change"]) or rel <= verification["mesh_convergence_rel_change"]:
                domain, uh, u_grid, solver_info = domain_f, uh_f, u_grid_f, solver_info_f
                verification["mesh_convergence_rel_change"] = rel
        except Exception:
            pass

    solver_info["verification"] = verification
    solver_info["wall_time_sec"] = float(time.perf_counter() - t0)
    return {"u": np.asarray(u_grid, dtype=np.float64), "solver_info": solver_info}


if __name__ == "__main__":
    demo_case = {
        "output": {
            "grid": {
                "nx": 64,
                "ny": 64,
                "bbox": [0.0, 1.0, 0.0, 1.0],
            }
        }
    }
    result = solve(demo_case)
    print(result["u"].shape)
    print(result["solver_info"])
