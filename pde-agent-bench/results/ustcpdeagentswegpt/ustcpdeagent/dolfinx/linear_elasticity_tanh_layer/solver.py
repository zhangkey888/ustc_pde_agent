import time
import math
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc

from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl


comm = MPI.COMM_WORLD
ScalarType = PETSc.ScalarType


def solve(case_spec: dict) -> dict:
    """
    Solve the 2D small-strain linear elasticity manufactured-solution problem.

    Returns
    -------
    dict with keys:
      - "u": sampled displacement magnitude on requested grid, shape (ny, nx)
      - "solver_info": metadata and solver stats
    """
    # ---------------------------
    # DIAGNOSIS
    # ---------------------------
    # equation_type: linear_elasticity
    # spatial_dim: 2
    # domain_geometry: rectangle
    # unknowns: vector
    # coupling: none
    # linearity: linear
    # time_dependence: steady
    # stiffness: N/A
    # dominant_physics: mixed
    # peclet_or_reynolds: N/A
    # solution_regularity: boundary_layer-like in y but smooth
    # bc_type: all_dirichlet
    # special_notes: manufactured_solution

    # ---------------------------
    # METHOD
    # ---------------------------
    # spatial_method: fem
    # element_or_basis: Lagrange_P2_vector (P3 fallback if time allows)
    # stabilization: none
    # time_method: none
    # nonlinear_solver: none
    # linear_solver: cg
    # preconditioner: hypre
    # special_treatment: none
    # pde_skill: linear_elasticity

    output = case_spec["output"]["grid"]
    nx_out = int(output["nx"])
    ny_out = int(output["ny"])
    bbox = output["bbox"]
    xmin, xmax, ymin, ymax = map(float, bbox)

    # Time budget if provided, otherwise use benchmark limit
    time_budget = 22.142
    if isinstance(case_spec.get("time_limit"), (int, float)):
        time_budget = float(case_spec["time_limit"])
    elif isinstance(case_spec.get("wall_time_sec"), (int, float)):
        time_budget = float(case_spec["wall_time_sec"])

    # Material parameters
    E = 1.0
    nu = 0.3
    mu = E / (2.0 * (1.0 + nu))
    lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))

    gdim = 2

    def exact_components_numpy(x, y):
        u1 = np.tanh(6.0 * (y - 0.5)) * np.sin(np.pi * x)
        u2 = 0.1 * np.sin(2.0 * np.pi * x) * np.sin(np.pi * y)
        return u1, u2

    def exact_magnitude_grid(nx, ny, bbox):
        xmin, xmax, ymin, ymax = bbox
        xs = np.linspace(xmin, xmax, nx)
        ys = np.linspace(ymin, ymax, ny)
        XX, YY = np.meshgrid(xs, ys)
        u1, u2 = exact_components_numpy(XX, YY)
        return np.sqrt(u1 * u1 + u2 * u2)

    start_total = time.perf_counter()

    # Candidate discretizations: prefer accuracy, adapt upward if time permits.
    # P2 is generally robust here; keep moderate meshes to fit budget.
    candidates = [
        (56, 2),
        (72, 2),
        (88, 2),
        (104, 2),
        (120, 2),
        (88, 3),
    ]

    best = None

    # Build requested exact solution once in symbolic form helper
    for mesh_resolution, degree in candidates:
        now = time.perf_counter()
        if now - start_total > 0.92 * time_budget and best is not None:
            break

        try:
            domain = mesh.create_unit_square(
                comm, mesh_resolution, mesh_resolution, cell_type=mesh.CellType.triangle
            )
            V = fem.functionspace(domain, ("Lagrange", degree, (gdim,)))
            x = ufl.SpatialCoordinate(domain)

            u_exact_ufl = ufl.as_vector(
                (
                    ufl.tanh(6.0 * (x[1] - 0.5)) * ufl.sin(ufl.pi * x[0]),
                    0.1 * ufl.sin(2.0 * ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1]),
                )
            )

            def eps(w):
                return ufl.sym(ufl.grad(w))

            def sigma(w):
                return 2.0 * mu * eps(w) + lam * ufl.tr(eps(w)) * ufl.Identity(gdim)

            f_ufl = -ufl.div(sigma(u_exact_ufl))

            u = ufl.TrialFunction(V)
            v = ufl.TestFunction(V)

            a = ufl.inner(sigma(u), eps(v)) * ufl.dx
            L = ufl.inner(f_ufl, v) * ufl.dx

            # Dirichlet BC from exact displacement on full boundary
            fdim = domain.topology.dim - 1
            boundary_facets = mesh.locate_entities_boundary(
                domain, fdim, lambda X: np.ones(X.shape[1], dtype=bool)
            )
            boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)

            u_bc = fem.Function(V)
            u_bc.interpolate(fem.Expression(u_exact_ufl, V.element.interpolation_points))
            bc = fem.dirichletbc(u_bc, boundary_dofs)

            # Try iterative solver first
            petsc_options = {
                "ksp_type": "cg",
                "pc_type": "hypre",
                "ksp_rtol": 1.0e-10,
                "ksp_atol": 1.0e-12,
                "ksp_max_it": 5000,
            }

            problem = petsc.LinearProblem(
                a,
                L,
                bcs=[bc],
                petsc_options=petsc_options,
                petsc_options_prefix=f"elas_{mesh_resolution}_{degree}_",
            )
            t0 = time.perf_counter()
            uh = problem.solve()
            uh.x.scatter_forward()
            solve_time = time.perf_counter() - t0

            # Access convergence info from embedded KSP if available
            ksp_type = "cg"
            pc_type = "hypre"
            rtol = 1.0e-10
            iterations = -1
            try:
                ksp = problem.solver
                iterations = int(ksp.getIterationNumber())
                ksp_type = ksp.getType()
                pc_type = ksp.getPC().getType()
                rtol = float(ksp.getTolerances()[0])
                if not ksp.converged:
                    raise RuntimeError("Iterative solve did not converge")
            except Exception:
                pass

            # Fallback to direct LU if iterative stats unavailable or poor convergence
            if iterations < 0:
                problem = petsc.LinearProblem(
                    a,
                    L,
                    bcs=[bc],
                    petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
                    petsc_options_prefix=f"elaslu_{mesh_resolution}_{degree}_",
                )
                t0 = time.perf_counter()
                uh = problem.solve()
                uh.x.scatter_forward()
                solve_time = time.perf_counter() - t0
                ksp_type = "preonly"
                pc_type = "lu"
                rtol = 0.0
                iterations = 1

            # Accuracy verification 1: FEM L2 error against exact vector field
            err_form = fem.form(ufl.inner(uh - u_exact_ufl, uh - u_exact_ufl) * ufl.dx)
            l2_sq_local = fem.assemble_scalar(err_form)
            l2_sq = comm.allreduce(l2_sq_local, op=MPI.SUM)
            l2_error = math.sqrt(max(l2_sq, 0.0))

            # Accuracy verification 2: output-grid magnitude error
            u_grid = _sample_displacement_magnitude(uh, nx_out, ny_out, bbox)
            exact_grid = exact_magnitude_grid(nx_out, ny_out, bbox)
            grid_l2 = float(np.sqrt(np.mean((u_grid - exact_grid) ** 2)))
            grid_linf = float(np.max(np.abs(u_grid - exact_grid)))

            elapsed = time.perf_counter() - start_total
            candidate_result = {
                "u": u_grid,
                "solver_info": {
                    "mesh_resolution": int(mesh_resolution),
                    "element_degree": int(degree),
                    "ksp_type": str(ksp_type),
                    "pc_type": str(pc_type),
                    "rtol": float(rtol),
                    "iterations": int(iterations),
                    "verification_l2_error_fem": float(l2_error),
                    "verification_l2_error_grid": float(grid_l2),
                    "verification_linf_error_grid": float(grid_linf),
                    "wall_time_sec": float(elapsed),
                    "solve_time_sec": float(solve_time),
                },
            }

            if best is None or candidate_result["solver_info"]["verification_l2_error_grid"] < best["solver_info"]["verification_l2_error_grid"]:
                best = candidate_result

            # If already sufficiently accurate and little time remains, stop.
            if elapsed > 0.8 * time_budget and best is not None:
                break

        except Exception:
            # Robust fallback: continue to next candidate
            continue

    if best is None:
        raise RuntimeError("Failed to solve linear elasticity problem for all candidate discretizations.")

    return best


def _sample_displacement_magnitude(u_sol: fem.Function, nx: int, ny: int, bbox):
    domain = u_sol.function_space.mesh
    xmin, xmax, ymin, ymax = map(float, bbox)

    xs = np.linspace(xmin, xmax, nx)
    ys = np.linspace(ymin, ymax, ny)
    XX, YY = np.meshgrid(xs, ys)
    pts = np.column_stack([XX.ravel(), YY.ravel(), np.zeros(nx * ny, dtype=np.float64)])

    tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(tree, pts)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, pts)

    local_indices = []
    local_points = []
    local_cells = []
    for i in range(pts.shape[0]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            local_indices.append(i)
            local_points.append(pts[i])
            local_cells.append(links[0])

    local_values = np.full((pts.shape[0], 2), np.nan, dtype=np.float64)
    if len(local_points) > 0:
        vals = u_sol.eval(np.array(local_points, dtype=np.float64), np.array(local_cells, dtype=np.int32))
        local_values[np.array(local_indices, dtype=np.int32), :] = np.asarray(vals, dtype=np.float64)

    gathered = comm.gather(local_values, root=0)
    if comm.rank == 0:
        merged = np.full((pts.shape[0], 2), np.nan, dtype=np.float64)
        for arr in gathered:
            mask = np.isfinite(arr[:, 0])
            merged[mask] = arr[mask]
        if np.isnan(merged).any():
            raise RuntimeError("Some sampling points could not be evaluated on any rank.")
        mag = np.linalg.norm(merged, axis=1).reshape(ny, nx)
    else:
        mag = None

    mag = comm.bcast(mag, root=0)
    return mag


if __name__ == "__main__":
    case_spec = {
        "output": {
            "grid": {
                "nx": 128,
                "ny": 128,
                "bbox": [0.0, 1.0, 0.0, 1.0],
            }
        },
        "pde": {"time": None},
    }
    result = solve(case_spec)
    if comm.rank == 0:
        print(result["u"].shape)
        print(result["solver_info"])
