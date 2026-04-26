import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
import ufl
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import time

ScalarType = PETSc.ScalarType


def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD

    # ```DIAGNOSIS
    # equation_type:        helmholtz
    # spatial_dim:          2
    # domain_geometry:      rectangle
    # unknowns:             scalar
    # coupling:             none
    # linearity:            linear
    # time_dependence:      steady
    # stiffness:            N/A
    # dominant_physics:     wave
    # peclet_or_reynolds:   N/A
    # solution_regularity:  smooth
    # bc_type:              all_dirichlet
    # special_notes:        none
    # ```
    #
    # ```METHOD
    # spatial_method:       fem
    # element_or_basis:     Lagrange_P2
    # stabilization:        none
    # time_method:          none
    # nonlinear_solver:     none
    # linear_solver:        gmres
    # preconditioner:       ilu
    # special_treatment:    none
    # pde_skill:            helmholtz
    # ```

    pde = case_spec.get("pde", {})
    out_grid = case_spec["output"]["grid"]
    nx_out = int(out_grid["nx"])
    ny_out = int(out_grid["ny"])
    xmin, xmax, ymin, ymax = map(float, out_grid["bbox"])

    k = float(pde.get("k", case_spec.get("wavenumber", 18.0)))
    time_limit = float(case_spec.get("time_limit", 601.470))

    degree = 2
    rtol = 1.0e-9

    # Adaptive refinement to use a modest portion of available time for better accuracy
    mesh_candidates = [64, 96, 128, 160]
    soft_budget = min(120.0, max(20.0, 0.20 * time_limit))

    def src_ufl(x):
        return 12.0 * (
            ufl.exp(-90.0 * ((x[0] - 0.3) ** 2 + (x[1] - 0.7) ** 2))
            - ufl.exp(-90.0 * ((x[0] - 0.7) ** 2 + (x[1] - 0.3) ** 2))
        )

    def sample_on_grid(domain, uh):
        xs = np.linspace(xmin, xmax, nx_out, dtype=np.float64)
        ys = np.linspace(ymin, ymax, ny_out, dtype=np.float64)
        XX, YY = np.meshgrid(xs, ys, indexing="xy")
        pts = np.column_stack([XX.ravel(), YY.ravel(), np.zeros(nx_out * ny_out, dtype=np.float64)])

        tree = geometry.bb_tree(domain, domain.topology.dim)
        candidates = geometry.compute_collisions_points(tree, pts)
        colliding = geometry.compute_colliding_cells(domain, candidates, pts)

        values = np.full((pts.shape[0],), np.nan, dtype=np.float64)
        points_on_proc = []
        cells_on_proc = []
        ids = []

        for i in range(pts.shape[0]):
            links = colliding.links(i)
            if len(links) > 0:
                points_on_proc.append(pts[i])
                cells_on_proc.append(links[0])
                ids.append(i)

        if points_on_proc:
            vals = uh.eval(np.array(points_on_proc, dtype=np.float64),
                           np.array(cells_on_proc, dtype=np.int32))
            values[np.array(ids, dtype=np.int64)] = np.real(np.asarray(vals).reshape(-1))

        gathered = comm.gather(values, root=0)
        if comm.rank == 0:
            merged = np.full_like(values, np.nan)
            for arr in gathered:
                m = np.isfinite(arr)
                merged[m] = arr[m]
            merged[np.isnan(merged)] = 0.0
            return merged.reshape((ny_out, nx_out))
        return None

    best_result = None
    prev_grid = None
    start_all = time.perf_counter()

    for n in mesh_candidates:
        domain = mesh.create_unit_square(comm, n, n, cell_type=mesh.CellType.triangle)
        V = fem.functionspace(domain, ("Lagrange", degree))

        fdim = domain.topology.dim - 1
        facets = mesh.locate_entities_boundary(domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool))
        bdofs = fem.locate_dofs_topological(V, fdim, facets)
        bc = fem.dirichletbc(ScalarType(0.0), bdofs, V)

        u = ufl.TrialFunction(V)
        v = ufl.TestFunction(V)
        x = ufl.SpatialCoordinate(domain)
        f = src_ufl(x)

        a = (ufl.inner(ufl.grad(u), ufl.grad(v)) - ScalarType(k * k) * u * v) * ufl.dx
        L = f * v * ufl.dx

        ksp_type = "gmres"
        pc_type = "ilu"
        opts = {
            "ksp_type": ksp_type,
            "pc_type": pc_type,
            "ksp_rtol": rtol,
            "ksp_atol": 1.0e-14,
            "ksp_max_it": 5000,
            "ksp_gmres_restart": 200,
        }

        t0 = time.perf_counter()
        problem = petsc.LinearProblem(
            a, L, bcs=[bc],
            petsc_options=opts,
            petsc_options_prefix=f"helm_{n}_"
        )
        uh = problem.solve()
        uh.x.scatter_forward()
        solve_time = time.perf_counter() - t0

        ksp = problem.solver
        iterations = int(ksp.getIterationNumber())
        reason = int(ksp.getConvergedReason())

        if reason <= 0:
            ksp_type = "preonly"
            pc_type = "lu"
            t0 = time.perf_counter()
            problem = petsc.LinearProblem(
                a, L, bcs=[bc],
                petsc_options={"ksp_type": ksp_type, "pc_type": pc_type},
                petsc_options_prefix=f"helm_lu_{n}_"
            )
            uh = problem.solve()
            uh.x.scatter_forward()
            solve_time = time.perf_counter() - t0
            ksp = problem.solver
            iterations = int(ksp.getIterationNumber())

        # Accuracy verification module: residual norm + grid-convergence indicator
        x = ufl.SpatialCoordinate(domain)
        residual_form = fem.form(
            ufl.inner(-ufl.div(ufl.grad(uh)) - ScalarType(k * k) * uh - src_ufl(x),
                      -ufl.div(ufl.grad(uh)) - ScalarType(k * k) * uh - src_ufl(x)) * ufl.dx
        )
        try:
            residual_l2 = float(np.sqrt(comm.allreduce(fem.assemble_scalar(residual_form), op=MPI.SUM)))
        except Exception:
            residual_l2 = float("nan")

        u_grid = sample_on_grid(domain, uh)

        if comm.rank == 0:
            if prev_grid is None:
                grid_change_l2 = np.inf
            else:
                grid_change_l2 = float(np.sqrt(np.mean((u_grid - prev_grid) ** 2)))
        else:
            grid_change_l2 = None

        grid_change_l2 = comm.bcast(grid_change_l2, root=0)

        best_result = {
            "u": u_grid if comm.rank == 0 else None,
            "solver_info": {
                "mesh_resolution": int(n),
                "element_degree": int(degree),
                "ksp_type": str(ksp_type),
                "pc_type": str(pc_type),
                "rtol": float(rtol),
                "iterations": int(iterations),
                "verification": {
                    "residual_l2": residual_l2,
                    "grid_change_l2": None if not np.isfinite(grid_change_l2) else float(grid_change_l2),
                    "solve_time_sec": float(solve_time),
                },
            },
        }

        prev_grid = u_grid if comm.rank == 0 else None

        elapsed = time.perf_counter() - start_all
        if np.isfinite(grid_change_l2) and grid_change_l2 < 2.0e-3:
            if elapsed > 5.0:
                break
        if elapsed > soft_budget:
            break

    return best_result


if __name__ == "__main__":
    case = {
        "pde": {"k": 18.0},
        "output": {"grid": {"nx": 64, "ny": 64, "bbox": [0.0, 1.0, 0.0, 1.0]}},
    }
    result = solve(case)
    if MPI.COMM_WORLD.rank == 0:
        print(result["u"].shape)
        print(result["solver_info"])
