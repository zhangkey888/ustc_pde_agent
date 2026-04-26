import time
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl

ScalarType = PETSc.ScalarType


def _make_case_defaults(case_spec: dict) -> dict:
    case = dict(case_spec) if case_spec is not None else {}
    case.setdefault("pde", {})
    case.setdefault("output", {})
    case["pde"].setdefault("time", {})
    case["output"].setdefault("grid", {})
    case["output"]["grid"].setdefault("nx", 64)
    case["output"]["grid"].setdefault("ny", 64)
    case["output"]["grid"].setdefault("bbox", [0.0, 1.0, 0.0, 1.0])
    return case


def _sample_on_grid(domain, uh, grid_spec):
    nx = int(grid_spec["nx"])
    ny = int(grid_spec["ny"])
    xmin, xmax, ymin, ymax = grid_spec["bbox"]
    xs = np.linspace(xmin, xmax, nx)
    ys = np.linspace(ymin, ymax, ny)
    XX, YY = np.meshgrid(xs, ys, indexing="xy")
    points = np.column_stack([XX.ravel(), YY.ravel(), np.zeros(nx * ny, dtype=np.float64)])

    tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(tree, points)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points)

    local_vals = np.full(points.shape[0], np.nan, dtype=np.float64)
    pts_local = []
    cells_local = []
    ids_local = []
    for i in range(points.shape[0]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            pts_local.append(points[i])
            cells_local.append(links[0])
            ids_local.append(i)

    if len(pts_local) > 0:
        vals = uh.eval(np.array(pts_local, dtype=np.float64), np.array(cells_local, dtype=np.int32))
        local_vals[np.array(ids_local, dtype=np.int32)] = np.asarray(vals, dtype=np.float64).reshape(-1)

    comm = domain.comm
    gathered = comm.gather(local_vals, root=0)
    if comm.rank == 0:
        final = np.full(points.shape[0], np.nan, dtype=np.float64)
        for arr in gathered:
            mask = ~np.isnan(arr)
            final[mask] = arr[mask]
        if np.isnan(final).any():
            raise RuntimeError("Failed to evaluate FEM solution at some output grid points.")
        return final.reshape((ny, nx))
    return None


def _run_solver(case_spec: dict, mesh_resolution=64, degree=1, dt=0.005, reaction_rho=2.0,
                nonlinear_rtol=1e-9, ksp_rtol=1e-10):
    case_spec = _make_case_defaults(case_spec)
    comm = MPI.COMM_WORLD

    pde_time = case_spec["pde"].get("time", {})
    t0 = float(pde_time.get("t0", 0.0))
    t_end = float(pde_time.get("t_end", 0.3))
    dt = float(pde_time.get("dt", dt) or dt)
    scheme = pde_time.get("scheme", "backward_euler")
    if scheme != "backward_euler":
        scheme = "backward_euler"

    domain = mesh.create_unit_square(
        comm, mesh_resolution, mesh_resolution, cell_type=mesh.CellType.quadrilateral
    )
    V = fem.functionspace(domain, ("Lagrange", degree))

    fdim = domain.topology.dim - 1
    facets = mesh.locate_entities_boundary(domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool))
    dofs = fem.locate_dofs_topological(V, fdim, facets)
    bc = fem.dirichletbc(ScalarType(0.0), dofs, V)

    x = ufl.SpatialCoordinate(domain)
    u_n = fem.Function(V)
    u_n.interpolate(lambda X: 0.25 + 0.15 * np.sin(np.pi * X[0]) * np.sin(np.pi * X[1]))

    u = fem.Function(V)
    u.x.array[:] = u_n.x.array.copy()
    u.x.scatter_forward()

    v = ufl.TestFunction(V)
    eps = fem.Constant(domain, ScalarType(0.02))
    f = fem.Constant(domain, ScalarType(1.0))
    dt_c = fem.Constant(domain, ScalarType(dt))
    rho = fem.Constant(domain, ScalarType(reaction_rho))

    # Logistic reaction R(u)=rho*u*(1-u)
    F = ((u - u_n) / dt_c) * v * ufl.dx + eps * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx \
        + rho * u * (1.0 - u) * v * ufl.dx - f * v * ufl.dx
    J = ufl.derivative(F, u)

    nonlinear_iterations = []
    linear_iterations_total = 0

    petsc_options = {
        "snes_type": "newtonls",
        "snes_linesearch_type": "bt",
        "snes_rtol": nonlinear_rtol,
        "snes_atol": 1e-11,
        "snes_max_it": 30,
        "ksp_type": "gmres",
        "ksp_rtol": ksp_rtol,
        "pc_type": "ilu",
    }

    n_steps = int(round((t_end - t0) / dt))
    if n_steps <= 0:
        n_steps = 1
    dt_eff = (t_end - t0) / n_steps
    dt_c.value = ScalarType(dt_eff)

    u_initial_grid = _sample_on_grid(domain, u_n, case_spec["output"]["grid"])

    for step in range(n_steps):
        problem = petsc.NonlinearProblem(
            F, u, bcs=[bc], J=J,
            petsc_options_prefix=f"rd_{step}_",
            petsc_options=petsc_options
        )
        try:
            u = problem.solve()
        except Exception:
            fallback_options = dict(petsc_options)
            fallback_options["pc_type"] = "lu"
            fallback_options["ksp_type"] = "preonly"
            problem = petsc.NonlinearProblem(
                F, u, bcs=[bc], J=J,
                petsc_options_prefix=f"rd_fallback_{step}_",
                petsc_options=fallback_options
            )
            u = problem.solve()

        try:
            snes = problem.solver
            nonlinear_iterations.append(int(snes.getIterationNumber()))
            ksp = snes.getKSP()
            linear_iterations_total += int(ksp.getIterationNumber())
        except Exception:
            nonlinear_iterations.append(-1)

        u.x.scatter_forward()
        u_n.x.array[:] = u.x.array
        u_n.x.scatter_forward()

    u_grid = _sample_on_grid(domain, u, case_spec["output"]["grid"])

    solver_info = {
        "mesh_resolution": int(mesh_resolution),
        "element_degree": int(degree),
        "ksp_type": "gmres",
        "pc_type": "ilu",
        "rtol": float(ksp_rtol),
        "iterations": int(linear_iterations_total),
        "dt": float(dt_eff),
        "n_steps": int(n_steps),
        "time_scheme": "backward_euler",
        "nonlinear_iterations": nonlinear_iterations,
    }

    return {
        "u": u_grid,
        "u_initial": u_initial_grid,
        "solver_info": solver_info,
    }


def solve(case_spec: dict) -> dict:
    case_spec = _make_case_defaults(case_spec)
    return _run_solver(case_spec, mesh_resolution=96, degree=1, dt=0.0025, reaction_rho=2.0)


if __name__ == "__main__":
    case_spec = {
        "pde": {
            "time": {"t0": 0.0, "t_end": 0.3, "dt": 0.0025, "scheme": "backward_euler"}
        },
        "output": {
            "grid": {"nx": 64, "ny": 64, "bbox": [0.0, 1.0, 0.0, 1.0]}
        }
    }

    t_start = time.perf_counter()
    out_fine = solve(case_spec)
    wall = time.perf_counter() - t_start

    case_spec_coarse = {
        "pde": {
            "time": {"t0": 0.0, "t_end": 0.3, "dt": 0.01, "scheme": "backward_euler"}
        },
        "output": {
            "grid": {"nx": 64, "ny": 64, "bbox": [0.0, 1.0, 0.0, 1.0]}
        }
    }
    out_coarse = _run_solver(case_spec_coarse, mesh_resolution=64, degree=1, dt=0.005, reaction_rho=2.0)

    if MPI.COMM_WORLD.rank == 0:
        diff = out_fine["u"] - out_coarse["u"]
        l2_err = float(np.sqrt(np.mean(diff**2)))
        print(f"L2_ERROR: {l2_err:.12e}")
        print(f"WALL_TIME: {wall:.12e}")
        print(out_fine["u"].shape)
        print(out_fine["solver_info"])
