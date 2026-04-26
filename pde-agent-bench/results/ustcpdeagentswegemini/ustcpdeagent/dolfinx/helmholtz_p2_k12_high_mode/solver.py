import math
import time
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl

ScalarType = PETSc.ScalarType


def _manufactured_u(x):
    return np.sin(3.0 * np.pi * x[0]) * np.sin(3.0 * np.pi * x[1])


def _sample_function_on_grid(domain, uh, grid_spec):
    nx = int(grid_spec["nx"])
    ny = int(grid_spec["ny"])
    xmin, xmax, ymin, ymax = grid_spec["bbox"]

    xs = np.linspace(xmin, xmax, nx)
    ys = np.linspace(ymin, ymax, ny)
    XX, YY = np.meshgrid(xs, ys, indexing="xy")
    pts2 = np.column_stack([XX.ravel(), YY.ravel(), np.zeros(nx * ny, dtype=np.float64)])

    tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(tree, pts2)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, pts2)

    local_values = np.full(nx * ny, np.nan, dtype=np.float64)
    points_on_proc = []
    cells_on_proc = []
    eval_map = []

    for i in range(nx * ny):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(pts2[i])
            cells_on_proc.append(links[0])
            eval_map.append(i)

    if len(points_on_proc) > 0:
        vals = uh.eval(np.array(points_on_proc, dtype=np.float64), np.array(cells_on_proc, dtype=np.int32))
        local_values[np.array(eval_map, dtype=np.int32)] = np.asarray(vals).reshape(-1).real

    gathered = domain.comm.gather(local_values, root=0)
    if domain.comm.rank == 0:
        global_values = np.full(nx * ny, np.nan, dtype=np.float64)
        for arr in gathered:
            mask = np.isfinite(arr)
            global_values[mask] = arr[mask]
        if np.any(~np.isfinite(global_values)):
            missing = np.where(~np.isfinite(global_values))[0]
            raise RuntimeError(f"Failed to evaluate solution at {len(missing)} output points")
        u_grid = global_values.reshape(ny, nx)
    else:
        u_grid = None

    u_grid = domain.comm.bcast(u_grid, root=0)
    return u_grid


def _solve_once(n, degree, k_value, ksp_type, pc_type, rtol):
    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_square(comm, n, n, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", degree))

    x = ufl.SpatialCoordinate(domain)
    u_exact_ufl = ufl.sin(3.0 * ufl.pi * x[0]) * ufl.sin(3.0 * ufl.pi * x[1])
    f_ufl = (18.0 * ufl.pi**2 - k_value**2) * u_exact_ufl

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    a = (ufl.inner(ufl.grad(u), ufl.grad(v)) - (k_value**2) * u * v) * ufl.dx
    L = f_ufl * v * ufl.dx

    fdim = domain.topology.dim - 1
    facets = mesh.locate_entities_boundary(domain, fdim, lambda X: np.ones(X.shape[1], dtype=bool))
    bdofs = fem.locate_dofs_topological(V, fdim, facets)

    u_bc = fem.Function(V)
    u_bc.interpolate(_manufactured_u)
    bc = fem.dirichletbc(u_bc, bdofs)

    problem = petsc.LinearProblem(
        a,
        L,
        bcs=[bc],
        petsc_options_prefix="helmholtz_",
        petsc_options={
            "ksp_type": ksp_type,
            "pc_type": pc_type,
            "ksp_rtol": rtol,
            "ksp_atol": 1.0e-14,
            "ksp_max_it": 5000,
        },
    )
    uh = problem.solve()
    uh.x.scatter_forward()

    ksp = problem.solver
    its = int(ksp.getIterationNumber())

    u_exact = fem.Function(V)
    u_exact.interpolate(_manufactured_u)
    err_fun = fem.Function(V)
    err_fun.x.array[:] = uh.x.array - u_exact.x.array
    err_fun.x.scatter_forward()

    l2_err_local = fem.assemble_scalar(fem.form(ufl.inner(err_fun, err_fun) * ufl.dx))
    l2_err = math.sqrt(domain.comm.allreduce(l2_err_local, op=MPI.SUM))

    return domain, uh, l2_err, its


def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD
    k_value = float(case_spec.get("pde", {}).get("k", 12.0))
    grid_spec = case_spec["output"]["grid"]

    # DIAGNOSIS / METHOD required by task instructions:
    # ```DIAGNOSIS
    # equation_type: helmholtz
    # spatial_dim: 2
    # domain_geometry: rectangle
    # unknowns: scalar
    # coupling: none
    # linearity: linear
    # time_dependence: steady
    # stiffness: N/A
    # dominant_physics: wave
    # peclet_or_reynolds: N/A
    # solution_regularity: smooth
    # bc_type: all_dirichlet
    # special_notes: manufactured_solution
    # ```
    # ```METHOD
    # spatial_method: fem
    # element_or_basis: Lagrange_P2
    # stabilization: none
    # time_method: none
    # nonlinear_solver: none
    # linear_solver: gmres
    # preconditioner: ilu
    # special_treatment: none
    # pde_skill: helmholtz
    # ```

    candidates = [
        (48, 2, "gmres", "ilu", 1.0e-10),
        (64, 2, "gmres", "ilu", 1.0e-10),
        (80, 2, "gmres", "ilu", 1.0e-11),
        (96, 2, "gmres", "ilu", 1.0e-11),
        (112, 2, "gmres", "ilu", 1.0e-12),
        (128, 2, "gmres", "ilu", 1.0e-12),
        (96, 3, "preonly", "lu", 1.0e-12),
    ]

    target_error = 9.05e-05
    best = None
    t0 = time.time()

    for n, degree, ksp_type, pc_type, rtol in candidates:
        try:
            domain, uh, l2_err, its = _solve_once(n, degree, k_value, ksp_type, pc_type, rtol)
            elapsed = time.time() - t0
            best = {
                "domain": domain,
                "uh": uh,
                "l2_err": l2_err,
                "iterations": its,
                "mesh_resolution": n,
                "element_degree": degree,
                "ksp_type": ksp_type,
                "pc_type": pc_type,
                "rtol": rtol,
                "elapsed": elapsed,
            }
            if l2_err <= target_error:
                # Use some extra accuracy if we are far below the time budget, but stop before overdoing it.
                if elapsed > 60.0:
                    break
        except Exception:
            continue

    if best is None:
        # Robust fallback
        domain, uh, l2_err, its = _solve_once(96, 3, k_value, "preonly", "lu", 1.0e-12)
        best = {
            "domain": domain,
            "uh": uh,
            "l2_err": l2_err,
            "iterations": its,
            "mesh_resolution": 96,
            "element_degree": 3,
            "ksp_type": "preonly",
            "pc_type": "lu",
            "rtol": 1.0e-12,
            "elapsed": time.time() - t0,
        }

    u_grid = _sample_function_on_grid(best["domain"], best["uh"], grid_spec)

    solver_info = {
        "mesh_resolution": int(best["mesh_resolution"]),
        "element_degree": int(best["element_degree"]),
        "ksp_type": str(best["ksp_type"]),
        "pc_type": str(best["pc_type"]),
        "rtol": float(best["rtol"]),
        "iterations": int(best["iterations"]),
        "l2_error": float(best["l2_err"]),
    }

    return {"u": u_grid, "solver_info": solver_info}


if __name__ == "__main__":
    case_spec = {
        "pde": {"k": 12.0, "time": None},
        "output": {"grid": {"nx": 64, "ny": 64, "bbox": [0.0, 1.0, 0.0, 1.0]}},
    }
    result = solve(case_spec)
    if MPI.COMM_WORLD.rank == 0:
        print(result["u"].shape)
        print(result["solver_info"])
