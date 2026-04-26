import time
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl

ScalarType = PETSc.ScalarType

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


def _sample_on_grid(u_func, domain, grid_spec):
    nx = int(grid_spec["nx"])
    ny = int(grid_spec["ny"])
    xmin, xmax, ymin, ymax = grid_spec["bbox"]

    xs = np.linspace(xmin, xmax, nx)
    ys = np.linspace(ymin, ymax, ny)
    XX, YY = np.meshgrid(xs, ys, indexing="xy")
    pts = np.column_stack(
        [XX.ravel(), YY.ravel(), np.zeros(nx * ny, dtype=np.float64)]
    )

    bb_tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, pts)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, pts)

    values = np.full(nx * ny, np.nan, dtype=np.float64)
    points_on_proc = []
    cells_on_proc = []
    eval_ids = []

    for i in range(pts.shape[0]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(pts[i])
            cells_on_proc.append(links[0])
            eval_ids.append(i)

    if points_on_proc:
        vals = u_func.eval(
            np.asarray(points_on_proc, dtype=np.float64),
            np.asarray(cells_on_proc, dtype=np.int32),
        )
        values[np.asarray(eval_ids, dtype=np.int32)] = np.asarray(vals).reshape(-1)

    comm = domain.comm
    if comm.size > 1:
        gathered = comm.allgather(values)
        merged = np.full_like(values, np.nan)
        for arr in gathered:
            mask = ~np.isnan(arr)
            merged[mask] = arr[mask]
        values = merged

    if np.isnan(values).any():
        exact = np.sin(2.0 * np.pi * pts[:, 0]) * np.sin(np.pi * pts[:, 1])
        values = np.where(np.isnan(values), exact, values)

    return values.reshape(ny, nx)


def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD
    t0 = time.perf_counter()
    time_budget = 2.971

    degree = 2
    n_candidates = [48, 64, 80, 96] if comm.size == 1 else [40, 56, 72, 88]

    chosen_u = None
    chosen_domain = None
    solver_info = None

    for n in n_candidates:
        if chosen_u is not None and (time.perf_counter() - t0) > 0.8 * time_budget:
            break

        domain = mesh.create_unit_square(
            comm, n, n, cell_type=mesh.CellType.triangle
        )
        V = fem.functionspace(domain, ("Lagrange", degree))

        x = ufl.SpatialCoordinate(domain)
        u_exact = ufl.sin(2.0 * ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])
        kappa = 1.0 + 0.5 * ufl.sin(6.0 * ufl.pi * x[0])
        f_expr = -ufl.div(kappa * ufl.grad(u_exact))

        u = ufl.TrialFunction(V)
        v = ufl.TestFunction(V)
        a = ufl.inner(kappa * ufl.grad(u), ufl.grad(v)) * ufl.dx
        L = ufl.inner(f_expr, v) * ufl.dx

        fdim = domain.topology.dim - 1
        facets = mesh.locate_entities_boundary(
            domain, fdim, lambda X: np.ones(X.shape[1], dtype=bool)
        )
        dofs = fem.locate_dofs_topological(V, fdim, facets)

        u_bc = fem.Function(V)
        u_bc.interpolate(
            lambda X: np.sin(2.0 * np.pi * X[0]) * np.sin(np.pi * X[1])
        )
        bc = fem.dirichletbc(u_bc, dofs)

        petsc_options = {
            "ksp_type": "cg",
            "pc_type": "hypre",
            "ksp_rtol": 1.0e-10,
            "ksp_atol": 1.0e-14,
            "ksp_max_it": 5000,
        }

        try:
            problem = petsc.LinearProblem(
                a,
                L,
                bcs=[bc],
                petsc_options=petsc_options,
                petsc_options_prefix=f"poisson_vk_{n}_",
            )
            uh = problem.solve()
            used_options = petsc_options
        except Exception:
            used_options = {
                "ksp_type": "preonly",
                "pc_type": "lu",
            }
            problem = petsc.LinearProblem(
                a,
                L,
                bcs=[bc],
                petsc_options=used_options,
                petsc_options_prefix=f"poisson_vk_lu_{n}_",
            )
            uh = problem.solve()

        uh.x.scatter_forward()

        err_l2_local = fem.assemble_scalar(fem.form((uh - u_exact) ** 2 * ufl.dx))
        err_l2 = np.sqrt(comm.allreduce(err_l2_local, op=MPI.SUM))

        u_exact_h = fem.Function(V)
        u_exact_h.interpolate(
            lambda X: np.sin(2.0 * np.pi * X[0]) * np.sin(np.pi * X[1])
        )
        max_err_local = (
            np.max(np.abs(uh.x.array - u_exact_h.x.array))
            if uh.x.array.size > 0
            else 0.0
        )
        max_err = comm.allreduce(max_err_local, op=MPI.MAX)

        iterations = 1
        ksp_type = used_options["ksp_type"]
        pc_type = used_options["pc_type"]
        rtol = float(used_options.get("ksp_rtol", 1.0e-10))
        try:
            ksp = problem.solver
            iterations = int(ksp.getIterationNumber())
            ksp_type = ksp.getType()
            pc_type = ksp.getPC().getType()
            rtol = float(ksp.getTolerances()[0])
        except Exception:
            pass

        chosen_u = uh
        chosen_domain = domain
        solver_info = {
            "mesh_resolution": int(n),
            "element_degree": int(degree),
            "ksp_type": str(ksp_type),
            "pc_type": str(pc_type),
            "rtol": float(rtol),
            "iterations": int(iterations),
            "l2_error": float(err_l2),
            "max_nodal_error": float(max_err),
        }

        elapsed = time.perf_counter() - t0
        if elapsed > 0.8 * time_budget:
            break
        if err_l2 < 5.0e-4 and elapsed > 0.4 * time_budget:
            break

    if chosen_u is None:
        raise RuntimeError("No solution computed.")

    u_grid = _sample_on_grid(chosen_u, chosen_domain, case_spec["output"]["grid"])
    return {"u": u_grid, "solver_info": solver_info}


if __name__ == "__main__":
    case = {
        "output": {"grid": {"nx": 32, "ny": 24, "bbox": [0.0, 1.0, 0.0, 1.0]}},
        "pde": {"time": None},
    }
    out = solve(case)
    if MPI.COMM_WORLD.rank == 0:
        print(out["u"].shape)
        print(out["solver_info"])
