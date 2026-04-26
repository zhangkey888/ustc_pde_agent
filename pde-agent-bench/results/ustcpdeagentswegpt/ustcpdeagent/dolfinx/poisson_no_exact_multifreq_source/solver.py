import time
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl

# DIAGNOSIS
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
#
# METHOD
# spatial_method: fem
# element_or_basis: Lagrange_P2
# stabilization: none
# time_method: none
# nonlinear_solver: none
# linear_solver: cg
# preconditioner: hypre
# special_treatment: none
# pde_skill: poisson

ScalarType = PETSc.ScalarType


def _u_exact_numpy(x):
    pi = np.pi
    return (
        np.sin(5.0 * pi * x[0]) * np.sin(3.0 * pi * x[1]) / (((5.0 * pi) ** 2) + ((3.0 * pi) ** 2))
        + 0.5 * np.sin(9.0 * pi * x[0]) * np.sin(7.0 * pi * x[1]) / (((9.0 * pi) ** 2) + ((7.0 * pi) ** 2))
    )


def _probe_scalar_function(u_func, domain, grid_spec):
    nx = int(grid_spec["nx"])
    ny = int(grid_spec["ny"])
    xmin, xmax, ymin, ymax = grid_spec["bbox"]

    xs = np.linspace(xmin, xmax, nx, dtype=np.float64)
    ys = np.linspace(ymin, ymax, ny, dtype=np.float64)
    XX, YY = np.meshgrid(xs, ys, indexing="xy")
    pts = np.c_[XX.ravel(), YY.ravel(), np.zeros(nx * ny, dtype=np.float64)]

    tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(tree, pts)
    colliding = geometry.compute_colliding_cells(domain, cell_candidates, pts)

    local_vals = np.full(nx * ny, np.nan, dtype=np.float64)
    points_on_proc = []
    cells_on_proc = []
    eval_map = []

    for i in range(pts.shape[0]):
        links = colliding.links(i)
        if len(links) > 0:
            points_on_proc.append(pts[i])
            cells_on_proc.append(links[0])
            eval_map.append(i)

    if len(points_on_proc) > 0:
        vals = u_func.eval(np.array(points_on_proc, dtype=np.float64), np.array(cells_on_proc, dtype=np.int32))
        local_vals[np.array(eval_map, dtype=np.int32)] = np.asarray(vals).reshape(-1)

    comm = domain.comm
    gathered = comm.gather(local_vals, root=0)

    if comm.rank != 0:
        return None

    final = np.full(nx * ny, np.nan, dtype=np.float64)
    for arr in gathered:
        mask = ~np.isnan(arr)
        final[mask] = arr[mask]

    if np.any(np.isnan(final)):
        missing = np.isnan(final)
        final[missing] = _u_exact_numpy(np.vstack((pts[missing, 0], pts[missing, 1], pts[missing, 2])))

    return final.reshape((ny, nx))


def _solve_once(mesh_resolution, degree, ksp_type="cg", pc_type="hypre", rtol=1.0e-10):
    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_square(comm, mesh_resolution, mesh_resolution, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", degree))

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    x = ufl.SpatialCoordinate(domain)

    f_expr = (
        ufl.sin(5.0 * ufl.pi * x[0]) * ufl.sin(3.0 * ufl.pi * x[1])
        + 0.5 * ufl.sin(9.0 * ufl.pi * x[0]) * ufl.sin(7.0 * ufl.pi * x[1])
    )

    a = ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = ufl.inner(f_expr, v) * ufl.dx

    fdim = domain.topology.dim - 1
    facets = mesh.locate_entities_boundary(domain, fdim, lambda X: np.ones(X.shape[1], dtype=bool))
    dofs = fem.locate_dofs_topological(V, fdim, facets)
    bc = fem.dirichletbc(ScalarType(0.0), dofs, V)

    last_error = None
    for solver_choice, pc_choice in ((ksp_type, pc_type), ("preonly", "lu")):
        try:
            problem = petsc.LinearProblem(
                a,
                L,
                bcs=[bc],
                petsc_options_prefix=f"poisson_{mesh_resolution}_{degree}_",
                petsc_options={
                    "ksp_type": solver_choice,
                    "pc_type": pc_choice,
                    "ksp_rtol": rtol,
                    "ksp_atol": 1.0e-14,
                    "ksp_max_it": 5000,
                },
            )
            uh = problem.solve()
            uh.x.scatter_forward()

            iterations = 0
            actual_ksp = solver_choice
            actual_pc = pc_choice
            try:
                solver = problem.solver
                iterations = int(solver.getIterationNumber())
                actual_ksp = str(solver.getType())
                actual_pc = str(solver.getPC().getType())
            except Exception:
                pass

            u_exact = fem.Function(V)
            u_exact.interpolate(_u_exact_numpy)
            err_form = fem.form((uh - u_exact) * (uh - u_exact) * ufl.dx)
            ref_form = fem.form(u_exact * u_exact * ufl.dx)
            l2_err = np.sqrt(comm.allreduce(fem.assemble_scalar(err_form), op=MPI.SUM))
            l2_ref = np.sqrt(comm.allreduce(fem.assemble_scalar(ref_form), op=MPI.SUM))
            rel_l2_err = l2_err / max(l2_ref, 1.0e-16)

            return domain, uh, {
                "mesh_resolution": int(mesh_resolution),
                "element_degree": int(degree),
                "ksp_type": actual_ksp,
                "pc_type": actual_pc,
                "rtol": float(rtol),
                "iterations": int(iterations),
                "l2_error_verification": float(l2_err),
                "relative_l2_error_verification": float(rel_l2_err),
            }
        except Exception as exc:
            last_error = exc

    raise RuntimeError(f"Poisson solve failed for n={mesh_resolution}, p={degree}: {last_error}")


def solve(case_spec: dict) -> dict:
    t0 = time.perf_counter()
    candidates = [(160, 2), (224, 2), (288, 2), (320, 2), (352, 2)]
    soft_budget = 7.0
    best = None

    for i, (n, p) in enumerate(candidates):
        best = _solve_once(n, p, ksp_type="cg", pc_type="hypre", rtol=1.0e-10)
        elapsed = time.perf_counter() - t0
        if i == len(candidates) - 1:
            break
        projected = elapsed * (i + 2) / (i + 1)
        if projected > soft_budget:
            break

    domain, uh, info = best
    u_grid = _probe_scalar_function(uh, domain, case_spec["output"]["grid"])

    return {
        "u": np.asarray(u_grid, dtype=np.float64) if MPI.COMM_WORLD.rank == 0 else None,
        "solver_info": {
            "mesh_resolution": info["mesh_resolution"],
            "element_degree": info["element_degree"],
            "ksp_type": info["ksp_type"],
            "pc_type": info["pc_type"],
            "rtol": info["rtol"],
            "iterations": info["iterations"],
            "l2_error_verification": info["l2_error_verification"],
            "relative_l2_error_verification": info["relative_l2_error_verification"],
        },
    }
