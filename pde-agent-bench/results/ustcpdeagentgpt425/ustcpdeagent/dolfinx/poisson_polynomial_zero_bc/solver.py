import time
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl

# ```DIAGNOSIS
# equation_type:        poisson
# spatial_dim:          2
# domain_geometry:      rectangle
# unknowns:             scalar
# coupling:             none
# linearity:            linear
# time_dependence:      steady
# stiffness:            N/A
# dominant_physics:     diffusion
# peclet_or_reynolds:   N/A
# solution_regularity:  smooth
# bc_type:              all_dirichlet
# special_notes:        manufactured_solution
# ```
# ```METHOD
# spatial_method:       fem
# element_or_basis:     Lagrange_P2
# stabilization:        none
# time_method:          none
# nonlinear_solver:     none
# linear_solver:        direct_lu
# preconditioner:       none
# special_treatment:    none
# pde_skill:            poisson
# ```

ScalarType = PETSc.ScalarType


def _u_exact_vals(X):
    return X[0] * (1.0 - X[0]) * X[1] * (1.0 - X[1])


def _sample(u_fun, nx, ny, bbox):
    xmin, xmax, ymin, ymax = bbox
    xs = np.linspace(xmin, xmax, nx, dtype=np.float64)
    ys = np.linspace(ymin, ymax, ny, dtype=np.float64)
    XX, YY = np.meshgrid(xs, ys, indexing="xy")
    pts = np.column_stack([XX.ravel(), YY.ravel(), np.zeros(nx * ny, dtype=np.float64)])

    msh = u_fun.function_space.mesh
    tree = geometry.bb_tree(msh, msh.topology.dim)
    cands = geometry.compute_collisions_points(tree, pts)
    coll = geometry.compute_colliding_cells(msh, cands, pts)

    vals = np.full(nx * ny, np.nan, dtype=np.float64)
    p_local, c_local, ids = [], [], []
    for i in range(pts.shape[0]):
        links = coll.links(i)
        if len(links) > 0:
            p_local.append(pts[i])
            c_local.append(links[0])
            ids.append(i)

    if ids:
        ev = u_fun.eval(np.asarray(p_local, dtype=np.float64),
                        np.asarray(c_local, dtype=np.int32)).reshape(-1)
        vals[np.asarray(ids, dtype=np.int32)] = np.real(ev)

    if msh.comm.size > 1:
        send = np.where(np.isnan(vals), -1.0e300, vals)
        recv = np.empty_like(send)
        msh.comm.Allreduce(send, recv, op=MPI.MAX)
        vals = recv

    vals = vals.reshape(ny, nx)
    vals = np.where(np.isfinite(vals), vals, 0.0)
    return vals


def _solve_once(comm, n, degree):
    msh = mesh.create_unit_square(comm, n, n, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(msh, ("Lagrange", degree))

    x = ufl.SpatialCoordinate(msh)
    u_ex = x[0] * (1.0 - x[0]) * x[1] * (1.0 - x[1])
    f = 2.0 * (x[0] * (1.0 - x[0]) + x[1] * (1.0 - x[1]))

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    a = ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = ufl.inner(f, v) * ufl.dx

    fdim = msh.topology.dim - 1
    facets = mesh.locate_entities_boundary(msh, fdim, lambda X: np.ones(X.shape[1], dtype=bool))
    dofs = fem.locate_dofs_topological(V, fdim, facets)
    u_bc = fem.Function(V)
    u_bc.interpolate(_u_exact_vals)
    bc = fem.dirichletbc(u_bc, dofs)

    t0 = time.perf_counter()
    try:
        problem = petsc.LinearProblem(
            a, L, bcs=[bc],
            petsc_options_prefix=f"poisson_{n}_{degree}_",
            petsc_options={
                "ksp_type": "preonly",
                "pc_type": "lu",
            },
        )
        uh = problem.solve()
        ksp_type = "preonly"
        pc_type = "lu"
        iterations = 0
    except Exception:
        problem = petsc.LinearProblem(
            a, L, bcs=[bc],
            petsc_options_prefix=f"poisson_fallback_{n}_{degree}_",
            petsc_options={
                "ksp_type": "cg",
                "pc_type": "hypre",
                "ksp_rtol": 1.0e-12,
            },
        )
        uh = problem.solve()
        ksp_type = "cg"
        pc_type = "hypre"
        iterations = -1

    solve_time = time.perf_counter() - t0

    Ve = fem.functionspace(msh, ("Lagrange", max(degree + 1, 3)))
    ueh = fem.Function(Ve)
    ueh.interpolate(_u_exact_vals)
    uhe = fem.Function(Ve)
    uhe.interpolate(uh)
    e2 = fem.assemble_scalar(fem.form(ufl.inner(uhe - ueh, uhe - ueh) * ufl.dx))
    l2_error = np.sqrt(comm.allreduce(e2, op=MPI.SUM))

    return uh, {
        "mesh": msh,
        "n": n,
        "degree": degree,
        "ksp_type": ksp_type,
        "pc_type": pc_type,
        "rtol": 1.0e-12,
        "iterations": iterations,
        "l2_error": float(l2_error),
        "solve_time": float(solve_time),
    }


def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD

    candidates = [(24, 2), (40, 2), (64, 2)]
    chosen_uh = None
    chosen_info = None
    time_budget = 8.451
    target_fraction = 0.75

    for n, degree in candidates:
        uh, info = _solve_once(comm, n, degree)
        chosen_uh, chosen_info = uh, info
        projected_total = info["solve_time"] * 1.6
        if projected_total > target_fraction * time_budget:
            break

    grid = case_spec["output"]["grid"]
    u_grid = _sample(chosen_uh, int(grid["nx"]), int(grid["ny"]), grid["bbox"])

    return {
        "u": u_grid,
        "solver_info": {
            "mesh_resolution": chosen_info["n"],
            "element_degree": chosen_info["degree"],
            "ksp_type": chosen_info["ksp_type"],
            "pc_type": chosen_info["pc_type"],
            "rtol": chosen_info["rtol"],
            "iterations": chosen_info["iterations"],
            "l2_error_verification": chosen_info["l2_error"],
            "solve_time_sec": chosen_info["solve_time"],
        },
    }
