import math
import time
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from basix.ufl import element, mixed_element

# DIAGNOSIS
# equation_type: navier_stokes
# spatial_dim: 2
# domain_geometry: rectangle
# unknowns: vector+scalar
# coupling: saddle_point
# linearity: nonlinear
# time_dependence: steady
# stiffness: N/A
# dominant_physics: mixed
# peclet_or_reynolds: moderate
# solution_regularity: smooth
# bc_type: all_dirichlet
# special_notes: pressure_pinning / manufactured_solution
#
# METHOD
# spatial_method: fem
# element_or_basis: Taylor-Hood_P2P1
# stabilization: none
# time_method: none
# nonlinear_solver: newton
# linear_solver: gmres
# preconditioner: ilu
# special_treatment: pressure_pinning
# pde_skill: navier_stokes


def _u_exact_ufl(x):
    r2 = (x[0] - 0.5) ** 2 + (x[1] - 0.5) ** 2
    e = ufl.exp(-20.0 * r2)
    return ufl.as_vector((-40.0 * (x[1] - 0.5) * e, 40.0 * (x[0] - 0.5) * e))


def _f_exact_ufl(msh, nu):
    x = ufl.SpatialCoordinate(msh)
    return ufl.as_vector((0.0 * x[0], 0.0 * x[1]))



def _build_spaces(msh, degree_u=2, degree_p=1):
    cell = msh.topology.cell_name()
    gdim = msh.geometry.dim
    v_el = element("Lagrange", cell, degree_u, shape=(gdim,))
    q_el = element("Lagrange", cell, degree_p)
    W = fem.functionspace(msh, mixed_element([v_el, q_el]))
    V, _ = W.sub(0).collapse()
    Q, _ = W.sub(1).collapse()
    return W, V, Q


def _boundary_conditions(msh, W, V, Q):
    fdim = msh.topology.dim - 1
    facets = mesh.locate_entities_boundary(msh, fdim, lambda x: np.ones(x.shape[1], dtype=bool))

    uD = fem.Function(V)
    expr = fem.Expression(_u_exact_ufl(ufl.SpatialCoordinate(msh)), V.element.interpolation_points)
    uD.interpolate(expr)
    udofs = fem.locate_dofs_topological((W.sub(0), V), fdim, facets)
    bc_u = fem.dirichletbc(uD, udofs, W.sub(0))

    pdofs = fem.locate_dofs_geometrical(
        (W.sub(1), Q), lambda x: np.isclose(x[0], 0.0) & np.isclose(x[1], 0.0)
    )
    p0 = fem.Function(Q)
    p0.x.array[:] = 0.0
    bc_p = fem.dirichletbc(p0, pdofs, W.sub(1))
    return [bc_u, bc_p]


def _solve_stokes_initial_guess(msh, W, bcs, nu_value, direct=False):
    nu = PETSc.ScalarType(nu_value)
    u, p = ufl.TrialFunctions(W)
    v, q = ufl.TestFunctions(W)

    def eps(w):
        return ufl.sym(ufl.grad(w))

    dx = ufl.Measure("dx", domain=msh)
    a = (
        2.0 * nu * ufl.inner(eps(u), eps(v)) * dx
        - ufl.inner(p, ufl.div(v)) * dx
        + ufl.inner(ufl.div(u), q) * dx
    )
    L = ufl.inner(_f_exact_ufl(msh, nu), v) * dx
    opts = {
        "ksp_type": "preonly" if direct else "gmres",
        "pc_type": "lu" if direct else "ilu",
        "ksp_rtol": 1e-10,
    }
    problem = petsc.LinearProblem(
        a, L, bcs=bcs, petsc_options_prefix="stokes_init_", petsc_options=opts
    )
    return problem.solve()


def _solve_once(n, nu_value, newton_rtol=1e-9, newton_max_it=30):
    comm = MPI.COMM_WORLD
    msh = mesh.create_unit_square(comm, n, n, cell_type=mesh.CellType.triangle)
    W, V, Q = _build_spaces(msh)
    bcs = _boundary_conditions(msh, W, V, Q)

    w = fem.Function(W)
    w.x.array[:] = 0.0
    w.x.scatter_forward()

    u, p = ufl.split(w)
    v, q = ufl.TestFunctions(W)
    nu = PETSc.ScalarType(nu_value)

    def eps(uu):
        return ufl.sym(ufl.grad(uu))

    dx = ufl.Measure("dx", domain=msh)
    F = (
        2.0 * nu * ufl.inner(eps(u), eps(v)) * dx
        + ufl.inner(ufl.grad(u) * u, v) * dx
        - ufl.inner(p, ufl.div(v)) * dx
        + ufl.inner(ufl.div(u), q) * dx
        - ufl.inner(_f_exact_ufl(msh, nu), v) * dx
    )
    J = ufl.derivative(F, w)

    attempts = [
        {
            "snes_type": "newtonls",
            "snes_linesearch_type": "bt",
            "snes_rtol": newton_rtol,
            "snes_atol": 1e-12,
            "snes_max_it": newton_max_it,
            "ksp_type": "gmres",
            "pc_type": "ilu",
            "ksp_rtol": 1e-10,
        },
        {
            "snes_type": "newtonls",
            "snes_linesearch_type": "bt",
            "snes_rtol": newton_rtol,
            "snes_atol": 1e-12,
            "snes_max_it": newton_max_it + 10,
            "ksp_type": "preonly",
            "pc_type": "lu",
            "ksp_rtol": 1e-12,
        },
    ]

    last_error = None
    for i, opts in enumerate(attempts):
        try:
            problem = petsc.NonlinearProblem(
                F, w, bcs=bcs, J=J, petsc_options_prefix=f"ns_{n}_{i}_", petsc_options=opts
            )
            t0 = time.perf_counter()
            wh = problem.solve()
            tsolve = time.perf_counter() - t0
            wh.x.scatter_forward()

            u_h = wh.sub(0).collapse()
            u_ex = fem.Function(V)
            expr = fem.Expression(_u_exact_ufl(ufl.SpatialCoordinate(msh)), V.element.interpolation_points)
            u_ex.interpolate(expr)

            err_loc = fem.assemble_scalar(fem.form(ufl.inner(u_h - u_ex, u_h - u_ex) * dx))
            ref_loc = fem.assemble_scalar(fem.form(ufl.inner(u_ex, u_ex) * dx))
            err = math.sqrt(comm.allreduce(err_loc, op=MPI.SUM))
            ref = math.sqrt(max(comm.allreduce(ref_loc, op=MPI.SUM), 1e-30))

            snes = problem.solver
            ksp = snes.getKSP()
            info = {
                "mesh_resolution": int(n),
                "element_degree": 2,
                "ksp_type": ksp.getType(),
                "pc_type": ksp.getPC().getType(),
                "rtol": float(ksp.getTolerances()[0]),
                "iterations": int(snes.getLinearSolveIterations()),
                "nonlinear_iterations": [int(snes.getIterationNumber())],
                "l2_error_velocity": float(err),
                "relative_l2_error_velocity": float(err / ref),
                "solve_time_sec": float(tsolve),
            }
            return msh, u_h, info
        except Exception as exc:
            last_error = exc
    raise RuntimeError(f"Navier-Stokes solve failed on mesh {n}: {last_error}")


def _sample_velocity_magnitude(u_func, grid):
    msh = u_func.function_space.mesh
    nx = int(grid["nx"])
    ny = int(grid["ny"])
    xmin, xmax, ymin, ymax = grid["bbox"]

    xs = np.linspace(xmin, xmax, nx)
    ys = np.linspace(ymin, ymax, ny)
    X, Y = np.meshgrid(xs, ys, indexing="xy")
    pts = np.column_stack((X.ravel(), Y.ravel(), np.zeros(nx * ny, dtype=np.float64)))

    tree = geometry.bb_tree(msh, msh.topology.dim)
    candidates = geometry.compute_collisions_points(tree, pts)
    colliding = geometry.compute_colliding_cells(msh, candidates, pts)

    local = np.full((pts.shape[0], msh.geometry.dim), np.nan, dtype=np.float64)
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
        vals = u_func.eval(np.array(points_on_proc, dtype=np.float64), np.array(cells_on_proc, dtype=np.int32))
        local[np.array(ids, dtype=np.int32)] = vals

    gathered = msh.comm.gather(local, root=0)
    if msh.comm.rank != 0:
        return None

    merged = np.full_like(local, np.nan)
    for arr in gathered:
        mask = np.isfinite(arr[:, 0])
        merged[mask] = arr[mask]
    merged = np.nan_to_num(merged, nan=0.0)
    return np.linalg.norm(merged, axis=1).reshape(ny, nx)


def solve(case_spec: dict) -> dict:
    nu_value = float(case_spec.get("pde", {}).get("nu", case_spec.get("viscosity", 0.12)))
    grid = case_spec["output"]["grid"]
    time_limit = float(case_spec.get("time_limit_sec", 663.422))
    start = time.perf_counter()

    best = None
    for n in [24, 32, 40, 48, 56, 64]:
        if time.perf_counter() - start > 0.92 * time_limit:
            break
        try:
            trial = _solve_once(n, nu_value)
            best = trial
        except Exception:
            if best is not None:
                break

    if best is None:
        raise RuntimeError("Could not compute a valid solution.")

    msh, u_h, info = best
    u_grid = _sample_velocity_magnitude(u_h, grid)

    if MPI.COMM_WORLD.rank == 0:
        return {
            "u": np.asarray(u_grid, dtype=np.float64).reshape(int(grid["ny"]), int(grid["nx"])),
            "solver_info": {
                "mesh_resolution": info["mesh_resolution"],
                "element_degree": info["element_degree"],
                "ksp_type": info["ksp_type"],
                "pc_type": info["pc_type"],
                "rtol": info["rtol"],
                "iterations": info["iterations"],
                "nonlinear_iterations": info["nonlinear_iterations"],
                "l2_error_velocity": info["l2_error_velocity"],
                "relative_l2_error_velocity": info["relative_l2_error_velocity"],
            },
        }
    return None


if __name__ == "__main__":
    case_spec = {
        "pde": {"nu": 0.12},
        "output": {"grid": {"nx": 16, "ny": 16, "bbox": [0.0, 1.0, 0.0, 1.0]}},
        "time_limit_sec": 60.0,
    }
    out = solve(case_spec)
    if MPI.COMM_WORLD.rank == 0:
        print(out["u"].shape)
        print(out["solver_info"])
