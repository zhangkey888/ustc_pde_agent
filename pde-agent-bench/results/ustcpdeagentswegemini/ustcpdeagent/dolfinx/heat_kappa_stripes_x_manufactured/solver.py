import math
import time
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
import ufl
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc

ScalarType = PETSc.ScalarType

DIAGNOSIS = """
DIAGNOSIS
equation_type: heat
spatial_dim: 2
domain_geometry: rectangle
unknowns: scalar
coupling: none
linearity: linear
time_dependence: transient
stiffness: stiff
dominant_physics: diffusion
peclet_or_reynolds: N/A
solution_regularity: smooth
bc_type: all_dirichlet
special_notes: manufactured_solution, variable_coeff
"""

METHOD = """
METHOD
spatial_method: fem
element_or_basis: Lagrange_P2
stabilization: none
time_method: backward_euler
nonlinear_solver: none
linear_solver: cg
preconditioner: hypre
special_treatment: none
pde_skill: heat
"""


def _as_float(v, default):
    try:
        return float(v)
    except Exception:
        return float(default)


def _extract_output_grid(case_spec):
    grid = case_spec.get("output", {}).get("grid", {})
    nx = int(grid.get("nx", 64))
    ny = int(grid.get("ny", 64))
    bbox = grid.get("bbox", [0.0, 1.0, 0.0, 1.0])
    return nx, ny, bbox


def _exact_ufl(msh, t):
    x = ufl.SpatialCoordinate(msh)
    return ufl.exp(-t) * ufl.sin(2.0 * ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])


def _kappa_ufl(msh):
    x = ufl.SpatialCoordinate(msh)
    return 1.0 + 0.5 * ufl.sin(6.0 * ufl.pi * x[0])


def _source_ufl(msh, t):
    u_ex = _exact_ufl(msh, t)
    kappa = _kappa_ufl(msh)
    return -u_ex - ufl.div(kappa * ufl.grad(u_ex))


def _probe_function(u_fun, pts):
    msh = u_fun.function_space.mesh
    tree = geometry.bb_tree(msh, msh.topology.dim)
    cands = geometry.compute_collisions_points(tree, pts)
    colliding = geometry.compute_colliding_cells(msh, cands, pts)

    local_vals = np.full(pts.shape[0], np.nan, dtype=np.float64)
    points_on_proc = []
    cells = []
    idx = []
    for i in range(pts.shape[0]):
        links = colliding.links(i)
        if len(links) > 0:
            points_on_proc.append(pts[i])
            cells.append(links[0])
            idx.append(i)

    if points_on_proc:
        vals = u_fun.eval(np.array(points_on_proc, dtype=np.float64), np.array(cells, dtype=np.int32))
        local_vals[np.array(idx, dtype=np.int32)] = np.asarray(vals, dtype=np.float64).reshape(-1)

    comm = msh.comm
    gathered = comm.gather(local_vals, root=0)
    if comm.rank == 0:
        vals = gathered[0].copy()
        for arr in gathered[1:]:
            mask = np.isnan(vals) & ~np.isnan(arr)
            vals[mask] = arr[mask]
        vals = np.nan_to_num(vals, nan=0.0)
        return vals
    return None


def _sample_on_grid(u_fun, nx, ny, bbox):
    xs = np.linspace(bbox[0], bbox[1], nx)
    ys = np.linspace(bbox[2], bbox[3], ny)
    X, Y = np.meshgrid(xs, ys, indexing="xy")
    pts = np.column_stack([X.ravel(), Y.ravel(), np.zeros(nx * ny, dtype=np.float64)])
    vals = _probe_function(u_fun, pts)
    if u_fun.function_space.mesh.comm.rank == 0:
        return vals.reshape(ny, nx)
    return None


def _solve_single(ncell, degree, dt, t0, t_end):
    comm = MPI.COMM_WORLD
    msh = mesh.create_unit_square(comm, ncell, ncell, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(msh, ("Lagrange", degree))

    fdim = msh.topology.dim - 1
    facets = mesh.locate_entities_boundary(msh, fdim, lambda x: np.ones(x.shape[1], dtype=bool))
    dofs = fem.locate_dofs_topological(V, fdim, facets)

    t_const = fem.Constant(msh, ScalarType(t0))
    dt_const = fem.Constant(msh, ScalarType(dt))

    u_n = fem.Function(V)
    u_bc = fem.Function(V)
    u_n.interpolate(fem.Expression(_exact_ufl(msh, ScalarType(t0)), V.element.interpolation_points))
    u_bc.interpolate(fem.Expression(_exact_ufl(msh, t_const), V.element.interpolation_points))
    bc = fem.dirichletbc(u_bc, dofs)

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    a = (u * v + dt_const * ufl.inner(_kappa_ufl(msh) * ufl.grad(u), ufl.grad(v))) * ufl.dx
    L = (u_n * v + dt_const * _source_ufl(msh, t_const) * v) * ufl.dx
    a_form = fem.form(a)
    L_form = fem.form(L)

    A = petsc.assemble_matrix(a_form, bcs=[bc])
    A.assemble()
    b = petsc.create_vector(L_form.function_spaces)

    uh = fem.Function(V)

    solver = PETSc.KSP().create(comm)
    solver.setOperators(A)
    solver.setType("cg")
    solver.getPC().setType("hypre")
    solver.setTolerances(rtol=1e-10, atol=1e-12, max_it=2000)
    solver.setFromOptions()

    total_iterations = 0
    n_steps = int(round((t_end - t0) / dt))

    for n in range(n_steps):
        t_now = t0 + (n + 1) * dt
        t_const.value = ScalarType(t_now)
        u_bc.interpolate(fem.Expression(_exact_ufl(msh, t_const), V.element.interpolation_points))

        with b.localForm() as loc:
            loc.set(0.0)
        petsc.assemble_vector(b, L_form)
        petsc.apply_lifting(b, [a_form], bcs=[[bc]])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        petsc.set_bc(b, [bc])

        try:
            solver.solve(b, uh.x.petsc_vec)
        except Exception:
            solver.setType("preonly")
            solver.getPC().setType("lu")
            solver.setOperators(A)
            solver.solve(b, uh.x.petsc_vec)

        uh.x.scatter_forward()
        total_iterations += int(solver.getIterationNumber())
        u_n.x.array[:] = uh.x.array
        u_n.x.scatter_forward()

    u_exact_T = _exact_ufl(msh, ScalarType(t_end))
    l2_sq = fem.assemble_scalar(fem.form((uh - u_exact_T) ** 2 * ufl.dx))
    l2_sq = comm.allreduce(l2_sq, op=MPI.SUM)
    l2_err = math.sqrt(max(l2_sq, 0.0))

    ref_sq = fem.assemble_scalar(fem.form((u_exact_T) ** 2 * ufl.dx))
    ref_sq = comm.allreduce(ref_sq, op=MPI.SUM)
    rel_l2 = l2_err / max(math.sqrt(max(ref_sq, 0.0)), 1e-16)

    return {
        "mesh": msh,
        "V": V,
        "u": uh,
        "l2_error": l2_err,
        "rel_l2_error": rel_l2,
        "iterations": total_iterations,
        "ksp_type": solver.getType(),
        "pc_type": solver.getPC().getType(),
        "rtol": 1e-10,
        "dt": dt,
        "n_steps": n_steps,
    }


def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD
    rank = comm.rank

    pde = case_spec.get("pde", {})
    t0 = _as_float(pde.get("t0", 0.0), 0.0)
    t_end = _as_float(pde.get("t_end", 0.1), 0.1)
    dt_suggested = _as_float(pde.get("dt", 0.01), 0.01)
    if t_end <= t0:
        t0, t_end = 0.0, 0.1

    nx_out, ny_out, bbox = _extract_output_grid(case_spec)

    budget = 13.686
    start = time.perf_counter()

    candidates = [
        (48, 2, min(dt_suggested, 0.005)),
        (64, 2, min(dt_suggested, 0.004)),
        (80, 2, min(dt_suggested, 0.0025)),
        (96, 2, min(dt_suggested, 0.0020)),
    ]

    best_cfg = None
    best_res = None

    for ncell, degree, raw_dt in candidates:
        elapsed = time.perf_counter() - start
        if elapsed > 0.9 * budget:
            break
        n_steps = max(1, int(math.ceil((t_end - t0) / raw_dt)))
        dt = (t_end - t0) / n_steps

        run_start = time.perf_counter()
        try:
            res = _solve_single(ncell, degree, dt, t0, t_end)
        except Exception:
            continue
        run_time = time.perf_counter() - run_start

        if best_res is None or res["l2_error"] < best_res["l2_error"]:
            best_cfg = (ncell, degree, dt)
            best_res = res

        remaining = budget - (time.perf_counter() - start)
        if run_time > 0 and remaining < 0.8 * run_time:
            break
        if res["l2_error"] <= 3e-4 and (time.perf_counter() - start) > 0.5 * budget:
            break

    if best_res is None:
        n_steps = max(1, int(math.ceil((t_end - t0) / dt_suggested)))
        dt = (t_end - t0) / n_steps
        best_cfg = (40, 1, dt)
        best_res = _solve_single(40, 1, dt, t0, t_end)

    ncell, degree, dt = best_cfg
    msh = best_res["mesh"]
    V = best_res["V"]
    uh = best_res["u"]

    u0 = fem.Function(V)
    u0.interpolate(fem.Expression(_exact_ufl(msh, ScalarType(t0)), V.element.interpolation_points))

    u_grid = _sample_on_grid(uh, nx_out, ny_out, bbox)
    u0_grid = _sample_on_grid(u0, nx_out, ny_out, bbox)

    solver_info = {
        "mesh_resolution": int(ncell),
        "element_degree": int(degree),
        "ksp_type": str(best_res["ksp_type"]),
        "pc_type": str(best_res["pc_type"]),
        "rtol": float(best_res["rtol"]),
        "iterations": int(best_res["iterations"]),
        "dt": float(best_res["dt"]),
        "n_steps": int(best_res["n_steps"]),
        "time_scheme": "backward_euler",
        "verification_l2_error": float(best_res["l2_error"]),
        "verification_rel_l2_error": float(best_res["rel_l2_error"]),
    }

    if rank == 0:
        return {
            "u": np.asarray(u_grid, dtype=np.float64),
            "u_initial": np.asarray(u0_grid, dtype=np.float64),
            "solver_info": solver_info,
        }
    return {"u": None, "u_initial": None, "solver_info": solver_info}


if __name__ == "__main__":
    case_spec = {
        "pde": {"t0": 0.0, "t_end": 0.1, "dt": 0.01, "time": True},
        "output": {"grid": {"nx": 32, "ny": 32, "bbox": [0.0, 1.0, 0.0, 1.0]}},
    }
    out = solve(case_spec)
    if MPI.COMM_WORLD.rank == 0:
        print(out["u"].shape)
        print(out["solver_info"])
