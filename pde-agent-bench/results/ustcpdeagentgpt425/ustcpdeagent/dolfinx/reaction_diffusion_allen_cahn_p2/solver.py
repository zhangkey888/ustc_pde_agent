import math
import time
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
import ufl
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc

ScalarType = PETSc.ScalarType

# DIAGNOSIS
# equation_type: reaction_diffusion
# spatial_dim: 2
# domain_geometry: rectangle
# unknowns: scalar
# coupling: none
# linearity: nonlinear
# time_dependence: transient
# stiffness: stiff
# dominant_physics: mixed
# peclet_or_reynolds: N/A
# solution_regularity: smooth
# bc_type: all_dirichlet
# special_notes: manufactured_solution

# METHOD
# spatial_method: fem
# element_or_basis: Lagrange_P2
# stabilization: none
# time_method: backward_euler
# nonlinear_solver: newton
# linear_solver: gmres
# preconditioner: ilu
# special_treatment: none
# pde_skill: reaction_diffusion


def _parse_time(case_spec):
    time_spec = case_spec.get("pde", {}).get("time", {})
    t0 = float(time_spec.get("t0", 0.0))
    t_end = float(time_spec.get("t_end", 0.2))
    dt = float(time_spec.get("dt", 0.005))
    scheme = str(time_spec.get("scheme", "backward_euler")).lower()
    return t0, t_end, dt, scheme


def _build_exact(msh):
    x = ufl.SpatialCoordinate(msh)
    t = fem.Constant(msh, ScalarType(0.0))
    u_exact = ufl.exp(-t) * (ScalarType(0.25) * ufl.sin(2 * ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1]))
    return t, u_exact


def _epsilon(case_spec):
    pde = case_spec.get("pde", {})
    return float(pde.get("epsilon", pde.get("eps", 0.02)))


def _reaction_expr(u):
    return u**3 - u


def _source_expr(u_exact, t_const, eps):
    return (-u_exact) - eps * ufl.div(ufl.grad(u_exact)) + _reaction_expr(u_exact)


def _interp_expr(V, expr):
    f = fem.Function(V)
    f.interpolate(fem.Expression(expr, V.element.interpolation_points))
    return f


def _sample_function(u_func, nx, ny, bbox):
    msh = u_func.function_space.mesh
    xs = np.linspace(bbox[0], bbox[1], nx)
    ys = np.linspace(bbox[2], bbox[3], ny)
    XX, YY = np.meshgrid(xs, ys, indexing="xy")
    pts = np.column_stack([XX.ravel(), YY.ravel(), np.zeros(nx * ny, dtype=np.float64)])

    tree = geometry.bb_tree(msh, msh.topology.dim)
    cell_candidates = geometry.compute_collisions_points(tree, pts)
    colliding_cells = geometry.compute_colliding_cells(msh, cell_candidates, pts)

    values_local = np.full(nx * ny, np.nan, dtype=np.float64)
    points_on_proc = []
    cells_on_proc = []
    ids = []
    for i in range(pts.shape[0]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(pts[i])
            cells_on_proc.append(links[0])
            ids.append(i)

    if points_on_proc:
        vals = u_func.eval(np.array(points_on_proc, dtype=np.float64), np.array(cells_on_proc, dtype=np.int32))
        values_local[np.array(ids, dtype=np.int32)] = np.asarray(vals).reshape(-1)

    comm = msh.comm
    gathered = comm.gather(values_local, root=0)
    if comm.rank == 0:
        values = np.full(nx * ny, np.nan, dtype=np.float64)
        for arr in gathered:
            mask = ~np.isnan(arr)
            values[mask] = arr[mask]
        if np.isnan(values).any():
            raise RuntimeError("Point evaluation failed for some sample points.")
        return values.reshape(ny, nx)
    return None


def _compute_l2_error(u_num, u_ex):
    comm = u_num.function_space.mesh.comm
    e2 = fem.assemble_scalar(fem.form((u_num - u_ex) ** 2 * ufl.dx))
    r2 = fem.assemble_scalar(fem.form((u_ex) ** 2 * ufl.dx))
    e2 = comm.allreduce(e2, op=MPI.SUM)
    r2 = comm.allreduce(r2, op=MPI.SUM)
    return math.sqrt(max(e2, 0.0)), math.sqrt(max(r2, 0.0))


def _solve_config(case_spec, mesh_resolution, degree, dt, eps, ksp_type, pc_type, rtol):
    comm = MPI.COMM_WORLD
    msh = mesh.create_unit_square(comm, mesh_resolution, mesh_resolution, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(msh, ("Lagrange", degree))

    t0, t_end, _, _ = _parse_time(case_spec)
    n_steps = int(round((t_end - t0) / dt))

    t_const, u_exact_expr = _build_exact(msh)
    f_expr = _source_expr(u_exact_expr, t_const, ScalarType(eps))

    u_n = fem.Function(V)
    u = fem.Function(V)
    v = ufl.TestFunction(V)

    t_const.value = ScalarType(t0)
    u0 = _interp_expr(V, u_exact_expr)
    u_n.x.array[:] = u0.x.array
    u_n.x.scatter_forward()
    u.x.array[:] = u_n.x.array
    u.x.scatter_forward()

    fdim = msh.topology.dim - 1
    facets = mesh.locate_entities_boundary(msh, fdim, lambda x: np.ones(x.shape[1], dtype=bool))
    bdofs = fem.locate_dofs_topological(V, fdim, facets)
    u_bc = fem.Function(V)

    dt_const = fem.Constant(msh, ScalarType(dt))
    F = ((u - u_n) / dt_const) * v * ufl.dx + ScalarType(eps) * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx + _reaction_expr(u) * v * ufl.dx - f_expr * v * ufl.dx
    J = ufl.derivative(F, u)

    nonlinear_iterations = []
    linear_iterations = 0

    grid = case_spec["output"]["grid"]
    nx_out = int(grid["nx"])
    ny_out = int(grid["ny"])
    bbox = tuple(grid["bbox"])
    u_initial = _sample_function(u_n, nx_out, ny_out, bbox)

    for step in range(1, n_steps + 1):
        t_const.value = ScalarType(t0 + step * dt)
        u_bc.interpolate(fem.Expression(u_exact_expr, V.element.interpolation_points))
        bc = fem.dirichletbc(u_bc, bdofs)

        problem = petsc.NonlinearProblem(
            F, u, bcs=[bc], J=J,
            petsc_options_prefix=f"rd_{mesh_resolution}_{step}_",
            petsc_options={
                "snes_type": "newtonls",
                "snes_linesearch_type": "bt",
                "snes_rtol": 1e-10,
                "snes_atol": 1e-12,
                "snes_max_it": 25,
                "ksp_type": ksp_type,
                "ksp_rtol": rtol,
                "pc_type": pc_type,
            },
        )
        u = problem.solve()
        u.x.scatter_forward()

        snes = problem.solver
        nonlinear_iterations.append(int(snes.getIterationNumber()))
        linear_iterations += int(snes.getLinearSolveIterations())

        u_n.x.array[:] = u.x.array
        u_n.x.scatter_forward()

    t_const.value = ScalarType(t_end)
    u_exact_final = _interp_expr(V, u_exact_expr)
    l2_error, l2_ref = _compute_l2_error(u, u_exact_final)
    u_grid = _sample_function(u, nx_out, ny_out, bbox)

    return {
        "u": u_grid,
        "u_initial": u_initial,
        "solver_info": {
            "mesh_resolution": int(mesh_resolution),
            "element_degree": int(degree),
            "ksp_type": str(ksp_type),
            "pc_type": str(pc_type),
            "rtol": float(rtol),
            "iterations": int(linear_iterations),
            "dt": float(dt),
            "n_steps": int(n_steps),
            "time_scheme": "backward_euler",
            "nonlinear_iterations": [int(it) for it in nonlinear_iterations],
            "l2_error": float(l2_error),
            "relative_l2_error": float(l2_error / l2_ref if l2_ref > 0 else l2_error),
        },
    }


def solve(case_spec: dict) -> dict:
    start = time.time()
    t0, t_end, dt_suggested, scheme = _parse_time(case_spec)
    if scheme != "backward_euler":
        dt_suggested = dt_suggested

    eps = _epsilon(case_spec)

    configs = [
        (48, 2, min(dt_suggested, 0.005), "gmres", "ilu", 1e-9),
        (64, 2, min(dt_suggested, 0.004), "gmres", "ilu", 1e-10),
    ]

    best = None
    for i, cfg in enumerate(configs):
        result = _solve_config(case_spec, *cfg, eps=eps) if False else None
        break

    # Explicit calls for compatibility with keyword ordering
    result = _solve_config(case_spec, configs[0][0], configs[0][1], configs[0][2], eps, configs[0][3], configs[0][4], configs[0][5])
    elapsed = time.time() - start
    if elapsed < 25.0:
        result = _solve_config(case_spec, configs[1][0], configs[1][1], configs[1][2], eps, configs[1][3], configs[1][4], configs[1][5])
    if elapsed < 25.0 and result["solver_info"]["l2_error"] > 1.0e-3:
        result = _solve_config(case_spec, configs[1][0], configs[1][1], configs[1][2], eps, configs[1][3], configs[1][4], configs[1][5])

    return result


if __name__ == "__main__":
    case = {
        "case_id": "reaction_diffusion_allen_cahn_p2",
        "pde": {
            "type": "reaction_diffusion",
            "time": {"t0": 0.0, "t_end": 0.2, "dt": 0.005, "scheme": "backward_euler"},
            "epsilon": 0.02,
        },
        "output": {"grid": {"nx": 32, "ny": 32, "bbox": [0.0, 1.0, 0.0, 1.0]}},
    }
    out = solve(case)
    if MPI.COMM_WORLD.rank == 0:
        print(out["u"].shape)
        print(out["solver_info"])
