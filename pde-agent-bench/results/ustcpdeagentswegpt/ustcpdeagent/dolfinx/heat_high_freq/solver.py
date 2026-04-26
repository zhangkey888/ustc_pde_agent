import math
import time
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl

# ```DIAGNOSIS
# equation_type:        heat
# spatial_dim:          2
# domain_geometry:      rectangle
# unknowns:             scalar
# coupling:             none
# linearity:            linear
# time_dependence:      transient
# stiffness:            stiff
# dominant_physics:     diffusion
# peclet_or_reynolds:   N/A
# solution_regularity:  smooth
# bc_type:              all_dirichlet
# special_notes:        manufactured_solution
# ```
#
# ```METHOD
# spatial_method:       fem
# element_or_basis:     Lagrange_P2
# stabilization:        none
# time_method:          backward_euler
# nonlinear_solver:     none
# linear_solver:        cg
# preconditioner:       hypre
# special_treatment:    none
# pde_skill:            heat
# ```

ScalarType = PETSc.ScalarType


def _exact_value(x, t):
    return np.exp(-t) * np.sin(4.0 * np.pi * x[0]) * np.sin(4.0 * np.pi * x[1])


def _rhs_value(x, t, kappa=1.0):
    s = np.sin(4.0 * np.pi * x[0]) * np.sin(4.0 * np.pi * x[1])
    return np.exp(-t) * (-1.0 + 32.0 * (np.pi ** 2) * kappa) * s


def _probe_function(u_func, points):
    domain = u_func.function_space.mesh
    tree = geometry.bb_tree(domain, domain.topology.dim)
    candidates = geometry.compute_collisions_points(tree, points)
    colliding = geometry.compute_colliding_cells(domain, candidates, points)

    values = np.full(points.shape[0], np.nan, dtype=np.float64)
    pts_local = []
    cells_local = []
    idx_map = []
    for i in range(points.shape[0]):
        links = colliding.links(i)
        if len(links) > 0:
            pts_local.append(points[i])
            cells_local.append(links[0])
            idx_map.append(i)

    if pts_local:
        vals = u_func.eval(np.asarray(pts_local, dtype=np.float64), np.asarray(cells_local, dtype=np.int32))
        values[np.asarray(idx_map, dtype=np.int32)] = np.asarray(vals).reshape(-1)

    if domain.comm.size > 1:
        recv = np.empty_like(values)
        mask = np.where(np.isnan(values), -1.0e300, values)
        domain.comm.Allreduce(mask, recv, op=MPI.MAX)
        values = recv
    return values


def _sample_on_grid(u_func, grid_spec):
    nx = int(grid_spec["nx"])
    ny = int(grid_spec["ny"])
    bbox = grid_spec["bbox"]
    xs = np.linspace(bbox[0], bbox[1], nx)
    ys = np.linspace(bbox[2], bbox[3], ny)
    XX, YY = np.meshgrid(xs, ys, indexing="xy")
    pts = np.column_stack([XX.ravel(), YY.ravel(), np.zeros(nx * ny, dtype=np.float64)])
    vals = _probe_function(u_func, pts)
    vals = np.where(np.isnan(vals), 0.0, vals)
    return vals.reshape(ny, nx)


def _run_simulation(mesh_resolution, degree, dt, t0, t_end, kappa=1.0, ksp_type="cg", pc_type="hypre", rtol=1.0e-10):
    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_square(comm, mesh_resolution, mesh_resolution, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", degree))

    u_bc = fem.Function(V)
    u_n = fem.Function(V)
    u_bc.interpolate(lambda x: _exact_value(x, t0))
    u_n.interpolate(lambda x: _exact_value(x, t0))

    fdim = domain.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool))
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc = fem.dirichletbc(u_bc, boundary_dofs)

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    dt_c = fem.Constant(domain, ScalarType(dt))
    kappa_c = fem.Constant(domain, ScalarType(kappa))
    f_fun = fem.Function(V)

    a = (u * v + dt_c * kappa_c * ufl.inner(ufl.grad(u), ufl.grad(v))) * ufl.dx
    L = (u_n * v + dt_c * f_fun * v) * ufl.dx

    a_form = fem.form(a)
    L_form = fem.form(L)

    A = petsc.assemble_matrix(a_form, bcs=[bc])
    A.assemble()
    b = petsc.create_vector(L_form.function_spaces)

    solver = PETSc.KSP().create(comm)
    solver.setOperators(A)
    solver.setType(ksp_type)
    solver.getPC().setType(pc_type)
    solver.setTolerances(rtol=rtol)
    try:
        solver.setNormType(PETSc.KSP.NormType.PRECONDITIONED)
    except Exception:
        pass
    solver.setFromOptions()

    uh = fem.Function(V)
    total_iterations = 0
    n_steps = max(1, int(round((t_end - t0) / dt)))
    dt_used = (t_end - t0) / n_steps
    dt_c.value = ScalarType(dt_used)
    t = t0

    for _ in range(n_steps):
        t += dt_used
        u_bc.interpolate(lambda x, tt=t: _exact_value(x, tt))
        f_fun.interpolate(lambda x, tt=t: _rhs_value(x, tt, kappa))

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
        its = solver.getIterationNumber()
        if its >= 0:
            total_iterations += int(its)
        u_n.x.array[:] = uh.x.array
        u_n.x.scatter_forward()

    x = ufl.SpatialCoordinate(domain)
    u_exact_expr = ufl.exp(-t_end) * ufl.sin(4.0 * ufl.pi * x[0]) * ufl.sin(4.0 * ufl.pi * x[1])
    err_sq = fem.assemble_scalar(fem.form((uh - u_exact_expr) ** 2 * ufl.dx))
    exact_sq = fem.assemble_scalar(fem.form((u_exact_expr) ** 2 * ufl.dx))
    l2_err = math.sqrt(comm.allreduce(err_sq, op=MPI.SUM))
    l2_exact = math.sqrt(comm.allreduce(exact_sq, op=MPI.SUM))
    rel_l2_err = l2_err / l2_exact if l2_exact > 0 else l2_err

    info = {
        "mesh_resolution": int(mesh_resolution),
        "element_degree": int(degree),
        "ksp_type": str(solver.getType()),
        "pc_type": str(solver.getPC().getType()),
        "rtol": float(rtol),
        "iterations": int(total_iterations),
        "dt": float(dt_used),
        "n_steps": int(n_steps),
        "time_scheme": "backward_euler",
        "l2_error": float(l2_err),
        "relative_l2_error": float(rel_l2_err),
    }
    return uh, info


def solve(case_spec: dict) -> dict:
    pde = case_spec.get("pde", {})
    output_grid = case_spec["output"]["grid"]

    t0 = float(pde.get("t0", 0.0))
    t_end = float(pde.get("t_end", 0.1))
    if t_end <= t0:
        t0 = 0.0
        t_end = 0.1
    dt_suggested = float(pde.get("dt", 0.005))
    scheme = str(pde.get("scheme", "backward_euler"))
    if scheme != "backward_euler":
        scheme = "backward_euler"

    budget = 7.981
    started = time.perf_counter()

    candidates = [
        (48, 2, min(dt_suggested, 0.005)),
        (64, 2, min(dt_suggested, 0.0025)),
        (80, 2, min(dt_suggested, 0.00125)),
    ]

    best_u = None
    best_info = None

    for mesh_resolution, degree, dt in candidates:
        run_start = time.perf_counter()
        uh, info = _run_simulation(
            mesh_resolution=mesh_resolution,
            degree=degree,
            dt=dt,
            t0=t0,
            t_end=t_end,
            kappa=1.0,
            ksp_type="cg",
            pc_type="hypre",
            rtol=1.0e-10,
        )
        run_time = time.perf_counter() - run_start
        info["wall_time_sec_estimate"] = float(run_time)

        best_u = uh
        best_info = info

        elapsed = time.perf_counter() - started
        remaining = budget - elapsed
        if run_time < 0.45 * budget and remaining > 1.5 * max(run_time, 1.0e-6):
            continue
        break

    u_grid = _sample_on_grid(best_u, output_grid)

    nx = int(output_grid["nx"])
    ny = int(output_grid["ny"])
    bbox = output_grid["bbox"]
    xs = np.linspace(bbox[0], bbox[1], nx)
    ys = np.linspace(bbox[2], bbox[3], ny)
    XX, YY = np.meshgrid(xs, ys, indexing="xy")
    u_initial = np.exp(-t0) * np.sin(4.0 * np.pi * XX) * np.sin(4.0 * np.pi * YY)

    return {
        "u": np.asarray(u_grid, dtype=np.float64).reshape(ny, nx),
        "u_initial": np.asarray(u_initial, dtype=np.float64).reshape(ny, nx),
        "solver_info": best_info,
    }


if __name__ == "__main__":
    case_spec = {
        "pde": {"t0": 0.0, "t_end": 0.1, "dt": 0.005, "scheme": "backward_euler", "time": True},
        "output": {"grid": {"nx": 64, "ny": 64, "bbox": [0.0, 1.0, 0.0, 1.0]}},
    }
    result = solve(case_spec)
    if MPI.COMM_WORLD.rank == 0:
        print(result["u"].shape)
        print(result["solver_info"])
