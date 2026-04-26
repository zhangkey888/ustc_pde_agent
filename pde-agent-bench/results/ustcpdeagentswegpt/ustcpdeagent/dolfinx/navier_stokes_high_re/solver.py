import time
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
import ufl

from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc


ScalarType = PETSc.ScalarType


def _manufactured_u_expr(x):
    return ufl.as_vector(
        [
            ufl.pi * ufl.cos(ufl.pi * x[1]) * ufl.sin(ufl.pi * x[0]),
            -ufl.pi * ufl.cos(ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1]),
        ]
    )


def _forcing_expr(msh, nu):
    x = ufl.SpatialCoordinate(msh)
    uex = _manufactured_u_expr(x)
    return ufl.grad(uex) * uex - nu * ufl.div(ufl.grad(uex))


def _sample_velocity_magnitude(u_func, msh, nx, ny, bbox):
    xmin, xmax, ymin, ymax = bbox
    xs = np.linspace(xmin, xmax, nx)
    ys = np.linspace(ymin, ymax, ny)
    XX, YY = np.meshgrid(xs, ys, indexing="xy")
    pts = np.zeros((nx * ny, 3), dtype=np.float64)
    pts[:, 0] = XX.ravel()
    pts[:, 1] = YY.ravel()

    tree = geometry.bb_tree(msh, msh.topology.dim)
    cell_candidates = geometry.compute_collisions_points(tree, pts)
    colliding_cells = geometry.compute_colliding_cells(msh, cell_candidates, pts)

    local_mag = np.full((nx * ny,), np.nan, dtype=np.float64)
    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    for i in range(pts.shape[0]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(pts[i])
            cells_on_proc.append(links[0])
            eval_map.append(i)

    if points_on_proc:
        vals = u_func.eval(
            np.asarray(points_on_proc, dtype=np.float64),
            np.asarray(cells_on_proc, dtype=np.int32),
        )
        local_mag[np.asarray(eval_map, dtype=np.int32)] = np.linalg.norm(vals, axis=1)

    gathered = msh.comm.gather(local_mag, root=0)
    if msh.comm.rank == 0:
        merged = np.full((nx * ny,), np.nan, dtype=np.float64)
        for arr in gathered:
            mask = np.isnan(merged) & ~np.isnan(arr)
            merged[mask] = arr[mask]
        out = np.nan_to_num(merged, nan=0.0).reshape(ny, nx)
    else:
        out = None
    return msh.comm.bcast(out, root=0)


def _compute_errors(u_sol, msh):
    x = ufl.SpatialCoordinate(msh)
    uex = _manufactured_u_expr(x)
    l2_form = fem.form(ufl.inner(u_sol - uex, u_sol - uex) * ufl.dx)
    h1_form = fem.form(ufl.inner(ufl.grad(u_sol - uex), ufl.grad(u_sol - uex)) * ufl.dx)
    div_form = fem.form((ufl.div(u_sol)) ** 2 * ufl.dx)
    l2_local = fem.assemble_scalar(l2_form)
    h1_local = fem.assemble_scalar(h1_form)
    div_local = fem.assemble_scalar(div_form)
    l2 = np.sqrt(msh.comm.allreduce(l2_local, op=MPI.SUM))
    h1 = np.sqrt(msh.comm.allreduce(h1_local, op=MPI.SUM))
    divn = np.sqrt(msh.comm.allreduce(div_local, op=MPI.SUM))
    return float(l2), float(h1), float(divn)


def _solve_single_resolution(comm, nu, mesh_resolution, degree=2):
    msh = mesh.create_unit_square(comm, mesh_resolution, mesh_resolution, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(msh, ("Lagrange", degree, (msh.geometry.dim,)))

    x = ufl.SpatialCoordinate(msh)
    u_exact = _manufactured_u_expr(x)
    f_expr = _forcing_expr(msh, nu)

    u_bc = fem.Function(V)
    u_bc.interpolate(fem.Expression(u_exact, V.element.interpolation_points))

    fdim = msh.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(msh, fdim, lambda xx: np.ones(xx.shape[1], dtype=bool))
    bc = fem.dirichletbc(u_bc, fem.locate_dofs_topological(V, fdim, boundary_facets))

    uh = fem.Function(V)
    uh.interpolate(u_bc)

    v = ufl.TestFunction(V)
    F = (
        nu * ufl.inner(ufl.grad(uh), ufl.grad(v)) * ufl.dx
        + ufl.inner(ufl.grad(uh) * uh, v) * ufl.dx
        - ufl.inner(f_expr, v) * ufl.dx
    )
    J = ufl.derivative(F, uh)

    problem = petsc.NonlinearProblem(
        F,
        uh,
        bcs=[bc],
        J=J,
        petsc_options_prefix=f"velns_{mesh_resolution}_",
        petsc_options={
            "snes_type": "newtonls",
            "snes_linesearch_type": "bt",
            "snes_rtol": 1.0e-10,
            "snes_atol": 1.0e-12,
            "snes_max_it": 20,
            "ksp_type": "preonly",
            "pc_type": "lu",
        },
    )
    uh = problem.solve()
    uh.x.scatter_forward()

    nonlinear_iterations = [0]
    linear_iterations = 0
    ksp_type = "preonly"
    pc_type = "lu"
    try:
        snes = problem.solver
        nonlinear_iterations = [int(snes.getIterationNumber())]
        linear_iterations = int(snes.getLinearSolveIterations())
        ksp_type = str(snes.getKSP().getType())
        pc_type = str(snes.getKSP().getPC().getType())
    except Exception:
        pass

    l2_err, h1_err, div_err = _compute_errors(uh, msh)
    return {
        "mesh": msh,
        "u_sol": uh,
        "mesh_resolution": mesh_resolution,
        "element_degree": degree,
        "ksp_type": ksp_type,
        "pc_type": pc_type,
        "rtol": 1.0e-10,
        "iterations": linear_iterations,
        "nonlinear_iterations": nonlinear_iterations,
        "l2_error": l2_err,
        "h1_error": h1_err,
        "div_error": div_err,
    }


def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD
    t0 = time.perf_counter()

    nu = float(case_spec.get("pde", {}).get("nu", case_spec.get("viscosity", 0.02)))
    grid = case_spec["output"]["grid"]
    nx_out = int(grid["nx"])
    ny_out = int(grid["ny"])
    bbox = grid["bbox"]

    candidate_resolutions = [48, 72, 96]
    best = None
    for i, res in enumerate(candidate_resolutions):
        best = _solve_single_resolution(comm, nu, res, degree=2)
        if i < len(candidate_resolutions) - 1 and (time.perf_counter() - t0) > 120.0:
            break

    u_grid = _sample_velocity_magnitude(best["u_sol"], best["mesh"], nx_out, ny_out, bbox)
    solver_info = {
        "mesh_resolution": int(best["mesh_resolution"]),
        "element_degree": int(best["element_degree"]),
        "ksp_type": str(best["ksp_type"]),
        "pc_type": str(best["pc_type"]),
        "rtol": float(best["rtol"]),
        "iterations": int(best["iterations"]),
        "nonlinear_iterations": [int(v) for v in best["nonlinear_iterations"]],
        "l2_error": float(best["l2_error"]),
        "h1_error": float(best["h1_error"]),
        "divergence_l2": float(best["div_error"]),
        "wall_time_sec": float(time.perf_counter() - t0),
    }
    return {"u": u_grid, "solver_info": solver_info}
