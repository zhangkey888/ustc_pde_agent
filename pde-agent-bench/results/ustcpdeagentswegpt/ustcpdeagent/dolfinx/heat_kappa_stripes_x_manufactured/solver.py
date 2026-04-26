import math
import time
from typing import Dict, Any

import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
import ufl

from dolfinx import fem, mesh, geometry
from dolfinx.fem import petsc


ScalarType = PETSc.ScalarType


def _make_exact_callable(t: float):
    def exact(x):
        return np.exp(-t) * np.sin(2.0 * np.pi * x[0]) * np.sin(np.pi * x[1])
    return exact


def _probe_function(u_func: fem.Function, points_xyz: np.ndarray) -> np.ndarray:
    domain = u_func.function_space.mesh
    tree = geometry.bb_tree(domain, domain.topology.dim)
    pts = np.asarray(points_xyz, dtype=np.float64)
    cell_candidates = geometry.compute_collisions_points(tree, pts)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, pts)

    local_ids = []
    local_pts = []
    local_cells = []
    for i in range(pts.shape[0]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            local_ids.append(i)
            local_pts.append(pts[i])
            local_cells.append(links[0])

    local_vals = np.full(pts.shape[0], np.nan, dtype=np.float64)
    if local_pts:
        vals = u_func.eval(np.array(local_pts, dtype=np.float64), np.array(local_cells, dtype=np.int32))
        vals = np.asarray(vals).reshape(len(local_pts), -1)[:, 0]
        local_vals[np.array(local_ids, dtype=np.int32)] = vals

    gathered = domain.comm.allgather(local_vals)
    global_vals = np.full(pts.shape[0], np.nan, dtype=np.float64)
    for arr in gathered:
        mask = ~np.isnan(arr)
        global_vals[mask] = arr[mask]
    return global_vals


def _sample_on_grid(u_func: fem.Function, grid_spec: Dict[str, Any]) -> np.ndarray:
    nx = int(grid_spec["nx"])
    ny = int(grid_spec["ny"])
    bbox = grid_spec["bbox"]
    xmin, xmax, ymin, ymax = map(float, bbox)
    xs = np.linspace(xmin, xmax, nx)
    ys = np.linspace(ymin, ymax, ny)
    XX, YY = np.meshgrid(xs, ys, indexing="xy")
    pts = np.column_stack([XX.ravel(), YY.ravel(), np.zeros(nx * ny, dtype=np.float64)])
    vals = _probe_function(u_func, pts)
    return vals.reshape(ny, nx)


def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD
    rank = comm.rank
    t_start_wall = time.perf_counter()

    pde_time = case_spec.get("pde", {}).get("time", {})
    t0 = float(pde_time.get("t0", 0.0))
    t_end = float(pde_time.get("t_end", case_spec.get("t_end", 0.1)))
    dt_suggested = float(pde_time.get("dt", case_spec.get("dt", 0.01)))
    scheme = pde_time.get("scheme", "backward_euler")

    # Use higher accuracy than suggested while staying safely within time budget.
    # Choose dt so that final time is hit exactly.
    base_dt = min(dt_suggested, 0.00125)
    n_steps = max(1, int(math.ceil((t_end - t0) / base_dt)))
    dt = (t_end - t0) / n_steps

    # Spatial discretization: P2 for accuracy with moderate mesh.
    mesh_resolution = 72
    element_degree = 2

    domain = mesh.create_unit_square(
        comm, mesh_resolution, mesh_resolution, cell_type=mesh.CellType.triangle
    )
    V = fem.functionspace(domain, ("Lagrange", element_degree))

    x = ufl.SpatialCoordinate(domain)

    def kappa_expr(xu):
        return 1.0 + 0.5 * ufl.sin(6.0 * ufl.pi * xu[0])

    kappa = kappa_expr(x)

    t_const = fem.Constant(domain, ScalarType(t0))
    dt_const = fem.Constant(domain, ScalarType(dt))

    u_n = fem.Function(V)
    u_n.interpolate(_make_exact_callable(t0))
    u_n.x.scatter_forward()

    u_bc = fem.Function(V)
    u_bc.interpolate(_make_exact_callable(t0))
    u_bc.x.scatter_forward()

    fdim = domain.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda xx: np.ones(xx.shape[1], dtype=bool)
    )
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc = fem.dirichletbc(u_bc, boundary_dofs)

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    u_exact_t = ufl.exp(-t_const) * ufl.sin(2.0 * ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])
    f_expr = (
        -u_exact_t
        - ufl.div(kappa * ufl.grad(u_exact_t))
    )

    a = ((1.0 / dt_const) * u * v + ufl.inner(kappa * ufl.grad(u), ufl.grad(v))) * ufl.dx
    L = ((1.0 / dt_const) * u_n * v + f_expr * v) * ufl.dx

    a_form = fem.form(a)
    L_form = fem.form(L)

    A = petsc.assemble_matrix(a_form, bcs=[bc])
    A.assemble()
    b = petsc.create_vector(L_form.function_spaces)

    solver = PETSc.KSP().create(comm)
    solver.setOperators(A)
    solver.setType("cg")
    solver.getPC().setType("hypre")
    solver.setTolerances(rtol=1e-10, atol=1e-14, max_it=2000)
    solver.setFromOptions()

    uh = fem.Function(V)

    u_initial_grid = _sample_on_grid(u_n, case_spec["output"]["grid"])

    total_iterations = 0
    current_t = t0
    for _ in range(n_steps):
        current_t += dt
        t_const.value = ScalarType(current_t)

        u_bc.interpolate(_make_exact_callable(current_t))
        u_bc.x.scatter_forward()

        with b.localForm() as loc:
            loc.set(0)
        petsc.assemble_vector(b, L_form)
        petsc.apply_lifting(b, [a_form], bcs=[[bc]])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        petsc.set_bc(b, [bc])

        solver.solve(b, uh.x.petsc_vec)
        uh.x.scatter_forward()
        its = solver.getIterationNumber()
        total_iterations += int(max(its, 0))

        u_n.x.array[:] = uh.x.array
        u_n.x.scatter_forward()

    # Accuracy verification against analytical solution at t_end
    u_exact_fun = fem.Function(V)
    u_exact_fun.interpolate(_make_exact_callable(t_end))
    u_exact_fun.x.scatter_forward()

    err_fun = fem.Function(V)
    err_fun.x.array[:] = uh.x.array - u_exact_fun.x.array
    err_fun.x.scatter_forward()

    l2_err_local = fem.assemble_scalar(fem.form(ufl.inner(err_fun, err_fun) * ufl.dx))
    l2_ref_local = fem.assemble_scalar(fem.form(ufl.inner(u_exact_fun, u_exact_fun) * ufl.dx))
    l2_err = math.sqrt(comm.allreduce(l2_err_local, op=MPI.SUM))
    l2_ref = math.sqrt(comm.allreduce(l2_ref_local, op=MPI.SUM))
    rel_l2_err = l2_err / l2_ref if l2_ref > 0 else l2_err

    u_grid = _sample_on_grid(uh, case_spec["output"]["grid"])

    solver_info = {
        "mesh_resolution": mesh_resolution,
        "element_degree": element_degree,
        "ksp_type": solver.getType(),
        "pc_type": solver.getPC().getType(),
        "rtol": 1e-10,
        "iterations": int(total_iterations),
        "dt": float(dt),
        "n_steps": int(n_steps),
        "time_scheme": str(scheme),
        "l2_error": float(l2_err),
        "relative_l2_error": float(rel_l2_err),
        "wall_time_sec": float(time.perf_counter() - t_start_wall),
    }

    result = {
        "u": u_grid,
        "u_initial": u_initial_grid,
        "solver_info": solver_info,
    }

    return result


if __name__ == "__main__":
    case_spec = {
        "pde": {
            "time": {
                "t0": 0.0,
                "t_end": 0.1,
                "dt": 0.01,
                "scheme": "backward_euler",
            }
        },
        "output": {
            "grid": {
                "nx": 64,
                "ny": 64,
                "bbox": [0.0, 1.0, 0.0, 1.0],
            }
        },
    }
    out = solve(case_spec)
    if MPI.COMM_WORLD.rank == 0:
        print(out["u"].shape)
        print(out["solver_info"])
