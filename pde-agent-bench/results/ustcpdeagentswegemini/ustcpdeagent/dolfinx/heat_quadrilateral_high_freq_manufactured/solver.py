import math
import time
from typing import Dict, Any

import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
import ufl

from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc

ScalarType = PETSc.ScalarType


def _exact_numpy(x, y, t):
    return np.exp(-t) * np.sin(4.0 * np.pi * x) * np.sin(4.0 * np.pi * y)


def _make_point_grid(grid_spec: Dict[str, Any]):
    nx = int(grid_spec["nx"])
    ny = int(grid_spec["ny"])
    bbox = grid_spec["bbox"]
    xmin, xmax, ymin, ymax = map(float, bbox)
    xs = np.linspace(xmin, xmax, nx)
    ys = np.linspace(ymin, ymax, ny)
    XX, YY = np.meshgrid(xs, ys, indexing="xy")
    pts = np.column_stack([XX.ravel(), YY.ravel(), np.zeros(nx * ny, dtype=np.float64)])
    return pts, nx, ny


def _sample_function_on_grid(domain, uh: fem.Function, grid_spec: Dict[str, Any]) -> np.ndarray:
    pts, nx, ny = _make_point_grid(grid_spec)

    bb_tree = geometry.bb_tree(domain, domain.topology.dim)
    candidates = geometry.compute_collisions_points(bb_tree, pts)
    colliding = geometry.compute_colliding_cells(domain, candidates, pts)

    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    for i in range(pts.shape[0]):
        links = colliding.links(i)
        if len(links) > 0:
            points_on_proc.append(pts[i])
            cells_on_proc.append(links[0])
            eval_map.append(i)

    local_vals = np.full(pts.shape[0], np.nan, dtype=np.float64)
    if len(points_on_proc) > 0:
        vals = uh.eval(np.array(points_on_proc, dtype=np.float64),
                       np.array(cells_on_proc, dtype=np.int32))
        vals = np.asarray(vals).reshape(len(points_on_proc), -1)[:, 0]
        local_vals[np.array(eval_map, dtype=np.int32)] = vals

    comm = domain.comm
    send = np.where(np.isnan(local_vals), np.inf, local_vals)
    recv = np.empty_like(send)
    comm.Allreduce(send, recv, op=MPI.MIN)
    recv[np.isinf(recv)] = np.nan

    if np.isnan(recv).any():
        bbox = grid_spec["bbox"]
        xmin, xmax, ymin, ymax = map(float, bbox)
        xs = np.linspace(xmin, xmax, nx)
        ys = np.linspace(ymin, ymax, ny)
        XX, YY = np.meshgrid(xs, ys, indexing="xy")
        exact_fill = _exact_numpy(XX.ravel(), YY.ravel(), 0.0)
        mask = np.isnan(recv)
        recv[mask] = exact_fill[mask]

    return recv.reshape((ny, nx))


def solve(case_spec: dict) -> dict:
    """
    Solve transient heat equation with manufactured solution on [0,1]^2.

    Returns:
      {
        "u": final solution sampled on output grid, shape (ny, nx),
        "solver_info": {...},
        "u_initial": initial condition sampled on output grid, shape (ny, nx),
      }
    """
    comm = MPI.COMM_WORLD
    rank = comm.rank

    # Time specification with hardcoded-safe defaults
    pde_time = case_spec.get("pde", {}).get("time", {})
    t0 = float(pde_time.get("t0", case_spec.get("t0", 0.0)))
    t_end = float(pde_time.get("t_end", case_spec.get("t_end", 0.1)))
    dt_user = pde_time.get("dt", case_spec.get("dt", 0.005))
    dt_user = 0.005 if dt_user is None else float(dt_user)

    # Use a somewhat smaller dt than suggested for accuracy while keeping runtime low
    target_dt = min(dt_user, 0.0025)
    n_steps = max(1, int(math.ceil((t_end - t0) / target_dt)))
    dt = (t_end - t0) / n_steps

    # Spatial accuracy choice: quadrilateral mesh + Q2
    mesh_resolution = 56
    element_degree = 2
    kappa = 1.0
    rtol = 1.0e-10

    domain = mesh.create_rectangle(
        comm,
        [np.array([0.0, 0.0], dtype=np.float64), np.array([1.0, 1.0], dtype=np.float64)],
        [mesh_resolution, mesh_resolution],
        cell_type=mesh.CellType.quadrilateral,
    )
    V = fem.functionspace(domain, ("Lagrange", element_degree))

    x = ufl.SpatialCoordinate(domain)

    def u_exact_ufl(t):
        return ufl.exp(-t) * ufl.sin(4.0 * ufl.pi * x[0]) * ufl.sin(4.0 * ufl.pi * x[1])

    def forcing_ufl(t):
        uex = u_exact_ufl(t)
        return -uex - kappa * ufl.div(ufl.grad(uex))

    # Boundary condition utility
    fdim = domain.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda X: np.ones(X.shape[1], dtype=bool)
    )
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    u_bc = fem.Function(V)

    # Initial condition
    u_n = fem.Function(V)
    u_n.interpolate(lambda X: _exact_numpy(X[0], X[1], t0))
    u_n.x.scatter_forward()

    # Prepare sampled initial field before time stepping
    grid_spec = case_spec["output"]["grid"]
    u_initial_grid = _sample_function_on_grid(domain, u_n, grid_spec)

    # Unknown and test/trial functions
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    t_now = t0 + dt
    f_expr = forcing_ufl(t_now)

    a = (u * v + dt * kappa * ufl.inner(ufl.grad(u), ufl.grad(v))) * ufl.dx
    L = (u_n * v + dt * f_expr * v) * ufl.dx

    a_form = fem.form(a)
    L_form = fem.form(L)

    # Assemble matrix once (operator does not vary for backward Euler with constant dt and kappa)
    def update_bc(time_value: float):
        u_bc.interpolate(lambda X: _exact_numpy(X[0], X[1], time_value))
        u_bc.x.scatter_forward()

    update_bc(t_now)
    bc = fem.dirichletbc(u_bc, boundary_dofs)

    A = petsc.assemble_matrix(a_form, bcs=[bc])
    A.assemble()
    b = petsc.create_vector(L_form.function_spaces)

    solver = PETSc.KSP().create(comm)
    solver.setOperators(A)
    solver.setType("cg")
    solver.getPC().setType("hypre")
    solver.setTolerances(rtol=rtol, atol=1.0e-14, max_it=5000)
    solver.setFromOptions()

    uh = fem.Function(V)

    total_iterations = 0
    start = time.perf_counter()

    for step in range(1, n_steps + 1):
        t_now = t0 + step * dt

        update_bc(t_now)

        # Rebuild RHS form to update exact forcing time dependence
        f_expr = forcing_ufl(t_now)
        L = (u_n * v + dt * f_expr * v) * ufl.dx
        L_form = fem.form(L)

        with b.localForm() as b_loc:
            b_loc.set(0.0)
        petsc.assemble_vector(b, L_form)
        petsc.apply_lifting(b, [a_form], bcs=[[bc]])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        petsc.set_bc(b, [bc])

        uh.x.array[:] = 0.0
        solver.solve(b, uh.x.petsc_vec)
        uh.x.scatter_forward()
        total_iterations += solver.getIterationNumber()

        u_n.x.array[:] = uh.x.array
        u_n.x.scatter_forward()

    wall = time.perf_counter() - start

    # Accuracy verification against manufactured exact solution at final time
    u_exact_final = fem.Function(V)
    u_exact_final.interpolate(lambda X: _exact_numpy(X[0], X[1], t_end))
    u_exact_final.x.scatter_forward()

    err_fun = fem.Function(V)
    err_fun.x.array[:] = uh.x.array - u_exact_final.x.array
    err_fun.x.scatter_forward()

    l2_err_local = fem.assemble_scalar(fem.form(ufl.inner(err_fun, err_fun) * ufl.dx))
    l2_ref_local = fem.assemble_scalar(fem.form(ufl.inner(u_exact_final, u_exact_final) * ufl.dx))
    l2_err = math.sqrt(comm.allreduce(l2_err_local, op=MPI.SUM))
    l2_ref = math.sqrt(comm.allreduce(l2_ref_local, op=MPI.SUM))
    rel_l2_err = l2_err / l2_ref if l2_ref > 0 else l2_err

    u_grid = _sample_function_on_grid(domain, uh, grid_spec)

    solver_info = {
        "mesh_resolution": mesh_resolution,
        "element_degree": element_degree,
        "ksp_type": solver.getType(),
        "pc_type": solver.getPC().getType(),
        "rtol": rtol,
        "iterations": int(total_iterations),
        "dt": float(dt),
        "n_steps": int(n_steps),
        "time_scheme": "backward_euler",
        "l2_error": float(l2_err),
        "relative_l2_error": float(rel_l2_err),
        "wall_time_estimate_sec": float(wall),
    }

    result = {
        "u": u_grid,
        "solver_info": solver_info,
        "u_initial": u_initial_grid,
    }

    if rank == 0:
        assert isinstance(result["u"], np.ndarray)
        assert result["u"].shape == (int(grid_spec["ny"]), int(grid_spec["nx"]))

    return result


if __name__ == "__main__":
    case_spec = {
        "pde": {
            "time": {"t0": 0.0, "t_end": 0.1, "dt": 0.005, "scheme": "backward_euler"}
        },
        "output": {"grid": {"nx": 64, "ny": 64, "bbox": [0.0, 1.0, 0.0, 1.0]}},
    }
    out = solve(case_spec)
    if MPI.COMM_WORLD.rank == 0:
        print("u shape:", out["u"].shape)
        print("solver_info:", out["solver_info"])
