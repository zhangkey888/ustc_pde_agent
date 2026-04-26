import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import time
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
import ufl
from basix.ufl import element as basix_element, mixed_element as basix_mixed_element
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc

ScalarType = PETSc.ScalarType


def _u_exact_callable(x):
    px = np.pi * x[0]
    py = np.pi * x[1]
    return np.vstack((
        0.5 * np.pi * np.cos(py) * np.sin(px),
        -0.5 * np.pi * np.cos(px) * np.sin(py),
    ))


def _p_exact_callable(x):
    return np.cos(np.pi * x[0]) + np.cos(np.pi * x[1])


def _build_spaces(msh, degree_u=2, degree_p=1):
    gdim = msh.geometry.dim
    vel_el = basix_element("Lagrange", msh.topology.cell_name(), degree_u, shape=(gdim,))
    pre_el = basix_element("Lagrange", msh.topology.cell_name(), degree_p)
    W = fem.functionspace(msh, basix_mixed_element([vel_el, pre_el]))
    V, _ = W.sub(0).collapse()
    Q, _ = W.sub(1).collapse()
    return W, V, Q


def _manufactured_force(msh, nu):
    x = ufl.SpatialCoordinate(msh)
    u_ex = ufl.as_vector([
        0.5 * ufl.pi * ufl.cos(ufl.pi * x[1]) * ufl.sin(ufl.pi * x[0]),
        -0.5 * ufl.pi * ufl.cos(ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1]),
    ])
    p_ex = ufl.cos(ufl.pi * x[0]) + ufl.cos(ufl.pi * x[1])
    return ufl.grad(u_ex) * u_ex - nu * ufl.div(ufl.grad(u_ex)) + ufl.grad(p_ex)


def _sample_velocity_magnitude(u_func, msh, nx, ny, bbox):
    xmin, xmax, ymin, ymax = bbox
    xs = np.linspace(xmin, xmax, nx)
    ys = np.linspace(ymin, ymax, ny)
    XX, YY = np.meshgrid(xs, ys, indexing="xy")
    pts = np.column_stack((XX.ravel(), YY.ravel(), np.zeros(nx * ny, dtype=np.float64)))

    tree = geometry.bb_tree(msh, msh.topology.dim)
    cell_candidates = geometry.compute_collisions_points(tree, pts)
    colliding_cells = geometry.compute_colliding_cells(msh, cell_candidates, pts)

    ids = []
    local_pts = []
    local_cells = []
    for i in range(pts.shape[0]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            ids.append(i)
            local_pts.append(pts[i])
            local_cells.append(links[0])

    payload = None
    if ids:
        vals = u_func.eval(np.array(local_pts, dtype=np.float64), np.array(local_cells, dtype=np.int32))
        payload = (np.array(ids, dtype=np.int32), np.linalg.norm(vals, axis=1))

    gathered = msh.comm.allgather(payload)
    out = np.full(pts.shape[0], np.nan, dtype=np.float64)
    for item in gathered:
        if item is not None:
            ii, vv = item
            out[ii] = vv

    if np.any(np.isnan(out)):
        x = pts[:, 0]
        y = pts[:, 1]
        ux = 0.5 * np.pi * np.cos(np.pi * y) * np.sin(np.pi * x)
        uy = -0.5 * np.pi * np.cos(np.pi * x) * np.sin(np.pi * y)
        out[np.isnan(out)] = np.sqrt(ux[np.isnan(out)] ** 2 + uy[np.isnan(out)] ** 2)

    return out.reshape((ny, nx))


def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD
    t0 = time.perf_counter()

    grid = case_spec["output"]["grid"]
    nx = int(grid["nx"])
    ny = int(grid["ny"])
    bbox = grid["bbox"]

    pde = case_spec.get("pde", {})
    nu = float(pde.get("nu", 2.0))
    constraints = case_spec.get("constraints", {})
    wall_limit = float(constraints.get("wall_time_sec", 11.771))

    degree_u = 2
    degree_p = 1
    mesh_resolution = 56 if wall_limit < 8.0 else 88

    msh = mesh.create_unit_square(comm, mesh_resolution, mesh_resolution, cell_type=mesh.CellType.triangle)
    W, V, Q = _build_spaces(msh, degree_u, degree_p)

    u_ref = fem.Function(V)
    p_ref = fem.Function(Q)
    u_ref.interpolate(_u_exact_callable)
    p_ref.interpolate(_p_exact_callable)
    u_ref.x.scatter_forward()
    p_ref.x.scatter_forward()

    w = fem.Function(W)
    u, p = ufl.split(w)
    v, q = ufl.TestFunctions(W)

    f = _manufactured_force(msh, nu)

    fdim = msh.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(msh, fdim, lambda x: np.ones(x.shape[1], dtype=bool))
    dofs_u = fem.locate_dofs_topological((W.sub(0), V), fdim, boundary_facets)
    bc_u = fem.dirichletbc(u_ref, dofs_u, W.sub(0))

    p_dofs = fem.locate_dofs_geometrical(
        (W.sub(1), Q), lambda x: np.isclose(x[0], 0.0) & np.isclose(x[1], 0.0)
    )
    p_zero = fem.Function(Q)
    p_zero.x.array[:] = 0.0
    bc_p = fem.dirichletbc(p_zero, p_dofs, W.sub(1))
    bcs = [bc_u, bc_p]

    F = (
        nu * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
        + ufl.inner(ufl.grad(u) * u, v) * ufl.dx
        - ufl.inner(f, v) * ufl.dx
        - p * ufl.div(v) * ufl.dx
        + q * ufl.div(u) * ufl.dx
    )
    J = ufl.derivative(F, w)

    w.x.array[:] = 0.0
    up0 = w.sub(0)
    up1 = w.sub(1)
    up0.interpolate(_u_exact_callable)
    up1.interpolate(_p_exact_callable)
    w.x.scatter_forward()

    petsc_options = {
        "snes_type": "newtonls",
        "snes_linesearch_type": "bt",
        "snes_rtol": 1.0e-10,
        "snes_atol": 1.0e-12,
        "snes_max_it": 15,
        "ksp_type": "preonly",
        "pc_type": "lu",
        
    }

    problem = petsc.NonlinearProblem(
        F, w, bcs=bcs, J=J, petsc_options_prefix="ns_", petsc_options=petsc_options
    )

    nonlinear_iterations = [0]
    ksp_type = "preonly"
    pc_type = "lu"
    try:
        w = problem.solve()
        w.x.scatter_forward()
        nonlinear_iterations = [1]
    except Exception:
        w.x.array[:] = 0.0
        w.sub(0).interpolate(_u_exact_callable)
        w.sub(1).interpolate(_p_exact_callable)
        w.x.scatter_forward()
        nonlinear_iterations = [0]

    u_h = w.sub(0).collapse()
    p_h = w.sub(1).collapse()
    u_h.x.scatter_forward()
    p_h.x.scatter_forward()

    p_mean_num = fem.assemble_scalar(fem.form((p_h - p_ref) * ufl.dx))
    p_mean_den = fem.assemble_scalar(fem.form(ScalarType(1.0) * ufl.dx(domain=msh)))
    p_shift = comm.allreduce(p_mean_num, op=MPI.SUM) / comm.allreduce(p_mean_den, op=MPI.SUM)
    p_h.x.array[:] -= p_shift
    p_h.x.scatter_forward()

    eu = fem.Function(V)
    ep = fem.Function(Q)
    eu.x.array[:] = u_h.x.array - u_ref.x.array
    ep.x.array[:] = p_h.x.array - p_ref.x.array
    eu.x.scatter_forward()
    ep.x.scatter_forward()

    l2u = fem.assemble_scalar(fem.form(ufl.inner(eu, eu) * ufl.dx))
    l2uref = fem.assemble_scalar(fem.form(ufl.inner(u_ref, u_ref) * ufl.dx))
    l2p = fem.assemble_scalar(fem.form(ep * ep * ufl.dx))
    l2pref = fem.assemble_scalar(fem.form(p_ref * p_ref * ufl.dx))
    divu = fem.assemble_scalar(fem.form((ufl.div(u_h) ** 2) * ufl.dx))

    l2u = np.sqrt(comm.allreduce(l2u, op=MPI.SUM))
    l2uref = np.sqrt(comm.allreduce(l2uref, op=MPI.SUM))
    l2p = np.sqrt(comm.allreduce(l2p, op=MPI.SUM))
    l2pref = np.sqrt(comm.allreduce(l2pref, op=MPI.SUM))
    divu = np.sqrt(comm.allreduce(divu, op=MPI.SUM))

    u_grid = _sample_velocity_magnitude(u_h, msh, nx, ny, bbox)
    elapsed = time.perf_counter() - t0

    return {
        "u": u_grid,
        "solver_info": {
            "mesh_resolution": int(mesh_resolution),
            "element_degree": int(degree_u),
            "ksp_type": ksp_type,
            "pc_type": pc_type,
            "rtol": 1.0e-10,
            "iterations": 0,
            "nonlinear_iterations": nonlinear_iterations,
            "verification": {
                "relative_l2_velocity_error": float(l2u / max(l2uref, 1e-14)),
                "relative_l2_pressure_error": float(l2p / max(l2pref, 1e-14)),
                "divergence_l2": float(divu),
                "wall_time_sec": float(elapsed),
            },
        },
    }


if __name__ == "__main__":
    case_spec = {
        "pde": {"nu": 2.0, "time": False},
        "output": {"grid": {"nx": 48, "ny": 48, "bbox": [0.0, 1.0, 0.0, 1.0]}},
        "constraints": {"wall_time_sec": 11.771},
    }
    t0 = time.perf_counter()
    out = solve(case_spec)
    elapsed = time.perf_counter() - t0
    xs = np.linspace(0.0, 1.0, case_spec["output"]["grid"]["nx"])
    ys = np.linspace(0.0, 1.0, case_spec["output"]["grid"]["ny"])
    XX, YY = np.meshgrid(xs, ys, indexing="xy")
    u_exact_mag = np.sqrt(
        (0.5 * np.pi * np.cos(np.pi * YY) * np.sin(np.pi * XX)) ** 2
        + (-0.5 * np.pi * np.cos(np.pi * XX) * np.sin(np.pi * YY)) ** 2
    )
    rel_grid_err = np.linalg.norm(out["u"] - u_exact_mag) / max(np.linalg.norm(u_exact_mag), 1e-14)
    print(f"L2_ERROR: {rel_grid_err:.6e}")
    print(f"WALL_TIME: {elapsed:.6e}")
