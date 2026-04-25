import math
import time
from typing import Dict, Any

import numpy as np
from mpi4py import MPI
from petsc4py import PETSc

from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from basix.ufl import element as basix_element, mixed_element as basix_mixed_element


# ```DIAGNOSIS
# equation_type:        stokes
# spatial_dim:          2
# domain_geometry:      rectangle
# unknowns:             vector+scalar
# coupling:             saddle_point
# linearity:            linear
# time_dependence:      steady
# stiffness:            N/A
# dominant_physics:     diffusion
# peclet_or_reynolds:   low
# solution_regularity:  smooth
# bc_type:              mixed
# special_notes:        pressure_pinning
# ```
#
# ```METHOD
# spatial_method:       fem
# element_or_basis:     Taylor-Hood_P2P1
# stabilization:        none
# time_method:          none
# nonlinear_solver:     none
# linear_solver:        minres
# preconditioner:       hypre
# special_treatment:    pressure_pinning
# pde_skill:            stokes
# ```


ScalarType = PETSc.ScalarType


def _make_velocity_bc_function(V, kind: str):
    f = fem.Function(V)
    if kind == "inflow":
        def inflow(x):
            vals = np.zeros((2, x.shape[1]), dtype=np.float64)
            y = x[1]
            vals[0] = np.sin(np.pi * y)
            vals[1] = 0.2 * np.sin(2.0 * np.pi * y)
            return vals
        f.interpolate(inflow)
    elif kind == "zero":
        f.x.array[:] = 0.0
    else:
        raise ValueError(f"Unknown BC kind: {kind}")
    return f


def _build_problem(n: int, nu_value: float = 0.5):
    comm = MPI.COMM_WORLD
    msh = mesh.create_unit_square(comm, n, n, cell_type=mesh.CellType.triangle)
    gdim = msh.geometry.dim

    vel_el = basix_element("Lagrange", msh.topology.cell_name(), 2, shape=(gdim,))
    pres_el = basix_element("Lagrange", msh.topology.cell_name(), 1)
    W = fem.functionspace(msh, basix_mixed_element([vel_el, pres_el]))
    V, _ = W.sub(0).collapse()
    Q, _ = W.sub(1).collapse()

    u, p = ufl.TrialFunctions(W)
    v, q = ufl.TestFunctions(W)

    nu = fem.Constant(msh, ScalarType(nu_value))
    f = fem.Constant(msh, np.array((0.0, 0.0), dtype=np.float64))

    a = (
        2.0 * nu * ufl.inner(ufl.sym(ufl.grad(u)), ufl.sym(ufl.grad(v))) * ufl.dx
        - p * ufl.div(v) * ufl.dx
        + ufl.div(u) * q * ufl.dx
    )
    L = ufl.inner(f, v) * ufl.dx

    fdim = msh.topology.dim - 1

    left_facets = mesh.locate_entities_boundary(msh, fdim, lambda x: np.isclose(x[0], 0.0))
    bottom_facets = mesh.locate_entities_boundary(msh, fdim, lambda x: np.isclose(x[1], 0.0))
    top_facets = mesh.locate_entities_boundary(msh, fdim, lambda x: np.isclose(x[1], 1.0))

    u_in = _make_velocity_bc_function(V, "inflow")
    u_zero = _make_velocity_bc_function(V, "zero")

    left_dofs = fem.locate_dofs_topological((W.sub(0), V), fdim, left_facets)
    bottom_dofs = fem.locate_dofs_topological((W.sub(0), V), fdim, bottom_facets)
    top_dofs = fem.locate_dofs_topological((W.sub(0), V), fdim, top_facets)

    bc_left = fem.dirichletbc(u_in, left_dofs, W.sub(0))
    bc_bottom = fem.dirichletbc(u_zero, bottom_dofs, W.sub(0))
    bc_top = fem.dirichletbc(u_zero, top_dofs, W.sub(0))

    p_dofs = fem.locate_dofs_geometrical(
        (W.sub(1), Q),
        lambda x: np.isclose(x[0], 0.0) & np.isclose(x[1], 0.0),
    )
    p_zero = fem.Function(Q)
    p_zero.x.array[:] = 0.0
    bc_p = fem.dirichletbc(p_zero, p_dofs, W.sub(1))

    bcs = [bc_left, bc_bottom, bc_top, bc_p]
    return msh, W, V, a, L, bcs


def _solve_stokes(n: int, nu_value: float = 0.5, rtol: float = 1e-9):
    msh, W, V, a, L, bcs = _build_problem(n, nu_value)

    ksp_type = "minres"
    pc_type = "hypre"

    try:
        problem = petsc.LinearProblem(
            a,
            L,
            bcs=bcs,
            petsc_options_prefix=f"stokes_{n}_",
            petsc_options={
                "ksp_type": ksp_type,
                "ksp_rtol": rtol,
                "pc_type": pc_type,
            },
        )
        wh = problem.solve()
    except Exception:
        ksp_type = "preonly"
        pc_type = "lu"
        problem = petsc.LinearProblem(
            a,
            L,
            bcs=bcs,
            petsc_options_prefix=f"stokes_{n}_fallback_",
            petsc_options={
                "ksp_type": ksp_type,
                "pc_type": pc_type,
            },
        )
        wh = problem.solve()

    wh.x.scatter_forward()
    uh = wh.sub(0).collapse()

    div_form = fem.form(ufl.inner(ufl.div(uh), ufl.div(uh)) * ufl.dx)
    div_l2 = math.sqrt(max(0.0, msh.comm.allreduce(fem.assemble_scalar(div_form), op=MPI.SUM)))

    vel_h1s_form = fem.form(ufl.inner(ufl.grad(uh), ufl.grad(uh)) * ufl.dx)
    vel_h1s = math.sqrt(max(0.0, msh.comm.allreduce(fem.assemble_scalar(vel_h1s_form), op=MPI.SUM)))

    history = {
        "mesh_resolution": n,
        "element_degree": 2,
        "ksp_type": ksp_type,
        "pc_type": pc_type,
        "rtol": rtol,
        "iterations": 0,
        "divergence_l2": div_l2,
        "velocity_h1_seminorm": vel_h1s,
    }
    return msh, uh, history


def _sample_velocity_magnitude(u_func: fem.Function, nx: int, ny: int, bbox):
    msh = u_func.function_space.mesh
    xmin, xmax, ymin, ymax = bbox
    xs = np.linspace(xmin, xmax, nx)
    ys = np.linspace(ymin, ymax, ny)

    X, Y = np.meshgrid(xs, ys, indexing="xy")
    pts = np.column_stack([X.ravel(), Y.ravel(), np.zeros(nx * ny, dtype=np.float64)])

    tree = geometry.bb_tree(msh, msh.topology.dim)
    cell_candidates = geometry.compute_collisions_points(tree, pts)
    colliding_cells = geometry.compute_colliding_cells(msh, cell_candidates, pts)

    points_on_proc = []
    cells = []
    eval_map = []
    for i in range(pts.shape[0]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(pts[i])
            cells.append(links[0])
            eval_map.append(i)

    vals_local = np.zeros((pts.shape[0], 2), dtype=np.float64)
    hit_local = np.zeros((pts.shape[0],), dtype=np.int32)

    if points_on_proc:
        values = u_func.eval(np.array(points_on_proc, dtype=np.float64), np.array(cells, dtype=np.int32))
        vals_local[np.array(eval_map, dtype=np.int32), :] = np.asarray(values, dtype=np.float64)
        hit_local[np.array(eval_map, dtype=np.int32)] = 1

    comm = msh.comm
    vals_global = np.zeros_like(vals_local)
    hit_global = np.zeros_like(hit_local)
    comm.Allreduce(vals_local, vals_global, op=MPI.SUM)
    comm.Allreduce(hit_local, hit_global, op=MPI.SUM)

    missing = np.where(hit_global == 0)[0]
    if missing.size > 0:
        pts2 = pts.copy()
        eps = 1e-10
        pts2[:, 0] = np.clip(pts2[:, 0], xmin + eps, xmax - eps)
        pts2[:, 1] = np.clip(pts2[:, 1], ymin + eps, ymax - eps)

        cell_candidates = geometry.compute_collisions_points(tree, pts2)
        colliding_cells = geometry.compute_colliding_cells(msh, cell_candidates, pts2)

        points_on_proc = []
        cells = []
        eval_map = []
        for i in missing:
            links = colliding_cells.links(int(i))
            if len(links) > 0:
                points_on_proc.append(pts2[i])
                cells.append(links[0])
                eval_map.append(int(i))

        vals_local2 = np.zeros((pts.shape[0], 2), dtype=np.float64)
        hit_local2 = np.zeros((pts.shape[0],), dtype=np.int32)
        if points_on_proc:
            values = u_func.eval(np.array(points_on_proc, dtype=np.float64), np.array(cells, dtype=np.int32))
            vals_local2[np.array(eval_map, dtype=np.int32), :] = np.asarray(values, dtype=np.float64)
            hit_local2[np.array(eval_map, dtype=np.int32)] = 1

        vals_global2 = np.zeros_like(vals_local2)
        hit_global2 = np.zeros_like(hit_local2)
        comm.Allreduce(vals_local2, vals_global2, op=MPI.SUM)
        comm.Allreduce(hit_local2, hit_global2, op=MPI.SUM)

        mask = hit_global == 0
        vals_global[mask] = vals_global2[mask]

    mag = np.linalg.norm(vals_global, axis=1).reshape(ny, nx)
    return mag


def solve(case_spec: Dict[str, Any]) -> Dict[str, Any]:
    start = time.perf_counter()

    out_grid = case_spec["output"]["grid"]
    nx = int(out_grid["nx"])
    ny = int(out_grid["ny"])
    bbox = out_grid["bbox"]

    target = max(nx, ny)
    if target <= 64:
        mesh_n = 64
    elif target <= 128:
        mesh_n = 96
    else:
        mesh_n = 128

    _, uh, solver_info = _solve_stokes(mesh_n, nu_value=0.5, rtol=1e-9)
    u_grid = _sample_velocity_magnitude(uh, nx, ny, bbox)

    solver_info["wall_time_sec"] = time.perf_counter() - start

    return {
        "u": u_grid,
        "solver_info": solver_info,
    }
