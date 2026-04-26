import math
import time
from typing import Dict, Any

import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl

ScalarType = PETSc.ScalarType

"""
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
special_notes: manufactured_solution
"""

"""
METHOD
spatial_method: fem
element_or_basis: Lagrange_P3
stabilization: none
time_method: backward_euler
nonlinear_solver: none
linear_solver: cg
preconditioner: hypre
special_treatment: none
pde_skill: heat
"""


def _u_exact_numpy(x, t):
    return np.exp(-t) * np.sin(3.0 * np.pi * x[0]) * np.sin(3.0 * np.pi * x[1])


def _u_exact_ufl(x, t):
    return ufl.exp(-t) * ufl.sin(3.0 * ufl.pi * x[0]) * ufl.sin(3.0 * ufl.pi * x[1])


def _f_exact_ufl(x, t, kappa):
    uex = _u_exact_ufl(x, t)
    return -uex + 18.0 * ufl.pi**2 * kappa * uex


def _probe_function(u_func: fem.Function, points: np.ndarray) -> np.ndarray:
    msh = u_func.function_space.mesh
    bb_tree = geometry.bb_tree(msh, msh.topology.dim)
    pts = np.asarray(points, dtype=np.float64)
    cell_candidates = geometry.compute_collisions_points(bb_tree, pts)
    colliding_cells = geometry.compute_colliding_cells(msh, cell_candidates, pts)

    values = np.full((pts.shape[0],), np.nan, dtype=np.float64)
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
        vals = u_func.eval(np.array(points_on_proc, dtype=np.float64),
                           np.array(cells_on_proc, dtype=np.int32))
        values[np.array(eval_map, dtype=np.int32)] = np.asarray(vals).reshape(-1)
    return values


def _sample_on_grid(u_func: fem.Function, grid_spec: Dict[str, Any]) -> np.ndarray:
    nx = int(grid_spec["nx"])
    ny = int(grid_spec["ny"])
    xmin, xmax, ymin, ymax = map(float, grid_spec["bbox"])
    xs = np.linspace(xmin, xmax, nx)
    ys = np.linspace(ymin, ymax, ny)
    xx, yy = np.meshgrid(xs, ys, indexing="xy")
    pts = np.column_stack([xx.ravel(), yy.ravel(), np.zeros(nx * ny, dtype=np.float64)])
    vals = _probe_function(u_func, pts)
    return vals.reshape(ny, nx)


def _solve_once(mesh_resolution: int, degree: int, dt: float, t0: float, t_end: float, kappa_value: float):
    comm = MPI.COMM_WORLD
    msh = mesh.create_unit_square(comm, mesh_resolution, mesh_resolution, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(msh, ("Lagrange", degree))
    x = ufl.SpatialCoordinate(msh)

    u_n = fem.Function(V)
    u_n.interpolate(lambda X: _u_exact_numpy(X, t0))
    u_n.x.scatter_forward()

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    fdim = msh.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(msh, fdim, lambda X: np.ones(X.shape[1], dtype=bool))
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)

    bc_func = fem.Function(V)
    bc_func.interpolate(lambda X: _u_exact_numpy(X, t0))
    bc = fem.dirichletbc(bc_func, boundary_dofs)

    kappa = fem.Constant(msh, ScalarType(kappa_value))
    dt_c = fem.Constant(msh, ScalarType(dt))
    f_func = fem.Function(V)

    a = (u * v + dt_c * kappa * ufl.inner(ufl.grad(u), ufl.grad(v))) * ufl.dx
    L = (u_n * v + dt_c * f_func * v) * ufl.dx

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
    solver.setTolerances(rtol=1e-10, atol=1e-14, max_it=10000)
    solver.setFromOptions()

    total_iters = 0
    n_steps = max(1, int(round((t_end - t0) / dt)))

    for n in range(1, n_steps + 1):
        t = t0 + n * dt

        bc_func.interpolate(lambda X, tt=t: _u_exact_numpy(X, tt))
        bc_func.x.scatter_forward()

        f_expr = fem.Expression(_f_exact_ufl(x, ScalarType(t), kappa), V.element.interpolation_points)
        f_func.interpolate(f_expr)
        f_func.x.scatter_forward()

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
        try:
            total_iters += int(solver.getIterationNumber())
        except Exception:
            pass

        u_n.x.array[:] = uh.x.array
        u_n.x.scatter_forward()

    u_ex = fem.Function(V)
    u_ex.interpolate(lambda X: _u_exact_numpy(X, t_end))
    u_ex.x.scatter_forward()

    err_local = fem.assemble_scalar(fem.form((uh - u_ex) ** 2 * ufl.dx))
    err_l2 = math.sqrt(comm.allreduce(err_local, op=MPI.SUM))

    solver_info = {
        "mesh_resolution": int(mesh_resolution),
        "element_degree": int(degree),
        "ksp_type": str(solver.getType()),
        "pc_type": str(solver.getPC().getType()),
        "rtol": 1e-10,
        "iterations": int(total_iters),
        "dt": float(dt),
        "n_steps": int(n_steps),
        "time_scheme": "backward_euler",
        "manufactured_L2_error": float(err_l2),
    }
    return uh, solver_info


def solve(case_spec: dict) -> dict:
    t0 = float(case_spec.get("pde", {}).get("time", {}).get("t0", 0.0))
    t_end = float(case_spec.get("pde", {}).get("time", {}).get("t_end", 0.08))
    dt_suggested = float(case_spec.get("pde", {}).get("time", {}).get("dt", 0.008))
    grid_spec = case_spec["output"]["grid"]

    degree = 3
    mesh_resolution = 40
    dt = min(dt_suggested, 0.004)
    n_steps = max(1, int(math.ceil((t_end - t0) / dt)))
    dt = (t_end - t0) / n_steps

    start = time.perf_counter()
    uh, info = _solve_once(mesh_resolution, degree, dt, t0, t_end, 1.0)
    elapsed = time.perf_counter() - start

    if elapsed < 3.0:
        mesh_resolution2 = 52
        dt2 = min(dt, 0.0032)
        n_steps2 = max(1, int(math.ceil((t_end - t0) / dt2)))
        dt2 = (t_end - t0) / n_steps2
        uh2, info2 = _solve_once(mesh_resolution2, degree, dt2, t0, t_end, 1.0)
        if info2["manufactured_L2_error"] <= info["manufactured_L2_error"]:
            uh, info = uh2, info2

    u_grid = _sample_on_grid(uh, grid_spec)

    V = uh.function_space
    u0 = fem.Function(V)
    u0.interpolate(lambda X: _u_exact_numpy(X, t0))
    u0.x.scatter_forward()
    u_initial = _sample_on_grid(u0, grid_spec)

    return {
        "u": u_grid,
        "solver_info": info,
        "u_initial": u_initial,
    }
