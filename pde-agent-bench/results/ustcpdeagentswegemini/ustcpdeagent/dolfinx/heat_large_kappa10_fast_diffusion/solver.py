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

import math
import time
from typing import Tuple

import numpy as np
import ufl
from mpi4py import MPI
from petsc4py import PETSc

from dolfinx import fem, geometry, mesh
from dolfinx.fem import petsc

ScalarType = PETSc.ScalarType


def _manufactured_exact(points: np.ndarray, t: float) -> np.ndarray:
    return np.exp(-t) * np.sin(np.pi * points[0]) * np.sin(np.pi * points[1])


def _sample_function_on_grid(domain, uh: fem.Function, grid_spec: dict) -> np.ndarray:
    nx = int(grid_spec["nx"])
    ny = int(grid_spec["ny"])
    xmin, xmax, ymin, ymax = map(float, grid_spec["bbox"])

    xs = np.linspace(xmin, xmax, nx)
    ys = np.linspace(ymin, ymax, ny)
    XX, YY = np.meshgrid(xs, ys, indexing="xy")
    pts = np.column_stack([XX.ravel(), YY.ravel(), np.zeros(nx * ny, dtype=np.float64)])

    tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(tree, pts)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, pts)

    local_vals = np.full(pts.shape[0], np.nan, dtype=np.float64)
    points_on_proc = []
    cells_on_proc = []
    idx_map = []
    for i in range(pts.shape[0]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(pts[i])
            cells_on_proc.append(links[0])
            idx_map.append(i)

    if points_on_proc:
        vals = uh.eval(np.asarray(points_on_proc, dtype=np.float64), np.asarray(cells_on_proc, dtype=np.int32))
        local_vals[np.asarray(idx_map, dtype=np.int32)] = np.asarray(vals).reshape(-1)

    gathered = domain.comm.gather(local_vals, root=0)
    if domain.comm.rank == 0:
        merged = np.full(pts.shape[0], np.nan, dtype=np.float64)
        for arr in gathered:
            mask = np.isnan(merged) & ~np.isnan(arr)
            merged[mask] = arr[mask]
        if np.isnan(merged).any():
            merged[np.isnan(merged)] = 0.0
        return merged.reshape(ny, nx)
    return np.empty((ny, nx), dtype=np.float64)


def _compute_errors(domain, uh: fem.Function, t_value: float) -> Tuple[float, float]:
    x = ufl.SpatialCoordinate(domain)
    u_exact = ufl.exp(-t_value) * ufl.sin(ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])

    V = uh.function_space
    u_ex = fem.Function(V)
    u_ex.interpolate(fem.Expression(u_exact, V.element.interpolation_points))

    e = fem.Function(V)
    e.x.array[:] = uh.x.array - u_ex.x.array
    e.x.scatter_forward()

    l2_sq = fem.assemble_scalar(fem.form(ufl.inner(e, e) * ufl.dx))
    h1_sq = fem.assemble_scalar(fem.form((ufl.inner(e, e) + ufl.inner(ufl.grad(e), ufl.grad(e))) * ufl.dx))
    l2 = math.sqrt(domain.comm.allreduce(l2_sq, op=MPI.SUM))
    h1 = math.sqrt(domain.comm.allreduce(h1_sq, op=MPI.SUM))
    return l2, h1


def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD
    t_wall0 = time.perf_counter()

    coeffs = case_spec.get("coefficients", {})
    pde = case_spec.get("pde", {})
    time_spec = case_spec.get("time", {})
    output = case_spec.get("output", {})
    grid_spec = output.get("grid", {"nx": 64, "ny": 64, "bbox": [0.0, 1.0, 0.0, 1.0]})

    kappa = float(coeffs.get("kappa", 10.0))
    t0 = float(time_spec.get("t0", pde.get("t0", 0.0)))
    t_end = float(time_spec.get("t_end", pde.get("t_end", 0.05)))
    dt_suggested = float(time_spec.get("dt", pde.get("dt", 0.005)))

    dt = min(dt_suggested, 0.000625)
    n_steps = max(1, int(round((t_end - t0) / dt)))
    dt = (t_end - t0) / n_steps

    mesh_resolution = 128
    element_degree = 2

    domain = mesh.create_unit_square(comm, mesh_resolution, mesh_resolution, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", element_degree))

    u_n = fem.Function(V)
    u_n.interpolate(lambda X: _manufactured_exact(X, t0))
    u_n.x.scatter_forward()

    u_initial = _sample_function_on_grid(domain, u_n, grid_spec)

    fdim = domain.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(domain, fdim, lambda X: np.ones(X.shape[1], dtype=bool))
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)

    bc_fun = fem.Function(V)
    bc_fun.interpolate(lambda X: _manufactured_exact(X, t0 + dt))
    bc_fun.x.scatter_forward()
    bc = fem.dirichletbc(bc_fun, boundary_dofs)

    x = ufl.SpatialCoordinate(domain)
    t_c = fem.Constant(domain, ScalarType(t0 + dt))
    dt_c = fem.Constant(domain, ScalarType(dt))
    kappa_c = fem.Constant(domain, ScalarType(kappa))

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    f_expr = (
        (-1.0 + 2.0 * kappa * np.pi**2)
        * ufl.exp(-t_c)
        * ufl.sin(ufl.pi * x[0])
        * ufl.sin(ufl.pi * x[1])
    )

    a = (u * v + dt_c * kappa_c * ufl.inner(ufl.grad(u), ufl.grad(v))) * ufl.dx
    L = (u_n * v + dt_c * f_expr * v) * ufl.dx

    a_form = fem.form(a)
    L_form = fem.form(L)

    A = petsc.assemble_matrix(a_form, bcs=[bc])
    A.assemble()
    b = petsc.create_vector(L_form.function_spaces)

    solver = PETSc.KSP().create(comm)
    solver.setOperators(A)
    solver.setType("cg")
    solver.getPC().setType("hypre")
    solver.setTolerances(rtol=1e-10, atol=1e-14, max_it=5000)
    solver.setFromOptions()

    uh = fem.Function(V)
    total_iterations = 0
    current_t = t0

    for _ in range(n_steps):
        current_t += dt
        t_c.value = ScalarType(current_t)
        bc_fun.interpolate(lambda X, tt=current_t: _manufactured_exact(X, tt))
        bc_fun.x.scatter_forward()

        A.zeroEntries()
        petsc.assemble_matrix(A, a_form, bcs=[bc])
        A.assemble()

        with b.localForm() as loc:
            loc.set(0.0)
        petsc.assemble_vector(b, L_form)
        petsc.apply_lifting(b, [a_form], bcs=[[bc]])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        petsc.set_bc(b, [bc])

        try:
            solver.solve(b, uh.x.petsc_vec)
            if solver.getConvergedReason() <= 0:
                raise RuntimeError("CG failed")
        except Exception:
            solver.setType("preonly")
            solver.getPC().setType("lu")
            solver.setOperators(A)
            solver.solve(b, uh.x.petsc_vec)

        uh.x.scatter_forward()
        total_iterations += int(max(solver.getIterationNumber(), 1))
        u_n.x.array[:] = uh.x.array
        u_n.x.scatter_forward()

    l2_error, h1_error = _compute_errors(domain, uh, current_t)
    u_grid = _sample_function_on_grid(domain, uh, grid_spec)
    wall = time.perf_counter() - t_wall0

    solver_info = {
        "mesh_resolution": mesh_resolution,
        "element_degree": element_degree,
        "ksp_type": solver.getType(),
        "pc_type": solver.getPC().getType(),
        "rtol": 1.0e-10,
        "iterations": int(total_iterations),
        "dt": float(dt),
        "n_steps": int(n_steps),
        "time_scheme": "backward_euler",
        "l2_error": float(l2_error),
        "h1_error": float(h1_error),
        "wall_time_sec": float(wall),
    }

    if comm.rank == 0:
        return {"u": u_grid, "u_initial": u_initial, "solver_info": solver_info}
    return {"u": None, "u_initial": None, "solver_info": solver_info}


if __name__ == "__main__":
    case_spec = {
        "coefficients": {"kappa": 10.0},
        "time": {"t0": 0.0, "t_end": 0.05, "dt": 0.005},
        "output": {"grid": {"nx": 41, "ny": 39, "bbox": [0.0, 1.0, 0.0, 1.0]}},
        "pde": {"time": True},
    }
    result = solve(case_spec)
    if MPI.COMM_WORLD.rank == 0:
        print(result["u"].shape)
        print(result["solver_info"])
