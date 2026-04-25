# ```DIAGNOSIS
# equation_type: poisson
# spatial_dim: 2
# domain_geometry: rectangle
# unknowns: scalar
# coupling: none
# linearity: linear
# time_dependence: steady
# stiffness: N/A
# dominant_physics: diffusion
# peclet_or_reynolds: N/A
# solution_regularity: smooth
# bc_type: all_dirichlet
# special_notes: manufactured_solution
# ```
#
# ```METHOD
# spatial_method: fem
# element_or_basis: Lagrange_P2
# stabilization: none
# time_method: none
# nonlinear_solver: none
# linear_solver: cg
# preconditioner: hypre
# special_treatment: none
# pde_skill: poisson
# ```

from __future__ import annotations

import time
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl

ScalarType = PETSc.ScalarType


def _exact_u_expr(x):
    return ufl.sin(ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1]) + 0.3 * ufl.sin(6.0 * ufl.pi * x[0]) * ufl.sin(
        6.0 * ufl.pi * x[1]
    )


def _rhs_expr(x, kappa=1.0):
    return ScalarType(kappa) * (
        2.0 * ufl.pi**2 * ufl.sin(ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])
        + 0.3 * 72.0 * ufl.pi**2 * ufl.sin(6.0 * ufl.pi * x[0]) * ufl.sin(6.0 * ufl.pi * x[1])
    )


def _all_boundary(x):
    return np.ones(x.shape[1], dtype=bool)


def _probe_points(u_func: fem.Function, points_xyz: np.ndarray) -> np.ndarray:
    msh = u_func.function_space.mesh
    tree = geometry.bb_tree(msh, msh.topology.dim)
    candidates = geometry.compute_collisions_points(tree, points_xyz)
    colliding = geometry.compute_colliding_cells(msh, candidates, points_xyz)

    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    for i in range(points_xyz.shape[0]):
        links = colliding.links(i)
        if len(links) > 0:
            points_on_proc.append(points_xyz[i])
            cells_on_proc.append(links[0])
            eval_map.append(i)

    values = np.full(points_xyz.shape[0], np.nan, dtype=np.float64)
    if points_on_proc:
        vals = u_func.eval(np.array(points_on_proc, dtype=np.float64), np.array(cells_on_proc, dtype=np.int32))
        vals = np.real(np.asarray(vals)).reshape(len(points_on_proc), -1)
        values[np.array(eval_map, dtype=np.int32)] = vals[:, 0]
    return values


def _sample_on_uniform_grid(u_func: fem.Function, grid_spec: dict) -> np.ndarray:
    nx = int(grid_spec["nx"])
    ny = int(grid_spec["ny"])
    xmin, xmax, ymin, ymax = [float(v) for v in grid_spec["bbox"]]

    xs = np.linspace(xmin, xmax, nx, dtype=np.float64)
    ys = np.linspace(ymin, ymax, ny, dtype=np.float64)
    xx, yy = np.meshgrid(xs, ys, indexing="xy")
    pts = np.column_stack([xx.ravel(), yy.ravel(), np.zeros(nx * ny, dtype=np.float64)])
    vals_local = _probe_points(u_func, pts)

    comm = u_func.function_space.mesh.comm
    gathered = comm.allgather(vals_local)
    vals = gathered[0].copy()
    for arr in gathered[1:]:
        mask = np.isnan(vals) & ~np.isnan(arr)
        vals[mask] = arr[mask]

    if np.isnan(vals).any():
        raise RuntimeError("Failed to evaluate solution at some output grid points.")
    return vals.reshape(ny, nx)


def solve(case_spec: dict) -> dict:
    t0 = time.perf_counter()
    comm = MPI.COMM_WORLD

    try:
        kappa = float(case_spec.get("pde", {}).get("coefficients", {}).get("kappa", 1.0))
    except Exception:
        kappa = 1.0

    mesh_resolution = 72
    element_degree = 3
    rtol = 1e-10

    msh = mesh.create_unit_square(comm, mesh_resolution, mesh_resolution, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(msh, ("Lagrange", element_degree))

    x = ufl.SpatialCoordinate(msh)
    u_exact = _exact_u_expr(x)
    f_expr = _rhs_expr(x, kappa=kappa)

    uD = fem.Function(V)
    uD.interpolate(fem.Expression(u_exact, V.element.interpolation_points))

    fdim = msh.topology.dim - 1
    facets = mesh.locate_entities_boundary(msh, fdim, _all_boundary)
    dofs = fem.locate_dofs_topological(V, fdim, facets)
    bc = fem.dirichletbc(uD, dofs)

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    a = ScalarType(kappa) * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = ufl.inner(f_expr, v) * ufl.dx

    a_form = fem.form(a)
    L_form = fem.form(L)

    A = petsc.assemble_matrix(a_form, bcs=[bc])
    A.assemble()
    b = petsc.create_vector(L_form.function_spaces)

    with b.localForm() as loc:
        loc.set(0)
    petsc.assemble_vector(b, L_form)
    petsc.apply_lifting(b, [a_form], bcs=[[bc]])
    b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    petsc.set_bc(b, [bc])

    solver = PETSc.KSP().create(comm)
    solver.setOperators(A)
    solver.setType("cg")
    solver.getPC().setType("hypre")
    solver.setTolerances(rtol=rtol)

    uh = fem.Function(V)
    try:
        solver.solve(b, uh.x.petsc_vec)
        uh.x.scatter_forward()
        if solver.getConvergedReason() <= 0:
            raise RuntimeError("cg+hypre failed")
        ksp_type = solver.getType()
        pc_type = solver.getPC().getType()
        iterations = int(solver.getIterationNumber())
    except Exception:
        solver = PETSc.KSP().create(comm)
        solver.setOperators(A)
        solver.setType("preonly")
        solver.getPC().setType("lu")
        solver.solve(b, uh.x.petsc_vec)
        uh.x.scatter_forward()
        ksp_type = solver.getType()
        pc_type = solver.getPC().getType()
        iterations = 1

    grid_spec = case_spec["output"]["grid"]
    u_grid = _sample_on_uniform_grid(uh, grid_spec)

    nx = int(grid_spec["nx"])
    ny = int(grid_spec["ny"])
    xmin, xmax, ymin, ymax = [float(v) for v in grid_spec["bbox"]]
    xs = np.linspace(xmin, xmax, nx, dtype=np.float64)
    ys = np.linspace(ymin, ymax, ny, dtype=np.float64)
    xx, yy = np.meshgrid(xs, ys, indexing="xy")
    u_exact_grid = np.sin(np.pi * xx) * np.sin(np.pi * yy) + 0.3 * np.sin(6.0 * np.pi * xx) * np.sin(
        6.0 * np.pi * yy
    )

    solver_info = {
        "mesh_resolution": mesh_resolution,
        "element_degree": element_degree,
        "ksp_type": str(ksp_type),
        "pc_type": str(pc_type),
        "rtol": float(rtol),
        "iterations": iterations,
        "verification_l2_grid": float(np.sqrt(np.mean((u_grid - u_exact_grid) ** 2))),
        "verification_linf_grid": float(np.max(np.abs(u_grid - u_exact_grid))),
        "wall_time_sec": float(time.perf_counter() - t0),
    }

    return {"u": np.asarray(u_grid, dtype=np.float64).reshape(ny, nx), "solver_info": solver_info}


if __name__ == "__main__":
    case_spec = {
        "case_id": "poisson_multifrequency_u",
        "pde": {"time": None, "coefficients": {"kappa": 1.0}},
        "output": {"grid": {"nx": 64, "ny": 64, "bbox": [0.0, 1.0, 0.0, 1.0]}},
    }
    out = solve(case_spec)
    if MPI.COMM_WORLD.rank == 0:
        print(out["u"].shape)
        print(out["solver_info"])
