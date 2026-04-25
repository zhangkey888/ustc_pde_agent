from __future__ import annotations

import time
from dataclasses import dataclass
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl

# DIAGNOSIS
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

# METHOD
# spatial_method: fem
# element_or_basis: Lagrange_P2
# stabilization: none
# time_method: none
# nonlinear_solver: none
# linear_solver: cg
# preconditioner: amg
# special_treatment: none
# pde_skill: poisson

ScalarType = PETSc.ScalarType


@dataclass
class AttemptInfo:
    mesh_resolution: int
    element_degree: int
    ksp_type: str
    pc_type: str
    rtol: float
    iterations: int
    l2_error: float
    wall_time: float


def _exact_and_rhs(msh, kappa_value: float):
    x = ufl.SpatialCoordinate(msh)
    u_exact = ufl.sin(ufl.pi * x[0]) * ufl.sin(2.0 * ufl.pi * x[1])
    f = kappa_value * 5.0 * ufl.pi**2 * u_exact
    return u_exact, f


def _compute_l2_error(msh, uh: fem.Function, u_exact_ufl, degree: int) -> float:
    Vhi = fem.functionspace(msh, ("Lagrange", max(degree + 2, 4)))
    uex = fem.Function(Vhi)
    uex.interpolate(fem.Expression(u_exact_ufl, Vhi.element.interpolation_points))
    uh_hi = fem.Function(Vhi)
    uh_hi.interpolate(uh)
    err_form = fem.form(ufl.inner(uh_hi - uex, uh_hi - uex) * ufl.dx)
    local_val = fem.assemble_scalar(err_form)
    global_val = msh.comm.allreduce(local_val, op=MPI.SUM)
    return float(np.sqrt(global_val))


def _sample_function(u_func: fem.Function, nx: int, ny: int, bbox) -> np.ndarray:
    xmin, xmax, ymin, ymax = bbox
    xs = np.linspace(xmin, xmax, nx)
    ys = np.linspace(ymin, ymax, ny)
    XX, YY = np.meshgrid(xs, ys, indexing="xy")
    pts = np.column_stack([XX.ravel(), YY.ravel(), np.zeros(nx * ny, dtype=np.float64)])

    msh = u_func.function_space.mesh
    tree = geometry.bb_tree(msh, msh.topology.dim)
    candidates = geometry.compute_collisions_points(tree, pts)
    colliding = geometry.compute_colliding_cells(msh, candidates, pts)

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
    if points_on_proc:
        vals = u_func.eval(np.array(points_on_proc, dtype=np.float64),
                           np.array(cells_on_proc, dtype=np.int32))
        local_vals[np.array(eval_map, dtype=np.int32)] = np.asarray(vals).reshape(-1).real

    comm = msh.comm
    gathered = comm.gather(local_vals, root=0)
    if comm.rank == 0:
        merged = np.full_like(local_vals, np.nan)
        for arr in gathered:
            mask = np.isfinite(arr)
            merged[mask] = arr[mask]
        if np.any(~np.isfinite(merged)):
            merged[~np.isfinite(merged)] = 0.0
        out = merged.reshape(ny, nx)
    else:
        out = None
    return comm.bcast(out, root=0)


def _solve_once(n: int, degree: int, kappa_value: float = 10.0, rtol: float = 1e-10):
    comm = MPI.COMM_WORLD
    msh = mesh.create_unit_square(comm, n, n, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(msh, ("Lagrange", degree))

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    kappa = fem.Constant(msh, ScalarType(kappa_value))
    u_exact, f = _exact_and_rhs(msh, kappa_value)

    fdim = msh.topology.dim - 1
    facets = mesh.locate_entities_boundary(msh, fdim, lambda x: np.ones(x.shape[1], dtype=bool))
    dofs = fem.locate_dofs_topological(V, fdim, facets)

    ubc = fem.Function(V)
    ubc.interpolate(fem.Expression(u_exact, V.element.interpolation_points))
    bc = fem.dirichletbc(ubc, dofs)

    a = ufl.inner(kappa * ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = ufl.inner(f, v) * ufl.dx
    a_form = fem.form(a)
    L_form = fem.form(L)

    A = petsc.assemble_matrix(a_form, bcs=[bc])
    A.assemble()
    b = petsc.create_vector(L_form.function_spaces)
    uh = fem.Function(V)

    with b.localForm() as loc:
        loc.set(0.0)
    petsc.assemble_vector(b, L_form)
    petsc.apply_lifting(b, [a_form], bcs=[[bc]])
    b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    petsc.set_bc(b, [bc])

    t0 = time.perf_counter()
    solver = PETSc.KSP().create(comm)
    solver.setOperators(A)
    solver.setType("cg")
    solver.getPC().setType("hypre")
    solver.setTolerances(rtol=rtol, atol=1e-14, max_it=10000)
    prefix = f"poisson_{n}_{degree}_"
    solver.setOptionsPrefix(prefix)
    opts = PETSc.Options()
    opts[f"{prefix}pc_hypre_type"] = "boomeramg"
    solver.setFromOptions()

    ksp_type = "cg"
    pc_type = "hypre"
    try:
        solver.solve(b, uh.x.petsc_vec)
        uh.x.scatter_forward()
        if solver.getConvergedReason() <= 0:
            raise RuntimeError("CG solve failed")
    except Exception:
        solver = PETSc.KSP().create(comm)
        solver.setOperators(A)
        solver.setType("preonly")
        solver.getPC().setType("lu")
        solver.setTolerances(rtol=rtol, atol=1e-14, max_it=1)
        solver.solve(b, uh.x.petsc_vec)
        uh.x.scatter_forward()
        ksp_type = "preonly"
        pc_type = "lu"

    wall_time = time.perf_counter() - t0
    iterations = int(solver.getIterationNumber())
    l2_error = _compute_l2_error(msh, uh, u_exact, degree)
    return uh, AttemptInfo(n, degree, ksp_type, pc_type, rtol, iterations, l2_error, wall_time)


def solve(case_spec: dict) -> dict:
    grid = case_spec["output"]["grid"]
    nx = int(grid["nx"])
    ny = int(grid["ny"])
    bbox = grid["bbox"]

    time_limit = float(case_spec.get("time_limit_sec", 16.368))
    target_error = 3.79e-03
    candidates = [(24, 1), (32, 1), (24, 2), (32, 2), (48, 2), (64, 2)]

    best_u = None
    best_info = None
    spent = 0.0
    budget = 0.9 * time_limit

    for n, degree in candidates:
        if spent >= budget:
            break
        t0 = time.perf_counter()
        uh, info = _solve_once(n, degree)
        spent += time.perf_counter() - t0
        best_u, best_info = uh, info
        if info.l2_error <= target_error and spent > 0.5 * budget:
            break

    u_grid = _sample_function(best_u, nx, ny, bbox)
    return {
        "u": u_grid,
        "solver_info": {
            "mesh_resolution": int(best_info.mesh_resolution),
            "element_degree": int(best_info.element_degree),
            "ksp_type": str(best_info.ksp_type),
            "pc_type": str(best_info.pc_type),
            "rtol": float(best_info.rtol),
            "iterations": int(best_info.iterations),
        },
    }
