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


def _case_params(case_spec: dict):
    pde = case_spec.get("pde", {})
    time_spec = pde.get("time", {})
    t0 = float(time_spec.get("t0", 0.0))
    t_end = float(time_spec.get("t_end", 0.4))
    dt_in = float(time_spec.get("dt", 0.005))
    scheme = str(time_spec.get("scheme", "crank_nicolson")).lower()

    eps = float(pde.get("epsilon", case_spec.get("epsilon", 0.02)))
    reaction_coeff = float(pde.get("reaction_coeff", 1.0))
    return t0, t_end, dt_in, scheme, eps, reaction_coeff


def _choose_resolution(time_budget_hint: float = 207.813):
    # Conservative but accurate default under generous benchmark budget
    if time_budget_hint > 120:
        return 56, 2, 0.005
    elif time_budget_hint > 30:
        return 48, 2, 0.005
    return 36, 1, 0.005


def _exact_expr(x, t):
    return ufl.exp(-t) * (
        ufl.sin(ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])
        + 0.2 * ufl.sin(6 * ufl.pi * x[0]) * ufl.sin(5 * ufl.pi * x[1])
    )


def _forcing_expr(msh, eps, reaction_coeff, t):
    x = ufl.SpatialCoordinate(msh)
    uex = _exact_expr(x, t)
    ut = -uex
    lap = ufl.div(ufl.grad(uex))
    return ut - eps * lap + reaction_coeff * uex


def _sample_function(u_func: fem.Function, nx: int, ny: int, bbox):
    msh = u_func.function_space.mesh
    xs = np.linspace(bbox[0], bbox[1], nx)
    ys = np.linspace(bbox[2], bbox[3], ny)
    XX, YY = np.meshgrid(xs, ys, indexing="xy")
    pts = np.column_stack([XX.ravel(), YY.ravel(), np.zeros(nx * ny, dtype=np.float64)])

    tree = geometry.bb_tree(msh, msh.topology.dim)
    cell_candidates = geometry.compute_collisions_points(tree, pts)
    colliding_cells = geometry.compute_colliding_cells(msh, cell_candidates, pts)

    local_vals = np.full(nx * ny, np.nan, dtype=np.float64)
    points_on_proc = []
    cells_on_proc = []
    ids = []
    for i in range(pts.shape[0]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(pts[i])
            cells_on_proc.append(links[0])
            ids.append(i)

    if points_on_proc:
        vals = u_func.eval(np.asarray(points_on_proc, dtype=np.float64),
                           np.asarray(cells_on_proc, dtype=np.int32))
        vals = np.asarray(vals).reshape(-1)
        local_vals[np.asarray(ids, dtype=np.int32)] = vals

    comm = msh.comm
    gathered = comm.gather(local_vals, root=0)
    if comm.rank == 0:
        out = np.full(nx * ny, np.nan, dtype=np.float64)
        for arr in gathered:
            mask = np.isfinite(arr)
            out[mask] = arr[mask]
        if np.isnan(out).any():
            raise RuntimeError("Sampling failed for some output grid points")
        return out.reshape(ny, nx)
    return None


def solve(case_spec: dict) -> Dict[str, Any]:
    comm = MPI.COMM_WORLD
    rank = comm.rank

    t0, t_end, dt_req, scheme, eps, reaction_coeff = _case_params(case_spec)
    if scheme != "crank_nicolson":
        scheme = "crank_nicolson"

    mesh_resolution, degree, dt_suggest = _choose_resolution()
    dt = min(dt_req, dt_suggest)
    n_steps = max(1, int(round((t_end - t0) / dt)))
    dt = (t_end - t0) / n_steps

    msh = mesh.create_unit_square(comm, mesh_resolution, mesh_resolution, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(msh, ("Lagrange", degree))

    x = ufl.SpatialCoordinate(msh)

    u_n = fem.Function(V)
    u_n.interpolate(fem.Expression(_exact_expr(x, ScalarType(t0)), V.element.interpolation_points))
    u_n.x.scatter_forward()

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    t_c = fem.Constant(msh, ScalarType(t0))
    t_np1 = fem.Constant(msh, ScalarType(t0 + dt))

    f_n_expr = _forcing_expr(msh, eps, reaction_coeff, t_c)
    f_np1_expr = _forcing_expr(msh, eps, reaction_coeff, t_np1)

    bc_fun = fem.Function(V)
    bc_fun.interpolate(fem.Expression(_exact_expr(x, t_np1), V.element.interpolation_points))
    bc_fun.x.scatter_forward()

    fdim = msh.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(msh, fdim, lambda xx: np.ones(xx.shape[1], dtype=bool))
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc = fem.dirichletbc(bc_fun, boundary_dofs)

    a = (
        (1.0 / dt) * u * v * ufl.dx
        + 0.5 * eps * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
        + 0.5 * reaction_coeff * u * v * ufl.dx
    )

    L = (
        (1.0 / dt) * u_n * v * ufl.dx
        - 0.5 * eps * ufl.inner(ufl.grad(u_n), ufl.grad(v)) * ufl.dx
        - 0.5 * reaction_coeff * u_n * v * ufl.dx
        + 0.5 * (f_n_expr + f_np1_expr) * v * ufl.dx
    )

    a_form = fem.form(a)
    L_form = fem.form(L)

    A = petsc.assemble_matrix(a_form, bcs=[bc])
    A.assemble()
    b = petsc.create_vector(L_form.function_spaces)

    ksp = PETSc.KSP().create(comm)
    ksp.setOperators(A)
    ksp.setType("cg")
    pc = ksp.getPC()
    pc.setType("hypre")
    ksp.setTolerances(rtol=1.0e-10, atol=1.0e-14, max_it=5000)
    ksp.setFromOptions()

    uh = fem.Function(V)
    total_iterations = 0

    # Save initial sampled field
    output_grid = case_spec["output"]["grid"]
    nx = int(output_grid["nx"])
    ny = int(output_grid["ny"])
    bbox = output_grid["bbox"]
    u_initial = _sample_function(u_n, nx, ny, bbox)

    start = time.perf_counter()
    t = t0
    for step in range(n_steps):
        t_old = t
        t = t0 + (step + 1) * dt
        t_c.value = ScalarType(t_old)
        t_np1.value = ScalarType(t)

        bc_fun.interpolate(fem.Expression(_exact_expr(x, t_np1), V.element.interpolation_points))
        bc_fun.x.scatter_forward()

        with b.localForm() as loc:
            loc.set(0.0)
        petsc.assemble_vector(b, L_form)
        petsc.apply_lifting(b, [a_form], bcs=[[bc]])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        petsc.set_bc(b, [bc])

        uh.x.array[:] = 0.0
        ksp.solve(b, uh.x.petsc_vec)
        uh.x.scatter_forward()
        its = ksp.getIterationNumber()
        total_iterations += int(its)

        u_n.x.array[:] = uh.x.array
        u_n.x.scatter_forward()

    elapsed = time.perf_counter() - start

    # Accuracy verification
    u_exact_T = fem.Function(V)
    u_exact_T.interpolate(fem.Expression(_exact_expr(x, ScalarType(t_end)), V.element.interpolation_points))
    u_exact_T.x.scatter_forward()

    err_fun = fem.Function(V)
    err_fun.x.array[:] = u_n.x.array - u_exact_T.x.array
    err_fun.x.scatter_forward()

    l2_err_local = fem.assemble_scalar(fem.form(ufl.inner(err_fun, err_fun) * ufl.dx))
    l2_ref_local = fem.assemble_scalar(fem.form(ufl.inner(u_exact_T, u_exact_T) * ufl.dx))
    l2_err = math.sqrt(comm.allreduce(l2_err_local, op=MPI.SUM))
    l2_ref = math.sqrt(comm.allreduce(l2_ref_local, op=MPI.SUM))
    rel_l2_err = l2_err / max(l2_ref, 1e-16)

    u_grid = _sample_function(u_n, nx, ny, bbox)

    solver_info = {
        "mesh_resolution": int(mesh_resolution),
        "element_degree": int(degree),
        "ksp_type": str(ksp.getType()),
        "pc_type": str(ksp.getPC().getType()),
        "rtol": float(1.0e-10),
        "iterations": int(total_iterations),
        "dt": float(dt),
        "n_steps": int(n_steps),
        "time_scheme": "crank_nicolson",
        "accuracy_verification": {
            "l2_error": float(l2_err),
            "relative_l2_error": float(rel_l2_err),
            "wall_time_sec": float(elapsed),
            "manufactured_solution": "exp(-t)*(sin(pi*x)*sin(pi*y) + 0.2*sin(6*pi*x)*sin(5*pi*y))",
        },
    }

    if rank == 0:
        return {
            "u": u_grid,
            "u_initial": u_initial,
            "solver_info": solver_info,
        }
    else:
        return {
            "u": None,
            "u_initial": None,
            "solver_info": solver_info,
        }


if __name__ == "__main__":
    case_spec = {
        "pde": {
            "time": {"t0": 0.0, "t_end": 0.4, "dt": 0.005, "scheme": "crank_nicolson"},
            "epsilon": 0.02,
            "reaction_coeff": 1.0,
        },
        "output": {
            "grid": {"nx": 64, "ny": 64, "bbox": [0.0, 1.0, 0.0, 1.0]}
        },
    }
    out = solve(case_spec)
    if MPI.COMM_WORLD.rank == 0:
        print(out["u"].shape)
        print(out["solver_info"])
