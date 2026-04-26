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


def _ufact(x, t):
    return ufl.exp(-t) * ufl.sin(ufl.pi * x[0]) * ufl.sin(2.0 * ufl.pi * x[1])


def _kappa(x):
    return 1.0 + 30.0 * ufl.exp(-150.0 * ((x[0] - 0.35) ** 2 + (x[1] - 0.65) ** 2))


def _source(x, t):
    ue = _ufact(x, t)
    kap = _kappa(x)
    return -ue - ufl.div(kap * ufl.grad(ue))


def _sample_on_grid(msh, uh, grid_spec):
    nx = int(grid_spec["nx"])
    ny = int(grid_spec["ny"])
    xmin, xmax, ymin, ymax = map(float, grid_spec["bbox"])

    xs = np.linspace(xmin, xmax, nx)
    ys = np.linspace(ymin, ymax, ny)
    XX, YY = np.meshgrid(xs, ys, indexing="xy")
    pts = np.column_stack([XX.ravel(), YY.ravel(), np.zeros(nx * ny, dtype=np.float64)])

    tree = geometry.bb_tree(msh, msh.topology.dim)
    cell_candidates = geometry.compute_collisions_points(tree, pts)
    colliding_cells = geometry.compute_colliding_cells(msh, cell_candidates, pts)

    local_vals = np.full(nx * ny, np.nan, dtype=np.float64)
    points_on_proc = []
    cells_on_proc = []
    ids_on_proc = []
    for i in range(nx * ny):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(pts[i])
            cells_on_proc.append(links[0])
            ids_on_proc.append(i)

    if points_on_proc:
        vals = uh.eval(np.array(points_on_proc, dtype=np.float64),
                       np.array(cells_on_proc, dtype=np.int32))
        local_vals[np.array(ids_on_proc, dtype=np.int32)] = np.asarray(vals).reshape(-1)

    gathered = msh.comm.gather(local_vals, root=0)
    if msh.comm.rank == 0:
        global_vals = np.full(nx * ny, np.nan, dtype=np.float64)
        for arr in gathered:
            mask = np.isnan(global_vals) & ~np.isnan(arr)
            global_vals[mask] = arr[mask]
        if np.isnan(global_vals).any():
            raise RuntimeError("Some grid points could not be evaluated.")
        return global_vals.reshape(ny, nx)
    return None


def solve(case_spec: Dict[str, Any]) -> Dict[str, Any]:
    comm = MPI.COMM_WORLD

    pde = case_spec.get("pde", {})
    out_grid = case_spec["output"]["grid"]

    t0 = float(pde.get("t0", 0.0))
    t_end = float(pde.get("t_end", 0.1))
    dt_suggested = float(pde.get("dt", 0.01))

    element_degree = 2
    mesh_resolution = 56
    dt = min(dt_suggested, 0.005)
    n_steps = max(1, int(round((t_end - t0) / dt)))
    dt = (t_end - t0) / n_steps

    msh = mesh.create_unit_square(comm, mesh_resolution, mesh_resolution, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(msh, ("Lagrange", element_degree))
    x = ufl.SpatialCoordinate(msh)

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    t_c = fem.Constant(msh, ScalarType(t0))
    dt_c = fem.Constant(msh, ScalarType(dt))

    u_n = fem.Function(V)
    u_n.interpolate(fem.Expression(_ufact(x, ScalarType(t0)), V.element.interpolation_points))

    uh = fem.Function(V)
    uD = fem.Function(V)
    uD.interpolate(fem.Expression(_ufact(x, t_c), V.element.interpolation_points))

    fdim = msh.topology.dim - 1
    facets = mesh.locate_entities_boundary(msh, fdim, lambda X: np.ones(X.shape[1], dtype=bool))
    dofs = fem.locate_dofs_topological(V, fdim, facets)
    bc = fem.dirichletbc(uD, dofs)

    a = (u * v + dt_c * ufl.inner(_kappa(x) * ufl.grad(u), ufl.grad(v))) * ufl.dx
    L = (u_n * v + dt_c * _source(x, t_c) * v) * ufl.dx

    a_form = fem.form(a)
    L_form = fem.form(L)

    A = petsc.assemble_matrix(a_form, bcs=[bc])
    A.assemble()
    b = petsc.create_vector(L_form.function_spaces)

    ksp = PETSc.KSP().create(comm)
    ksp.setOperators(A)
    ksp.setType("cg")
    ksp.getPC().setType("hypre")
    ksp.setTolerances(rtol=1e-10, atol=1e-12, max_it=5000)
    ksp.setFromOptions()

    total_iterations = 0
    start = time.perf_counter()

    u_initial = _sample_on_grid(msh, u_n, out_grid)

    for n in range(1, n_steps + 1):
        t_c.value = ScalarType(t0 + n * dt)
        uD.interpolate(fem.Expression(_ufact(x, t_c), V.element.interpolation_points))

        with b.localForm() as loc:
            loc.set(0.0)
        petsc.assemble_vector(b, L_form)
        petsc.apply_lifting(b, [a_form], bcs=[[bc]])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        petsc.set_bc(b, [bc])

        ksp.solve(b, uh.x.petsc_vec)
        uh.x.scatter_forward()
        total_iterations += ksp.getIterationNumber()

        u_n.x.array[:] = uh.x.array
        u_n.x.scatter_forward()

    wall_time = time.perf_counter() - start

    u_exact_T = fem.Function(V)
    u_exact_T.interpolate(fem.Expression(_ufact(x, ScalarType(t_end)), V.element.interpolation_points))
    e = fem.Function(V)
    e.x.array[:] = uh.x.array - u_exact_T.x.array
    e.x.scatter_forward()

    l2e_local = fem.assemble_scalar(fem.form(ufl.inner(e, e) * ufl.dx))
    l2u_local = fem.assemble_scalar(fem.form(ufl.inner(u_exact_T, u_exact_T) * ufl.dx))
    l2_error = math.sqrt(comm.allreduce(l2e_local, op=MPI.SUM))
    l2_norm = math.sqrt(comm.allreduce(l2u_local, op=MPI.SUM))
    rel_l2_error = l2_error / l2_norm if l2_norm > 0 else l2_error

    u_grid = _sample_on_grid(msh, uh, out_grid)

    return {
        "u": u_grid,
        "u_initial": u_initial,
        "solver_info": {
            "mesh_resolution": mesh_resolution,
            "element_degree": element_degree,
            "ksp_type": ksp.getType(),
            "pc_type": ksp.getPC().getType(),
            "rtol": 1e-10,
            "iterations": int(total_iterations),
            "dt": float(dt),
            "n_steps": int(n_steps),
            "time_scheme": "backward_euler",
            "verification_l2_error": float(l2_error),
            "verification_rel_l2_error": float(rel_l2_error),
            "wall_time_sec": float(wall_time),
        },
    }
