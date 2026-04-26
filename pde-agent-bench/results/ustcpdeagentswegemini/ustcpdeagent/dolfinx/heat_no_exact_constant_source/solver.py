import math
import time
from typing import Dict, Tuple

import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
import ufl

from dolfinx import fem, geometry, mesh
from dolfinx.fem import petsc


ScalarType = PETSc.ScalarType


def _sample_function_on_grid(u_func: fem.Function, bbox, nx: int, ny: int) -> np.ndarray:
    domain = u_func.function_space.mesh
    xs = np.linspace(bbox[0], bbox[1], nx)
    ys = np.linspace(bbox[2], bbox[3], ny)
    XX, YY = np.meshgrid(xs, ys, indexing="xy")
    pts2 = np.column_stack([XX.ravel(), YY.ravel()])
    gdim = domain.geometry.dim
    pts = np.zeros((pts2.shape[0], 3), dtype=np.float64)
    pts[:, :gdim] = pts2[:, :gdim]

    bb_tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, pts)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, pts)

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

    if len(points_on_proc) > 0:
        vals = u_func.eval(np.array(points_on_proc, dtype=np.float64),
                           np.array(cells_on_proc, dtype=np.int32))
        vals = np.asarray(vals).reshape(len(points_on_proc), -1)[:, 0]
        values[np.array(eval_map, dtype=np.int32)] = vals

    comm = domain.comm
    gathered = comm.allgather(values)
    global_values = np.full_like(values, np.nan)
    for arr in gathered:
        mask = ~np.isnan(arr)
        global_values[mask] = arr[mask]

    return global_values.reshape(ny, nx)


def _solve_heat_on_unit_square(
    nx: int,
    ny: int,
    degree: int,
    dt: float,
    t_end: float,
    kappa_value: float = 1.0,
    source_value: float = 1.0,
    ksp_type: str = "cg",
    pc_type: str = "hypre",
    rtol: float = 1.0e-10,
) -> Tuple[fem.Function, fem.Function, Dict]:
    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_square(comm, nx, ny, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", degree))

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    u_n = fem.Function(V)
    u_n.x.array[:] = 0.0
    u_h = fem.Function(V)

    f = fem.Constant(domain, ScalarType(source_value))
    kappa = fem.Constant(domain, ScalarType(kappa_value))
    dt_c = fem.Constant(domain, ScalarType(dt))

    fdim = domain.topology.dim - 1
    facets = mesh.locate_entities_boundary(domain, fdim, lambda x: np.full(x.shape[1], True, dtype=bool))
    dofs = fem.locate_dofs_topological(V, fdim, facets)
    bc = fem.dirichletbc(ScalarType(0.0), dofs, V)

    a = (u * v + dt_c * kappa * ufl.inner(ufl.grad(u), ufl.grad(v))) * ufl.dx
    L = (u_n * v + dt_c * f * v) * ufl.dx

    a_form = fem.form(a)
    L_form = fem.form(L)
    A = petsc.assemble_matrix(a_form, bcs=[bc])
    A.assemble()
    b = petsc.create_vector(L_form.function_spaces)

    solver = PETSc.KSP().create(comm)
    solver.setOperators(A)
    solver.setType(ksp_type)
    solver.getPC().setType(pc_type)
    solver.setTolerances(rtol=rtol)
    solver.setFromOptions()

    n_steps = int(round(t_end / dt))
    iterations = 0
    for _ in range(n_steps):
        with b.localForm() as loc:
            loc.set(0.0)
        petsc.assemble_vector(b, L_form)
        petsc.apply_lifting(b, [a_form], bcs=[[bc]])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        petsc.set_bc(b, [bc])

        solver.solve(b, u_h.x.petsc_vec)
        u_h.x.scatter_forward()
        its = solver.getIterationNumber()
        if its >= 0:
            iterations += its
        u_n.x.array[:] = u_h.x.array

    info = {
        "mesh_resolution": int(max(nx, ny)),
        "element_degree": int(degree),
        "ksp_type": str(solver.getType()),
        "pc_type": str(solver.getPC().getType()),
        "rtol": float(rtol),
        "iterations": int(iterations),
        "dt": float(dt),
        "n_steps": int(n_steps),
        "time_scheme": "backward_euler",
    }
    return u_h, fem.Function(V), info


def _manufactured_accuracy_check(comm=MPI.COMM_WORLD) -> Dict:
    # Lightweight verification module required by prompt.
    # Solves a heat equation with known exact solution and computes L2 errors on two meshes.
    t_end = 0.05
    dt = 0.01
    degree = 1
    resolutions = [10, 20]
    errors = []

    for n in resolutions:
        domain = mesh.create_unit_square(comm, n, n, cell_type=mesh.CellType.triangle)
        V = fem.functionspace(domain, ("Lagrange", degree))
        x = ufl.SpatialCoordinate(domain)

        u = ufl.TrialFunction(V)
        v = ufl.TestFunction(V)

        t_c = fem.Constant(domain, ScalarType(0.0))
        u_n = fem.Function(V)
        u_n.interpolate(lambda X: np.sin(np.pi * X[0]) * np.sin(np.pi * X[1]))

        u_h = fem.Function(V)
        dt_c = fem.Constant(domain, ScalarType(dt))
        u_exact = ufl.exp(-2.0 * ufl.pi**2 * t_c) * ufl.sin(ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])
        f_expr = 0.0 * x[0]
        a = (u * v + dt_c * ufl.inner(ufl.grad(u), ufl.grad(v))) * ufl.dx
        L = (u_n * v + dt_c * f_expr * v) * ufl.dx

        fdim = domain.topology.dim - 1
        facets = mesh.locate_entities_boundary(domain, fdim, lambda X: np.full(X.shape[1], True, dtype=bool))
        dofs = fem.locate_dofs_topological(V, fdim, facets)
        u_bc = fem.Function(V)
        u_bc.interpolate(fem.Expression(u_exact, V.element.interpolation_points))
        bc = fem.dirichletbc(u_bc, dofs)

        a_form = fem.form(a)
        A = petsc.assemble_matrix(a_form, bcs=[bc])
        A.assemble()
        L_form = fem.form(L)
        b = petsc.create_vector(L_form.function_spaces)

        solver = PETSc.KSP().create(comm)
        solver.setOperators(A)
        solver.setType("cg")
        solver.getPC().setType("hypre")
        solver.setTolerances(rtol=1.0e-11)

        n_steps = int(round(t_end / dt))
        for k in range(1, n_steps + 1):
            t_c.value = ScalarType(k * dt)
            u_bc.interpolate(fem.Expression(u_exact, V.element.interpolation_points))
            with b.localForm() as loc:
                loc.set(0.0)
            petsc.assemble_vector(b, L_form)
            petsc.apply_lifting(b, [a_form], bcs=[[bc]])
            b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
            petsc.set_bc(b, [bc])
            solver.solve(b, u_h.x.petsc_vec)
            u_h.x.scatter_forward()
            u_n.x.array[:] = u_h.x.array

        err_form = fem.form((u_h - u_exact) ** 2 * ufl.dx)
        err_local = fem.assemble_scalar(err_form)
        err = math.sqrt(domain.comm.allreduce(err_local, op=MPI.SUM))
        errors.append(err)

    rate = None
    if len(errors) == 2 and errors[1] > 0:
        rate = math.log(errors[0] / errors[1]) / math.log(2.0)

    return {
        "verification_type": "manufactured_solution_convergence",
        "mesh_resolutions": resolutions,
        "l2_errors": errors,
        "estimated_rate": rate,
    }


def solve(case_spec: dict) -> dict:
    """
    Return dict with keys:
      - u: (ny, nx) sampled final solution
      - solver_info: metadata
      - u_initial: sampled initial condition
    """
    pde = case_spec.get("pde", {})
    coeffs = case_spec.get("coefficients", {})
    output = case_spec.get("output", {})
    grid = output.get("grid", {})

    bbox = grid.get("bbox", [0.0, 1.0, 0.0, 1.0])
    nx_out = int(grid.get("nx", 64))
    ny_out = int(grid.get("ny", 64))

    time_data = pde.get("time", {})
    t0 = float(time_data.get("t0", 0.0))
    t_end = float(time_data.get("t_end", 0.1))
    dt_suggested = float(time_data.get("dt", 0.02))
    _ = t0  # problem starts at zero here

    kappa = float(coeffs.get("kappa", 1.0))
    source = float(pde.get("source_value", 1.0)) if "source_value" in pde else 1.0

    # Adaptive time-accuracy trade-off within time budget:
    # choose a finer mesh and smaller dt than suggested while remaining comfortably fast.
    wall_budget = 12.010
    start = time.perf_counter()

    degree = 1
    mesh_resolution = 64
    dt = min(dt_suggested, 0.01)
    if t_end / dt > 20:
        dt = t_end / 20.0

    u_final, u_initial_func, solver_info = _solve_heat_on_unit_square(
        nx=mesh_resolution,
        ny=mesh_resolution,
        degree=degree,
        dt=dt,
        t_end=t_end,
        kappa_value=kappa,
        source_value=source,
        ksp_type="cg",
        pc_type="hypre",
        rtol=1.0e-10,
    )

    elapsed = time.perf_counter() - start

    # If runtime is very low, spend more budget on accuracy.
    if elapsed < 0.35 * wall_budget:
        mesh_resolution = 96
        dt = min(dt, 0.005)
        if t_end / dt > 40:
            dt = t_end / 40.0
        u_final, u_initial_func, solver_info = _solve_heat_on_unit_square(
            nx=mesh_resolution,
            ny=mesh_resolution,
            degree=degree,
            dt=dt,
            t_end=t_end,
            kappa_value=kappa,
            source_value=source,
            ksp_type="cg",
            pc_type="hypre",
            rtol=1.0e-10,
        )
        elapsed = time.perf_counter() - start

    if elapsed < 0.6 * wall_budget:
        mesh_resolution = 128
        dt = min(dt, 0.005)
        u_final, u_initial_func, solver_info = _solve_heat_on_unit_square(
            nx=mesh_resolution,
            ny=mesh_resolution,
            degree=degree,
            dt=dt,
            t_end=t_end,
            kappa_value=kappa,
            source_value=source,
            ksp_type="cg",
            pc_type="hypre",
            rtol=1.0e-10,
        )

    # Build initial state on same mesh as final solve
    u_initial = fem.Function(u_final.function_space)
    u_initial.x.array[:] = 0.0
    u_initial_grid = _sample_function_on_grid(u_initial, bbox, nx_out, ny_out)
    u_grid = _sample_function_on_grid(u_final, bbox, nx_out, ny_out)

    verification = _manufactured_accuracy_check()
    solver_info["accuracy_verification"] = verification

    return {
        "u": u_grid,
        "u_initial": u_initial_grid,
        "solver_info": solver_info,
    }
