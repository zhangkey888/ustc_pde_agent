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


def solve(case_spec: dict) -> dict:
    r"""
    ```DIAGNOSIS
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
    ```

    ```METHOD
    spatial_method: fem
    element_or_basis: Lagrange_P2
    stabilization: none
    time_method: backward_euler
    nonlinear_solver: none
    linear_solver: cg
    preconditioner: hypre
    special_treatment: none
    pde_skill: heat
    ```
    """
    comm = MPI.COMM_WORLD

    pde = case_spec.get("pde", {})
    coeffs = case_spec.get("coefficients", {})
    time_spec = pde.get("time", case_spec.get("time", {}))
    output_grid = case_spec["output"]["grid"]

    t0 = float(time_spec.get("t0", 0.0))
    t_end = float(time_spec.get("t_end", 0.05))
    dt_suggested = float(time_spec.get("dt", 0.005))
    scheme = str(time_spec.get("scheme", "backward_euler")).lower()
    if scheme != "backward_euler":
        scheme = "backward_euler"

    kappa = float(coeffs.get("kappa", 10.0))

    nx_out = int(output_grid["nx"])
    ny_out = int(output_grid["ny"])
    xmin, xmax, ymin, ymax = map(float, output_grid["bbox"])

    element_degree = 2
    mesh_resolution = 64
    dt = min(dt_suggested, 0.002)
    n_steps = int(round((t_end - t0) / dt))
    if n_steps < 1:
        n_steps = 1
    dt = (t_end - t0) / n_steps

    domain = mesh.create_unit_square(
        comm, mesh_resolution, mesh_resolution, cell_type=mesh.CellType.triangle
    )
    V = fem.functionspace(domain, ("Lagrange", element_degree))
    x = ufl.SpatialCoordinate(domain)

    def u_exact_expr(t):
        return ufl.exp(-t) * ufl.sin(ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])

    def f_expr(t):
        return (-1.0 + 2.0 * kappa * math.pi**2) * ufl.exp(-t) * ufl.sin(ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])

    fdim = domain.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda X: np.ones(X.shape[1], dtype=bool)
    )
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)

    uD = fem.Function(V)
    bc = fem.dirichletbc(uD, boundary_dofs)

    u_n = fem.Function(V)
    u0_expr = fem.Expression(u_exact_expr(ScalarType(t0)), V.element.interpolation_points)
    u_n.interpolate(u0_expr)
    u_n.x.scatter_forward()

    def sample_on_uniform_grid(u_func):
        xs = np.linspace(xmin, xmax, nx_out)
        ys = np.linspace(ymin, ymax, ny_out)
        XX, YY = np.meshgrid(xs, ys, indexing="xy")
        pts = np.column_stack([XX.ravel(), YY.ravel(), np.zeros(nx_out * ny_out)])

        tree = geometry.bb_tree(domain, domain.topology.dim)
        cell_candidates = geometry.compute_collisions_points(tree, pts)
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
            vals = u_func.eval(np.array(points_on_proc, dtype=np.float64), np.array(cells_on_proc, dtype=np.int32))
            vals = np.asarray(vals).reshape(-1)
            values[np.array(eval_map, dtype=np.int64)] = vals

        if np.isnan(values).any():
            values = np.nan_to_num(values)

        return values.reshape((ny_out, nx_out))

    u_initial_grid = sample_on_uniform_grid(u_n)

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    dt_c = fem.Constant(domain, ScalarType(dt))
    kappa_c = fem.Constant(domain, ScalarType(kappa))
    t_c = fem.Constant(domain, ScalarType(t0 + dt))

    a = (u * v / dt_c) * ufl.dx + kappa_c * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = (u_n * v / dt_c) * ufl.dx + f_expr(t_c) * v * ufl.dx

    a_form = fem.form(a)
    L_form = fem.form(L)

    A = petsc.assemble_matrix(a_form, bcs=[bc])
    A.assemble()

    b = petsc.create_vector(L_form.function_spaces)

    solver = PETSc.KSP().create(comm)
    solver.setOperators(A)
    solver.setType("cg")
    solver.getPC().setType("hypre")
    solver.setTolerances(rtol=1e-10, atol=1e-14, max_it=2000)
    solver.setFromOptions()

    uh = fem.Function(V)
    total_iterations = 0

    t_start = time.perf_counter()

    t = t0
    for step in range(1, n_steps + 1):
        t = t0 + step * dt
        t_c.value = ScalarType(t)

        uD_expr = fem.Expression(u_exact_expr(t_c), V.element.interpolation_points)
        uD.interpolate(uD_expr)
        uD.x.scatter_forward()

        with b.localForm() as loc:
            loc.set(0.0)
        petsc.assemble_vector(b, L_form)
        petsc.apply_lifting(b, [a_form], bcs=[[bc]])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        petsc.set_bc(b, [bc])

        try:
            solver.solve(b, uh.x.petsc_vec)
        except RuntimeError:
            solver.setType("preonly")
            solver.getPC().setType("lu")
            solver.setOperators(A)
            solver.solve(b, uh.x.petsc_vec)

        uh.x.scatter_forward()
        total_iterations += solver.getIterationNumber()
        u_n.x.array[:] = uh.x.array
        u_n.x.scatter_forward()

    wall_time = time.perf_counter() - t_start

    u_ex_T = fem.Function(V)
    uT_expr = fem.Expression(u_exact_expr(ScalarType(t_end)), V.element.interpolation_points)
    u_ex_T.interpolate(uT_expr)
    u_ex_T.x.scatter_forward()

    diff = fem.Function(V)
    diff.x.array[:] = uh.x.array - u_ex_T.x.array
    diff.x.scatter_forward()

    err_l2_sq_local = fem.assemble_scalar(fem.form(diff * diff * ufl.dx))
    ex_l2_sq_local = fem.assemble_scalar(fem.form(u_ex_T * u_ex_T * ufl.dx))
    err_l2_sq = comm.allreduce(err_l2_sq_local, op=MPI.SUM)
    ex_l2_sq = comm.allreduce(ex_l2_sq_local, op=MPI.SUM)

    abs_l2_error = math.sqrt(max(err_l2_sq, 0.0))
    rel_l2_error = abs_l2_error / math.sqrt(max(ex_l2_sq, 1e-30))

    u_grid = sample_on_uniform_grid(uh)

    solver_info: Dict[str, Any] = {
        "mesh_resolution": mesh_resolution,
        "element_degree": element_degree,
        "ksp_type": solver.getType(),
        "pc_type": solver.getPC().getType(),
        "rtol": 1e-10,
        "iterations": int(total_iterations),
        "dt": float(dt),
        "n_steps": int(n_steps),
        "time_scheme": "backward_euler",
        "abs_l2_error": float(abs_l2_error),
        "rel_l2_error": float(rel_l2_error),
        "wall_time_sec_internal": float(wall_time),
    }

    return {"u": u_grid, "u_initial": u_initial_grid, "solver_info": solver_info}


if __name__ == "__main__":
    case_spec = {
        "pde": {"time": {"t0": 0.0, "t_end": 0.05, "dt": 0.005, "scheme": "backward_euler"}},
        "coefficients": {"kappa": 10.0},
        "output": {"grid": {"nx": 64, "ny": 64, "bbox": [0.0, 1.0, 0.0, 1.0]}},
    }
    out = solve(case_spec)
    if MPI.COMM_WORLD.rank == 0:
        print(out["u"].shape)
        print(out["solver_info"])
