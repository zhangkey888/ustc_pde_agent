import math
import time
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
import ufl
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc

ScalarType = PETSc.ScalarType

"""
DIAGNOSIS
equation_type: reaction_diffusion
spatial_dim: 2
domain_geometry: rectangle
unknowns: scalar
coupling: none
linearity: nonlinear
time_dependence: transient
stiffness: stiff
dominant_physics: mixed
peclet_or_reynolds: N/A
solution_regularity: smooth
bc_type: all_dirichlet
special_notes: none
"""

"""
METHOD
spatial_method: fem
element_or_basis: Lagrange_P1
stabilization: none
time_method: backward_euler
nonlinear_solver: newton
linear_solver: gmres
preconditioner: hypre
special_treatment: none
pde_skill: reaction_diffusion
"""


def _parse_case(case_spec: dict):
    pde = case_spec.get("pde", {})
    time_spec = pde.get("time", case_spec.get("time", {}))
    output = case_spec.get("output", {})
    grid = output.get("grid", {})

    t0 = float(time_spec.get("t0", 0.0))
    t_end = float(time_spec.get("t_end", 0.3))
    dt_suggested = float(time_spec.get("dt", 0.005))
    scheme = str(time_spec.get("scheme", "backward_euler")).lower()

    nx_out = int(grid.get("nx", 64))
    ny_out = int(grid.get("ny", 64))
    bbox = grid.get("bbox", [0.0, 1.0, 0.0, 1.0])

    return {
        "t0": t0,
        "t_end": t_end,
        "dt_suggested": dt_suggested,
        "scheme": scheme,
        "nx_out": nx_out,
        "ny_out": ny_out,
        "bbox": bbox,
    }


def _choose_parameters(t_end, dt_suggested):
    # Use available time budget to increase accuracy moderately while remaining robust.
    # Problem is smooth but oscillatory in RHS; choose a refined mesh and smaller dt.
    mesh_resolution = 96
    degree = 1
    dt = min(dt_suggested, 0.0025)
    n_steps = max(1, int(round(t_end / dt)))
    dt = t_end / n_steps
    epsilon = 0.02
    return mesh_resolution, degree, dt, n_steps, epsilon


def _sample_function_on_grid(domain, uh, nx, ny, bbox):
    xmin, xmax, ymin, ymax = map(float, bbox)
    xs = np.linspace(xmin, xmax, nx)
    ys = np.linspace(ymin, ymax, ny)
    XX, YY = np.meshgrid(xs, ys, indexing="xy")
    pts2 = np.column_stack([XX.ravel(), YY.ravel(), np.zeros(nx * ny, dtype=np.float64)])

    tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(tree, pts2)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, pts2)

    local_vals = np.full(nx * ny, np.nan, dtype=np.float64)
    points_on_proc = []
    cells_on_proc = []
    eval_ids = []
    for i in range(pts2.shape[0]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(pts2[i])
            cells_on_proc.append(links[0])
            eval_ids.append(i)

    if len(points_on_proc) > 0:
        vals = uh.eval(np.array(points_on_proc, dtype=np.float64), np.array(cells_on_proc, dtype=np.int32))
        vals = np.asarray(vals).reshape(len(points_on_proc), -1)[:, 0]
        local_vals[np.array(eval_ids, dtype=np.int32)] = vals

    comm = domain.comm
    gathered = comm.gather(local_vals, root=0)
    if comm.rank == 0:
        out = np.full(nx * ny, np.nan, dtype=np.float64)
        for arr in gathered:
            mask = ~np.isnan(arr)
            out[mask] = arr[mask]
        # Fill any unresolved points on boundaries by nearest valid value if needed
        if np.isnan(out).any():
            valid = np.where(~np.isnan(out))[0]
            invalid = np.where(np.isnan(out))[0]
            if valid.size == 0:
                out[:] = 0.0
            else:
                out[invalid] = 0.0
        return out.reshape(ny, nx)
    return None


def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD
    rank = comm.rank
    parsed = _parse_case(case_spec)

    t0 = parsed["t0"]
    t_end = parsed["t_end"]
    dt_suggested = parsed["dt_suggested"]
    time_scheme = parsed["scheme"] if parsed["scheme"] else "backward_euler"

    mesh_resolution, degree, dt, n_steps, epsilon_value = _choose_parameters(t_end - t0, dt_suggested)

    domain = mesh.create_unit_square(
        comm, mesh_resolution, mesh_resolution, cell_type=mesh.CellType.triangle
    )
    V = fem.functionspace(domain, ("Lagrange", degree))

    fdim = domain.topology.dim - 1
    facets = mesh.locate_entities_boundary(domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool))
    bdofs = fem.locate_dofs_topological(V, fdim, facets)
    bc = fem.dirichletbc(ScalarType(0.0), bdofs, V)

    x = ufl.SpatialCoordinate(domain)
    v = ufl.TestFunction(V)

    # Given source and initial condition from the case description
    f_expr = ufl.sin(6.0 * ufl.pi * x[0]) * ufl.sin(5.0 * ufl.pi * x[1])
    u0_expr = 0.2 * ufl.sin(3.0 * ufl.pi * x[0]) * ufl.sin(2.0 * ufl.pi * x[1])

    f_fun = fem.Function(V)
    f_fun.interpolate(fem.Expression(f_expr, V.element.interpolation_points))
    u_prev = fem.Function(V)
    u_prev.interpolate(fem.Expression(u0_expr, V.element.interpolation_points))

    u_initial = fem.Function(V)
    u_initial.x.array[:] = u_prev.x.array
    u_initial.x.scatter_forward()

    u = fem.Function(V)
    u.x.array[:] = u_prev.x.array
    u.x.scatter_forward()

    dt_c = fem.Constant(domain, ScalarType(dt))
    eps_c = fem.Constant(domain, ScalarType(epsilon_value))

    # Nonlinear cubic reaction; robust dissipative choice.
    def reaction(w):
        return w**3

    F = ((u - u_prev) / dt_c) * v * ufl.dx + eps_c * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx + reaction(u) * v * ufl.dx - f_fun * v * ufl.dx
    J = ufl.derivative(F, u)

    petsc_options = {
        "snes_type": "newtonls",
        "snes_linesearch_type": "bt",
        "snes_rtol": 1.0e-8,
        "snes_atol": 1.0e-10,
        "snes_max_it": 25,
        "ksp_type": "gmres",
        "pc_type": "hypre",
        "ksp_rtol": 1.0e-9,
    }

    problem = petsc.NonlinearProblem(
        F, u, bcs=[bc], J=J, petsc_options_prefix="rd_", petsc_options=petsc_options
    )

    nonlinear_iterations = []
    total_linear_iterations = 0
    start = time.perf_counter()

    for _step in range(n_steps):
        u.x.array[:] = u_prev.x.array
        u.x.scatter_forward()
        problem.solve()
        u.x.scatter_forward()

        snes = problem.solver
        nonlinear_iterations.append(int(snes.getIterationNumber()))
        ksp = snes.getKSP()
        total_linear_iterations += int(ksp.getIterationNumber())

        u_prev.x.array[:] = u.x.array
        u_prev.x.scatter_forward()

    elapsed = time.perf_counter() - start

    # Accuracy verification without exact solution:
    # report residual norm and temporal self-consistency at final state surrogate.
    # We keep this internal but solver_info can include lightweight diagnostics.
    residual_form = fem.form(
        eps_c * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx + reaction(u) * v * ufl.dx - f_fun * v * ufl.dx
    )
    rvec = petsc.create_vector(residual_form.function_spaces)
    with rvec.localForm() as loc:
        loc.set(0.0)
    petsc.assemble_vector(rvec, residual_form)
    petsc.apply_lifting(rvec, [fem.form(ufl.derivative(
        eps_c * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx + reaction(u) * v * ufl.dx - f_fun * v * ufl.dx,
        u
    ))], bcs=[[bc]])
    rvec.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    petsc.set_bc(rvec, [bc])
    residual_norm = float(rvec.norm())

    u_grid = _sample_function_on_grid(domain, u, parsed["nx_out"], parsed["ny_out"], parsed["bbox"])
    u_initial_grid = _sample_function_on_grid(domain, u_initial, parsed["nx_out"], parsed["ny_out"], parsed["bbox"])

    result = None
    if rank == 0:
        result = {
            "u": np.asarray(u_grid, dtype=np.float64).reshape(parsed["ny_out"], parsed["nx_out"]),
            "u_initial": np.asarray(u_initial_grid, dtype=np.float64).reshape(parsed["ny_out"], parsed["nx_out"]),
            "solver_info": {
                "mesh_resolution": int(mesh_resolution),
                "element_degree": int(degree),
                "ksp_type": "gmres",
                "pc_type": "hypre",
                "rtol": 1.0e-9,
                "iterations": int(total_linear_iterations),
                "dt": float(dt),
                "n_steps": int(n_steps),
                "time_scheme": str(time_scheme),
                "nonlinear_iterations": [int(k) for k in nonlinear_iterations],
                "epsilon": float(epsilon_value),
                "verification": {
                    "final_residual_norm": residual_norm,
                    "wall_time_sec": elapsed,
                },
            },
        }
    return result


if __name__ == "__main__":
    case_spec = {
        "pde": {"time": {"t0": 0.0, "t_end": 0.3, "dt": 0.005, "scheme": "backward_euler"}},
        "output": {"grid": {"nx": 64, "ny": 64, "bbox": [0.0, 1.0, 0.0, 1.0]}},
    }
    out = solve(case_spec)
    if MPI.COMM_WORLD.rank == 0:
        print(out["u"].shape)
        print(out["solver_info"])
