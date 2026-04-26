import time
import math
import json
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl

ScalarType = PETSc.ScalarType


def _source_ufl(domain):
    x = ufl.SpatialCoordinate(domain)
    return (
        ufl.exp(-220.0 * ((x[0] - 0.25) ** 2 + (x[1] - 0.25) ** 2))
        + ufl.exp(-220.0 * ((x[0] - 0.75) ** 2 + (x[1] - 0.7) ** 2))
    )


def _build_problem(nx: int, degree: int, dt: float, t_end: float):
    domain = mesh.create_unit_square(MPI.COMM_WORLD, nx, nx, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", degree))

    fdim = domain.topology.dim - 1
    facets = mesh.locate_entities_boundary(domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool))
    dofs = fem.locate_dofs_topological(V, fdim, facets)
    bc = fem.dirichletbc(ScalarType(0.0), dofs, V)

    u_n = fem.Function(V)
    u_n.x.array[:] = 0.0
    uh = fem.Function(V)

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    f_expr = _source_ufl(domain)

    a = (u * v + ScalarType(dt) * ufl.inner(ufl.grad(u), ufl.grad(v))) * ufl.dx
    L = (u_n * v + ScalarType(dt) * f_expr * v) * ufl.dx

    a_form = fem.form(a)
    L_form = fem.form(L)
    A = petsc.assemble_matrix(a_form, bcs=[bc])
    A.assemble()
    b = petsc.create_vector(L_form.function_spaces)

    solver = PETSc.KSP().create(domain.comm)
    solver.setOperators(A)
    solver.setType("cg")
    pc = solver.getPC()
    pc.setType("hypre")
    solver.setTolerances(rtol=1e-9, atol=1e-12, max_it=3000)
    solver.setFromOptions()

    n_steps = max(1, int(round(t_end / dt)))
    return domain, V, bc, u_n, uh, a_form, L_form, b, solver, float(dt), n_steps


def _run(domain, V, bc, u_n, uh, a_form, L_form, b, solver, dt, n_steps):
    iterations = 0
    for _ in range(n_steps):
        with b.localForm() as loc:
            loc.set(0.0)
        petsc.assemble_vector(b, L_form)
        petsc.apply_lifting(b, [a_form], bcs=[[bc]])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        petsc.set_bc(b, [bc])
        solver.solve(b, uh.x.petsc_vec)
        uh.x.scatter_forward()
        iterations += solver.getIterationNumber()
        u_n.x.array[:] = uh.x.array
    return int(iterations)


def _eval_points(domain, uh, pts):
    tree = geometry.bb_tree(domain, domain.topology.dim)
    candidates = geometry.compute_collisions_points(tree, pts)
    colliding = geometry.compute_colliding_cells(domain, candidates, pts)

    values = np.full(pts.shape[0], np.nan, dtype=np.float64)
    points_on_proc = []
    cells = []
    ids = []
    for i in range(pts.shape[0]):
        links = colliding.links(i)
        if len(links) > 0:
            points_on_proc.append(pts[i])
            cells.append(links[0])
            ids.append(i)

    if points_on_proc:
        vals = uh.eval(np.array(points_on_proc, dtype=np.float64), np.array(cells, dtype=np.int32))
        values[np.array(ids, dtype=np.int32)] = np.asarray(vals).reshape(-1)

    values = np.where(np.isnan(values), 0.0, values)
    return domain.comm.allreduce(values, op=MPI.SUM)


def _sample_grid(domain, uh, grid):
    nx = int(grid["nx"])
    ny = int(grid["ny"])
    xmin, xmax, ymin, ymax = grid["bbox"]
    xs = np.linspace(xmin, xmax, nx)
    ys = np.linspace(ymin, ymax, ny)
    X, Y = np.meshgrid(xs, ys, indexing="xy")
    pts = np.column_stack([X.ravel(), Y.ravel(), np.zeros(nx * ny)])
    vals = _eval_points(domain, uh, pts)
    return vals.reshape((ny, nx))


def solve(case_spec: dict) -> dict:
    """
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
    special_notes: none
    ```
    ```METHOD
    spatial_method: fem
    element_or_basis: Lagrange_P2
    stabilization: none
    time_method: backward_euler
    nonlinear_solver: none
    linear_solver: cg
    preconditioner: amg
    special_treatment: none
    pde_skill: heat
    ```
    """
    pde = case_spec.get("pde", {})
    grid = case_spec["output"]["grid"]
    t_end = float(pde.get("t_end", case_spec.get("t_end", 0.1)))
    dt_suggest = float(pde.get("dt", case_spec.get("dt", 0.02)))

    wall_budget = 12.0
    start = time.perf_counter()

    candidates = [
        (72, 2, min(dt_suggest, 0.01)),
        (88, 2, 0.008),
        (104, 2, 0.00625),
        (120, 2, 0.005),
        (136, 2, 0.004),
    ]

    probe_x = np.linspace(0.05, 0.95, 29)
    probe_y = np.linspace(0.05, 0.95, 29)
    PX, PY = np.meshgrid(probe_x, probe_y, indexing="xy")
    probe_pts = np.column_stack([PX.ravel(), PY.ravel(), np.zeros(PX.size)])

    best = None
    previous_probe = None

    for nx, degree, dt in candidates:
        if time.perf_counter() - start > wall_budget:
            break
        try:
            pack = _build_problem(nx, degree, dt, t_end)
            domain, V, bc, u_n, uh, a_form, L_form, b, solver, dt_used, n_steps = pack
            iterations = _run(*pack)
            probe = _eval_points(domain, uh, probe_pts)
            err_est = None
            if previous_probe is not None:
                err_est = float(np.linalg.norm(probe - previous_probe) / math.sqrt(probe.size))
            previous_probe = probe
            best = (
                domain,
                uh,
                {
                    "mesh_resolution": int(nx),
                    "element_degree": int(degree),
                    "ksp_type": solver.getType(),
                    "pc_type": solver.getPC().getType(),
                    "rtol": 1.0e-9,
                    "iterations": int(iterations),
                    "dt": float(dt_used),
                    "n_steps": int(n_steps),
                    "time_scheme": "backward_euler",
                    "accuracy_verification": {
                        "type": "self_convergence_probe",
                        "probe_l2_difference_to_previous": err_est,
                        "n_probe_points": int(probe.size),
                    },
                },
            )
            if err_est is not None and err_est < 1.0e-4 and (time.perf_counter() - start) > 6.0:
                break
        except Exception:
            continue

    if best is None:
        pack = _build_problem(64, 1, min(dt_suggest, 0.01), t_end)
        domain, V, bc, u_n, uh, a_form, L_form, b, solver, dt_used, n_steps = pack
        iterations = _run(*pack)
        info = {
            "mesh_resolution": 64,
            "element_degree": 1,
            "ksp_type": solver.getType(),
            "pc_type": solver.getPC().getType(),
            "rtol": 1.0e-9,
            "iterations": int(iterations),
            "dt": float(dt_used),
            "n_steps": int(n_steps),
            "time_scheme": "backward_euler",
            "accuracy_verification": {"type": "fallback_run"},
        }
    else:
        domain, uh, info = best

    u_grid = _sample_grid(domain, uh, grid)
    return {
        "u": u_grid,
        "u_initial": np.zeros_like(u_grid),
        "solver_info": info,
    }


if __name__ == "__main__":
    case_spec = {
        "pde": {"time": True, "t0": 0.0, "t_end": 0.1, "dt": 0.02, "scheme": "backward_euler"},
        "output": {"grid": {"nx": 64, "ny": 64, "bbox": [0.0, 1.0, 0.0, 1.0]}},
    }
    t0 = time.perf_counter()
    out = solve(case_spec)
    wt = time.perf_counter() - t0
    l2_err = out["solver_info"].get("accuracy_verification", {}).get("probe_l2_difference_to_previous", -1.0)
    if l2_err is None:
        l2_err = -1.0
    if MPI.COMM_WORLD.rank == 0:
        print("L2_ERROR:", float(l2_err))
        print("WALL_TIME:", float(wt))
        print(json.dumps(out["solver_info"]))
