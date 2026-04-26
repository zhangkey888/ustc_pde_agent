import time
import math
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl

ScalarType = PETSc.ScalarType


def _default_case_spec():
    return {
        "output": {
            "grid": {
                "nx": 128,
                "ny": 128,
                "bbox": [0.0, 1.0, 0.0, 1.0],
            }
        },
        "pde": {"time": None},
    }


def _manufactured_exact_ufl(x):
    return ufl.sin(ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])


def _manufactured_source_ufl(x, eps, beta):
    u_exact = _manufactured_exact_ufl(x)
    lap_u = -2.0 * ufl.pi**2 * u_exact
    grad_u = ufl.as_vector(
        [
            ufl.pi * ufl.cos(ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1]),
            ufl.pi * ufl.sin(ufl.pi * x[0]) * ufl.cos(ufl.pi * x[1]),
        ]
    )
    return -eps * lap_u + ufl.dot(beta, grad_u)


def _build_problem(n, degree=1, eps_val=0.002, beta_vals=(25.0, 10.0), tau_scale=3.0):
    comm = MPI.COMM_WORLD
    msh = mesh.create_unit_square(comm, n, n, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(msh, ("Lagrange", degree))

    x = ufl.SpatialCoordinate(msh)
    eps_c = fem.Constant(msh, ScalarType(eps_val))
    beta_c = fem.Constant(msh, np.array(beta_vals, dtype=np.float64))

    u_ex_ufl = _manufactured_exact_ufl(x)
    f_ufl = _manufactured_source_ufl(x, eps_c, beta_c)

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    h = ufl.CellDiameter(msh)
    beta_norm = ufl.sqrt(ufl.dot(beta_c, beta_c) + 1.0e-16)
    Pe = beta_norm * h / (2.0 * eps_c + 1.0e-16)

    cothPe = (ufl.exp(2.0 * Pe) + 1.0) / (ufl.exp(2.0 * Pe) - 1.0 + 1.0e-16)
    tau_supg = tau_scale * h / (2.0 * beta_norm) * (cothPe - 1.0 / (Pe + 1.0e-16))

    Lu = -eps_c * ufl.div(ufl.grad(u)) + ufl.dot(beta_c, ufl.grad(u))
    Lv = ufl.dot(beta_c, ufl.grad(v))

    a = (
        eps_c * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
        + ufl.dot(beta_c, ufl.grad(u)) * v * ufl.dx
        + tau_supg * Lu * Lv * ufl.dx
    )
    L = f_ufl * v * ufl.dx + tau_supg * f_ufl * Lv * ufl.dx

    u_bc = fem.Function(V)
    u_bc.interpolate(fem.Expression(u_ex_ufl, V.element.interpolation_points))
    fdim = msh.topology.dim - 1
    facets = mesh.locate_entities_boundary(msh, fdim, lambda X: np.ones(X.shape[1], dtype=bool))
    dofs = fem.locate_dofs_topological(V, fdim, facets)
    bc = fem.dirichletbc(u_bc, dofs)

    petsc_options = {
        "ksp_type": "gmres",
        "ksp_rtol": 1.0e-10,
        "ksp_atol": 1.0e-12,
        "ksp_max_it": 2000,
        "pc_type": "ilu",
    }

    problem = petsc.LinearProblem(
        a,
        L,
        bcs=[bc],
        petsc_options=petsc_options,
        petsc_options_prefix=f"cd_{n}_{degree}_",
    )
    uh = problem.solve()
    uh.x.scatter_forward()

    return msh, V, uh, u_ex_ufl, petsc_options


def _compute_errors(msh, V, uh, u_ex_ufl):
    e = fem.Function(V)
    u_ex = fem.Function(V)
    u_ex.interpolate(fem.Expression(u_ex_ufl, V.element.interpolation_points))
    e.x.array[:] = uh.x.array - u_ex.x.array
    e.x.scatter_forward()

    l2_local = fem.assemble_scalar(fem.form(ufl.inner(e, e) * ufl.dx))
    h1s_local = fem.assemble_scalar(fem.form(ufl.inner(ufl.grad(e), ufl.grad(e)) * ufl.dx))
    l2 = math.sqrt(msh.comm.allreduce(l2_local, op=MPI.SUM))
    h1s = math.sqrt(msh.comm.allreduce(h1s_local, op=MPI.SUM))
    return l2, h1s


def _sample_on_grid(uh, bbox, nx, ny):
    msh = uh.function_space.mesh
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
    idx_map = []

    for i in range(nx * ny):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(pts[i])
            cells_on_proc.append(links[0])
            idx_map.append(i)

    if len(points_on_proc) > 0:
        vals = uh.eval(np.array(points_on_proc, dtype=np.float64), np.array(cells_on_proc, dtype=np.int32))
        vals = np.asarray(vals).reshape(-1)
        local_vals[np.array(idx_map, dtype=np.int32)] = vals

    comm = msh.comm
    global_vals = np.empty_like(local_vals)
    comm.Allreduce(local_vals, global_vals, op=MPI.SUM)

    nan_mask = np.isnan(global_vals)
    if np.any(nan_mask):
        global_vals[nan_mask] = 0.0

    return global_vals.reshape((ny, nx))


def solve(case_spec: dict) -> dict:
    if case_spec is None:
        case_spec = _default_case_spec()

    out_grid = case_spec.get("output", {}).get("grid", {})
    nx = int(out_grid.get("nx", 128))
    ny = int(out_grid.get("ny", 128))
    bbox = out_grid.get("bbox", [0.0, 1.0, 0.0, 1.0])

    eps_val = 0.002
    beta_vals = (25.0, 10.0)
    tol_target = 2.2e-4
    time_limit = 44.055
    soft_budget = 0.82 * time_limit

    candidates = [96, 128, 160, 192, 224, 256]
    chosen = None
    best = None
    t0 = time.time()

    for n in candidates:
        loop_t0 = time.time()
        try:
            msh, V, uh, u_ex_ufl, petsc_options = _build_problem(n=n, degree=1, eps_val=eps_val, beta_vals=beta_vals, tau_scale=3.0)
            l2, h1s = _compute_errors(msh, V, uh, u_ex_ufl)
            elapsed = time.time() - t0

            best = {
                "mesh_resolution": n,
                "element_degree": 1,
                "u": uh,
                "msh": msh,
                "V": V,
                "l2_error": l2,
                "h1_semi_error": h1s,
                "petsc_options": petsc_options,
            }

            predicted_next = elapsed + 1.7 * (time.time() - loop_t0)
            if l2 <= tol_target and predicted_next > soft_budget:
                chosen = best
                break
            if elapsed > soft_budget:
                chosen = best
                break
        except Exception:
            if best is not None:
                chosen = best
                break

    if chosen is None:
        chosen = best

    if chosen is None:
        raise RuntimeError("Failed to solve convection-diffusion problem.")

    u_grid = _sample_on_grid(chosen["u"], bbox, nx, ny)

    solver_info = {
        "mesh_resolution": int(chosen["mesh_resolution"]),
        "element_degree": int(chosen["element_degree"]),
        "ksp_type": str(chosen["petsc_options"]["ksp_type"]),
        "pc_type": str(chosen["petsc_options"]["pc_type"]),
        "rtol": float(chosen["petsc_options"]["ksp_rtol"]),
        "iterations": -1,
        "verification": {
            "manufactured_solution": "sin(pi*x)*sin(pi*y)",
            "l2_error": float(chosen["l2_error"]),
            "h1_semi_error": float(chosen["h1_semi_error"]),
            "target_error": tol_target,
        },
    }

    return {"u": u_grid, "solver_info": solver_info}


if __name__ == "__main__":
    result = solve(_default_case_spec())
    if MPI.COMM_WORLD.rank == 0:
        print(result["u"].shape)
        print(result["solver_info"])
