import math
import time
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
import ufl
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc

ScalarType = PETSc.ScalarType


def _make_exact_and_rhs(msh, epsilon, beta):
    x = ufl.SpatialCoordinate(msh)
    pi = math.pi
    u_exact = ufl.exp(x[0]) * ufl.sin(pi * x[1])

    ux = ufl.exp(x[0]) * ufl.sin(pi * x[1])
    uy = pi * ufl.exp(x[0]) * ufl.cos(pi * x[1])
    lap_u = ufl.exp(x[0]) * ufl.sin(pi * x[1]) * (1.0 - pi * pi)

    f_expr = -epsilon * lap_u + beta[0] * ux + beta[1] * uy
    return u_exact, f_expr


def _sample_function_on_grid(u_func, msh, grid_spec):
    nx = int(grid_spec["nx"])
    ny = int(grid_spec["ny"])
    bbox = grid_spec["bbox"]
    xmin, xmax, ymin, ymax = map(float, bbox)

    xs = np.linspace(xmin, xmax, nx)
    ys = np.linspace(ymin, ymax, ny)
    XX, YY = np.meshgrid(xs, ys, indexing="xy")
    pts = np.column_stack([XX.ravel(), YY.ravel(), np.zeros(nx * ny, dtype=np.float64)])

    tree = geometry.bb_tree(msh, msh.topology.dim)
    cell_candidates = geometry.compute_collisions_points(tree, pts)
    colliding_cells = geometry.compute_colliding_cells(msh, cell_candidates, pts)

    values = np.full(nx * ny, np.nan, dtype=np.float64)
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

    comm = msh.comm
    gathered = comm.gather(values, root=0)

    if comm.rank == 0:
        final_vals = np.full(nx * ny, np.nan, dtype=np.float64)
        for arr in gathered:
            mask = ~np.isnan(arr)
            final_vals[mask] = arr[mask]
        if np.isnan(final_vals).any():
            raise RuntimeError("Failed to evaluate solution at some output grid points.")
        return final_vals.reshape((ny, nx))
    return None


def _solve_once(n, degree, epsilon, beta_vec, ksp_type="gmres", pc_type="ilu", rtol=1e-9):
    comm = MPI.COMM_WORLD
    msh = mesh.create_unit_square(comm, n, n, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(msh, ("Lagrange", degree))

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    beta = ufl.as_vector((ScalarType(beta_vec[0]), ScalarType(beta_vec[1])))
    beta_norm = float(np.linalg.norm(beta_vec))

    u_exact, f_expr = _make_exact_and_rhs(msh, epsilon, beta_vec)

    u_bc = fem.Function(V)
    u_bc.interpolate(fem.Expression(u_exact, V.element.interpolation_points))
    f_fun = fem.Function(V)
    f_fun.interpolate(fem.Expression(f_expr, V.element.interpolation_points))

    fdim = msh.topology.dim - 1
    facets = mesh.locate_entities_boundary(msh, fdim, lambda x: np.ones(x.shape[1], dtype=bool))
    dofs = fem.locate_dofs_topological(V, fdim, facets)
    bc = fem.dirichletbc(u_bc, dofs)

    h = ufl.CellDiameter(msh)
    Pe_h = beta_norm * h / (2.0 * epsilon)
    coth_pe = (ufl.exp(2.0 * Pe_h) + 1.0) / (ufl.exp(2.0 * Pe_h) - 1.0)
    tau = h / (2.0 * beta_norm) * (coth_pe - 1.0 / Pe_h)

    Lu = -epsilon * ufl.div(ufl.grad(u)) + ufl.dot(beta, ufl.grad(u))
    Rv = ufl.dot(beta, ufl.grad(v))

    a = (
        epsilon * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
        + ufl.dot(beta, ufl.grad(u)) * v * ufl.dx
        + tau * Lu * Rv * ufl.dx
    )
    L = (
        f_fun * v * ufl.dx
        + tau * f_fun * Rv * ufl.dx
    )

    problem = petsc.LinearProblem(
        a,
        L,
        bcs=[bc],
        petsc_options_prefix="convdiff_",
        petsc_options={
            "ksp_type": ksp_type,
            "pc_type": pc_type,
            "ksp_rtol": rtol,
            "ksp_atol": 1e-12,
            "ksp_max_it": 5000,
        },
    )

    t0 = time.perf_counter()
    uh = problem.solve()
    uh.x.scatter_forward()
    elapsed = time.perf_counter() - t0

    ksp = problem.solver
    its = ksp.getIterationNumber()
    actual_ksp = ksp.getType()
    actual_pc = ksp.getPC().getType()

    err_L2_form = fem.form((uh - u_exact) ** 2 * ufl.dx)
    err_H1s_form = fem.form(ufl.inner(ufl.grad(uh - u_exact), ufl.grad(uh - u_exact)) * ufl.dx)
    eL2_local = fem.assemble_scalar(err_L2_form)
    eH1s_local = fem.assemble_scalar(err_H1s_form)
    eL2 = math.sqrt(comm.allreduce(eL2_local, op=MPI.SUM))
    eH1 = math.sqrt(comm.allreduce(eL2_local + eH1s_local, op=MPI.SUM))

    return {
        "mesh": msh,
        "V": V,
        "uh": uh,
        "elapsed": elapsed,
        "L2_error": eL2,
        "H1_error": eH1,
        "iterations": int(its),
        "ksp_type": str(actual_ksp),
        "pc_type": str(actual_pc),
        "rtol": float(rtol),
        "mesh_resolution": int(n),
        "element_degree": int(degree),
    }


def solve(case_spec: dict) -> dict:
    epsilon = 0.005
    beta_vec = np.array([-20.0, 5.0], dtype=np.float64)
    grid_spec = case_spec["output"]["grid"]
    time_budget = 4.755

    candidates = [
        (56, 1, "gmres", "ilu"),
        (72, 1, "gmres", "ilu"),
        (88, 1, "gmres", "ilu"),
        (64, 2, "gmres", "ilu"),
        (80, 2, "gmres", "ilu"),
    ]

    chosen = None
    spent = 0.0
    for n, degree, ksp_type, pc_type in candidates:
        result = _solve_once(n, degree, epsilon, beta_vec, ksp_type=ksp_type, pc_type=pc_type, rtol=1e-9)
        spent += result["elapsed"]
        chosen = result
        if result["L2_error"] <= 2.13e-3 and spent > 0.55 * time_budget:
            break
        if spent > 0.85 * time_budget:
            break

    if chosen is None:
        raise RuntimeError("Solver failed to produce a solution.")

    u_grid = _sample_function_on_grid(chosen["uh"], chosen["mesh"], grid_spec)

    solver_info = {
        "mesh_resolution": chosen["mesh_resolution"],
        "element_degree": chosen["element_degree"],
        "ksp_type": chosen["ksp_type"],
        "pc_type": chosen["pc_type"],
        "rtol": chosen["rtol"],
        "iterations": chosen["iterations"],
        "L2_error": chosen["L2_error"],
        "H1_error": chosen["H1_error"],
        "wall_time_sec": chosen["elapsed"],
        "stabilization": "SUPG",
    }

    if MPI.COMM_WORLD.rank == 0:
        return {"u": u_grid, "solver_info": solver_info}
    return {"u": None, "solver_info": solver_info}


if __name__ == "__main__":
    case_spec = {
        "output": {"grid": {"nx": 64, "ny": 64, "bbox": [0.0, 1.0, 0.0, 1.0]}},
        "pde": {"time": None},
    }
    out = solve(case_spec)
    if MPI.COMM_WORLD.rank == 0:
        print(out["u"].shape)
        print(out["solver_info"])
