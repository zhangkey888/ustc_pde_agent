import time
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl


def _sample_function_on_grid(u_func, domain, nx, ny, bbox):
    xmin, xmax, ymin, ymax = bbox
    xs = np.linspace(xmin, xmax, nx)
    ys = np.linspace(ymin, ymax, ny)
    XX, YY = np.meshgrid(xs, ys, indexing="xy")
    pts2 = np.column_stack([XX.ravel(), YY.ravel()])
    pts3 = np.zeros((pts2.shape[0], 3), dtype=np.float64)
    pts3[:, :2] = pts2

    tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(tree, pts3)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, pts3)

    values = np.full((pts3.shape[0],), np.nan, dtype=np.float64)
    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    for i in range(pts3.shape[0]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(pts3[i])
            cells_on_proc.append(links[0])
            eval_map.append(i)

    if len(points_on_proc) > 0:
        vals = u_func.eval(np.array(points_on_proc, dtype=np.float64),
                           np.array(cells_on_proc, dtype=np.int32))
        vals = np.real(vals).reshape(-1)
        values[np.array(eval_map, dtype=np.int32)] = vals

    comm = domain.comm
    gathered = comm.allgather(values)
    out = np.full_like(values, np.nan)
    for arr in gathered:
        mask = ~np.isnan(arr)
        out[mask] = arr[mask]

    if np.isnan(out).any():
        # Robust fallback for boundary points
        out = np.nan_to_num(out, nan=0.0)

    return out.reshape(ny, nx)


def solve(case_spec: dict) -> dict:
    """
    Solve -Δu - k^2 u = f on [0,1]^2 with manufactured solution
    u = x(1-x)y(1-y), and sample onto requested output grid.
    """
    comm = MPI.COMM_WORLD
    rank = comm.rank
    ScalarType = PETSc.ScalarType

    grid = case_spec["output"]["grid"]
    nx_out = int(grid["nx"])
    ny_out = int(grid["ny"])
    bbox = grid["bbox"]

    pde = case_spec.get("pde", {})
    k = float(pde.get("k", 2.0))

    # Adaptive accuracy-time tradeoff for strict budget:
    # P2 on a moderately fine mesh is accurate for this polynomial manufactured case.
    # Keep direct LU for robustness and low wall time at low k.
    mesh_resolution = 48
    element_degree = 2
    rtol = 1.0e-10
    ksp_type = "preonly"
    pc_type = "lu"

    t0 = time.perf_counter()

    domain = mesh.create_unit_square(
        comm, mesh_resolution, mesh_resolution, cell_type=mesh.CellType.triangle
    )
    V = fem.functionspace(domain, ("Lagrange", element_degree))

    x = ufl.SpatialCoordinate(domain)
    u_exact_ufl = x[0] * (1.0 - x[0]) * x[1] * (1.0 - x[1])

    f_ufl = -ufl.div(ufl.grad(u_exact_ufl)) - ScalarType(k * k) * u_exact_ufl

    uD = fem.Function(V)
    uD.interpolate(fem.Expression(u_exact_ufl, V.element.interpolation_points))

    fdim = domain.topology.dim - 1
    facets = mesh.locate_entities_boundary(
        domain, fdim, lambda X: np.ones(X.shape[1], dtype=bool)
    )
    dofs = fem.locate_dofs_topological(V, fdim, facets)
    bc = fem.dirichletbc(uD, dofs)

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    a = (ufl.inner(ufl.grad(u), ufl.grad(v)) - ScalarType(k * k) * ufl.inner(u, v)) * ufl.dx
    L = ufl.inner(f_ufl, v) * ufl.dx

    problem = petsc.LinearProblem(
        a, L, bcs=[bc],
        petsc_options_prefix="helmholtz_",
        petsc_options={
            "ksp_type": ksp_type,
            "pc_type": pc_type,
            "ksp_rtol": rtol,
        },
    )

    uh = problem.solve()
    uh.x.scatter_forward()

    # Accuracy verification: relative/absolute L2 error against exact manufactured solution
    err_form = fem.form((uh - uD) ** 2 * ufl.dx)
    exact_form = fem.form(uD ** 2 * ufl.dx)
    l2_err_sq_local = fem.assemble_scalar(err_form)
    l2_ex_sq_local = fem.assemble_scalar(exact_form)
    l2_err = np.sqrt(comm.allreduce(l2_err_sq_local, op=MPI.SUM))
    l2_ex = np.sqrt(comm.allreduce(l2_ex_sq_local, op=MPI.SUM))
    rel_l2_err = l2_err / l2_ex if l2_ex > 0 else l2_err

    u_grid = _sample_function_on_grid(uh, domain, nx_out, ny_out, bbox)

    elapsed = time.perf_counter() - t0

    # Iteration info from underlying KSP if available
    try:
        ksp = problem.solver
        iterations = int(ksp.getIterationNumber())
        actual_ksp_type = ksp.getType()
        actual_pc_type = ksp.getPC().getType()
    except Exception:
        iterations = 0
        actual_ksp_type = ksp_type
        actual_pc_type = pc_type

    solver_info = {
        "mesh_resolution": mesh_resolution,
        "element_degree": element_degree,
        "ksp_type": str(actual_ksp_type),
        "pc_type": str(actual_pc_type),
        "rtol": float(rtol),
        "iterations": int(iterations),
        "l2_error": float(l2_err),
        "relative_l2_error": float(rel_l2_err),
        "wall_time_sec": float(elapsed),
    }

    return {"u": u_grid, "solver_info": solver_info}


if __name__ == "__main__":
    case_spec = {
        "pde": {"k": 2.0, "time": None},
        "output": {"grid": {"nx": 64, "ny": 64, "bbox": [0.0, 1.0, 0.0, 1.0]}},
    }
    result = solve(case_spec)
    print(result["u"].shape)
    print(result["solver_info"])
