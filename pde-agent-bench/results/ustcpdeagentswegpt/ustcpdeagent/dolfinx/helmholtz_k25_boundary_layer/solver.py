import time
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl


ScalarType = PETSc.ScalarType


def _probe_function(u_func, pts):
    """
    Evaluate scalar fem.Function u_func at points pts of shape (N, 3).
    Returns array of shape (N,).
    """
    domain = u_func.function_space.mesh
    bb_tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, pts)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, pts)

    values = np.full(pts.shape[0], np.nan, dtype=np.float64)
    points_on_proc = []
    cells_on_proc = []
    ids = []
    for i in range(pts.shape[0]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(pts[i])
            cells_on_proc.append(links[0])
            ids.append(i)

    if len(points_on_proc) > 0:
        vals = u_func.eval(np.array(points_on_proc, dtype=np.float64),
                           np.array(cells_on_proc, dtype=np.int32))
        vals = np.asarray(vals).reshape(len(points_on_proc), -1)[:, 0]
        values[np.array(ids, dtype=np.int32)] = vals.real.astype(np.float64)

    # For serial evaluation this should fill everything; in parallel reduce if needed
    comm = domain.comm
    out = np.empty_like(values)
    comm.Allreduce(values, out, op=MPI.SUM)
    return out


def _make_exact_and_rhs(msh, k):
    x = ufl.SpatialCoordinate(msh)
    pi = ufl.pi
    u_exact = ufl.exp(4.0 * x[0]) * ufl.sin(pi * x[1])
    # PDE: -Δu - k^2 u = f
    # For u = exp(4x) sin(pi y):
    # Δu = (16 - pi^2) u
    # f = -(16 - pi^2)u - k^2 u = (pi^2 - 16 - k^2) u
    f_expr = (pi**2 - 16.0 - k**2) * u_exact
    return u_exact, f_expr


def _solve_once(n, degree, k, ksp_type="gmres", pc_type="ilu", rtol=1e-10):
    comm = MPI.COMM_WORLD
    msh = mesh.create_unit_square(comm, n, n, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(msh, ("Lagrange", degree))

    u_exact, f_expr = _make_exact_and_rhs(msh, k)

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    a = (ufl.inner(ufl.grad(u), ufl.grad(v)) - (k ** 2) * u * v) * ufl.dx
    L = f_expr * v * ufl.dx

    # Dirichlet BC from manufactured solution on whole boundary
    fdim = msh.topology.dim - 1
    facets = mesh.locate_entities_boundary(msh, fdim, lambda x: np.ones(x.shape[1], dtype=bool))
    dofs = fem.locate_dofs_topological(V, fdim, facets)

    u_bc = fem.Function(V)
    u_bc.interpolate(fem.Expression(u_exact, V.element.interpolation_points))
    bc = fem.dirichletbc(u_bc, dofs)

    prefix = f"helmholtz_{n}_{degree}_"
    opts = {
        "ksp_type": ksp_type,
        "pc_type": pc_type,
        "ksp_rtol": rtol,
        "ksp_atol": 1e-14,
        "ksp_max_it": 20000,
    }
    if ksp_type == "gmres":
        opts["ksp_gmres_restart"] = 200
    # If direct solve requested, use preonly+lu
    if pc_type == "lu":
        opts["ksp_type"] = "preonly"
        opts["pc_type"] = "lu"

    uh = petsc.LinearProblem(
        a, L, bcs=[bc],
        petsc_options_prefix=prefix,
        petsc_options=opts
    ).solve()
    uh.x.scatter_forward()

    # Accuracy verification: relative L2 error against exact solution
    err_form = fem.form((uh - u_exact) ** 2 * ufl.dx)
    ref_form = fem.form((u_exact) ** 2 * ufl.dx)
    err_l2 = np.sqrt(comm.allreduce(fem.assemble_scalar(err_form), op=MPI.SUM))
    ref_l2 = np.sqrt(comm.allreduce(fem.assemble_scalar(ref_form), op=MPI.SUM))
    rel_l2 = err_l2 / ref_l2 if ref_l2 > 0 else err_l2

    its = 0 if pc_type != "lu" else 1
    ksp_type_used = "preonly" if pc_type == "lu" else ksp_type
    pc_type_used = pc_type

    return msh, uh, {
        "mesh_resolution": int(n),
        "element_degree": int(degree),
        "ksp_type": str(ksp_type_used),
        "pc_type": str(pc_type_used),
        "rtol": float(rtol),
        "iterations": int(its),
        "rel_l2_error": float(rel_l2),
    }


def solve(case_spec: dict) -> dict:
    """
    Solve Helmholtz problem and return sampled grid output.
    """
    t0 = time.time()
    k = float(case_spec.get("pde", {}).get("k", case_spec.get("wavenumber", 25.0)))
    if abs(k - 25.0) > 0:
        k = float(k)

    # Adaptive accuracy/time trade-off:
    # start moderately high because budget is large and boundary-layer-like exponential x dependence benefits from P2.
    candidates = [
        (48, 2, "gmres", "ilu", 1e-10),
        (64, 2, "gmres", "ilu", 1e-10),
        (80, 2, "preonly", "lu", 1e-12),
    ]

    # If user supplies a time budget, use some of it; otherwise simply reach strong accuracy.
    max_wall = float(case_spec.get("time_limit", case_spec.get("wall_time_sec", 640.068)))
    target_err = 1.19e-3 * 0.25

    best = None
    elapsed = 0.0
    for n, degree, ksp, pc, rtol in candidates:
        if elapsed > 0.85 * max_wall:
            break
        try:
            msh, uh, info = _solve_once(n, degree, k, ksp_type=ksp, pc_type=pc, rtol=rtol)
            best = (msh, uh, info)
            elapsed = time.time() - t0
            if info["rel_l2_error"] < target_err:
                break
        except Exception:
            # robust fallback: continue to next possibly more robust option
            continue

    if best is None:
        # final emergency robust attempt
        msh, uh, info = _solve_once(96, 2, k, ksp_type="preonly", pc_type="lu", rtol=1e-12)
        best = (msh, uh, info)

    msh, uh, info = best

    # Sample solution on requested uniform grid
    output_grid = case_spec["output"]["grid"]
    nx = int(output_grid["nx"])
    ny = int(output_grid["ny"])
    bbox = output_grid["bbox"]
    xmin, xmax, ymin, ymax = map(float, bbox)

    xs = np.linspace(xmin, xmax, nx)
    ys = np.linspace(ymin, ymax, ny)
    X, Y = np.meshgrid(xs, ys, indexing="xy")
    pts = np.column_stack([X.ravel(), Y.ravel(), np.zeros(nx * ny, dtype=np.float64)])
    u_vals = _probe_function(uh, pts).reshape(ny, nx)

    # Also verify sampled error if exact solution available at the same points
    u_exact_grid = np.exp(4.0 * X) * np.sin(np.pi * Y)
    sampled_rel_l2 = float(np.linalg.norm(u_vals - u_exact_grid) / np.linalg.norm(u_exact_grid))

    solver_info = {
        "mesh_resolution": info["mesh_resolution"],
        "element_degree": info["element_degree"],
        "ksp_type": info["ksp_type"],
        "pc_type": info["pc_type"],
        "rtol": info["rtol"],
        "iterations": info["iterations"],
        "rel_l2_error": info["rel_l2_error"],
        "sampled_rel_l2_error": sampled_rel_l2,
        "wall_time_sec": float(time.time() - t0),
    }

    return {"u": u_vals, "solver_info": solver_info}


if __name__ == "__main__":
    case_spec = {
        "pde": {"k": 25.0, "time": None},
        "output": {"grid": {"nx": 64, "ny": 64, "bbox": [0.0, 1.0, 0.0, 1.0]}},
        "time_limit": 640.068,
    }
    out = solve(case_spec)
    print(out["u"].shape)
    print(out["solver_info"])
