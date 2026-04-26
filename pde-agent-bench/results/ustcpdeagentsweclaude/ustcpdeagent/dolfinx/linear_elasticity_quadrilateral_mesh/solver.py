import time
import math
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl


ScalarType = PETSc.ScalarType


def _make_exact_and_rhs(msh, E=1.0, nu=0.3):
    gdim = msh.geometry.dim
    x = ufl.SpatialCoordinate(msh)
    pi = ufl.pi

    u_exact = ufl.as_vector(
        [
            ufl.sin(2 * pi * x[0]) * ufl.cos(3 * pi * x[1]),
            ufl.sin(pi * x[0]) * ufl.sin(2 * pi * x[1]),
        ]
    )

    mu = E / (2.0 * (1.0 + nu))
    lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))

    def eps(u):
        return ufl.sym(ufl.grad(u))

    def sigma(u):
        return 2.0 * mu * eps(u) + lam * ufl.tr(eps(u)) * ufl.Identity(gdim)

    f = -ufl.div(sigma(u_exact))
    return u_exact, f, mu, lam


def _sample_on_grid(u_fun, msh, grid_spec):
    nx = int(grid_spec["nx"])
    ny = int(grid_spec["ny"])
    bbox = grid_spec["bbox"]
    xmin, xmax, ymin, ymax = map(float, bbox)

    xs = np.linspace(xmin, xmax, nx)
    ys = np.linspace(ymin, ymax, ny)
    XX, YY = np.meshgrid(xs, ys, indexing="xy")
    pts = np.column_stack(
        [XX.ravel(), YY.ravel(), np.zeros(nx * ny, dtype=np.float64)]
    )

    tree = geometry.bb_tree(msh, msh.topology.dim)
    cell_candidates = geometry.compute_collisions_points(tree, pts)
    colliding = geometry.compute_colliding_cells(msh, cell_candidates, pts)

    local_mag = np.full(nx * ny, np.nan, dtype=np.float64)
    points_on_proc = []
    cells_on_proc = []
    ids = []
    for i in range(pts.shape[0]):
        links = colliding.links(i)
        if len(links) > 0:
            points_on_proc.append(pts[i])
            cells_on_proc.append(links[0])
            ids.append(i)

    if points_on_proc:
        vals = u_fun.eval(np.array(points_on_proc, dtype=np.float64),
                          np.array(cells_on_proc, dtype=np.int32))
        mags = np.linalg.norm(vals, axis=1)
        local_mag[np.array(ids, dtype=np.int64)] = mags

    gathered = msh.comm.gather(local_mag, root=0)
    if msh.comm.rank == 0:
        out = np.full(nx * ny, np.nan, dtype=np.float64)
        for arr in gathered:
            mask = np.isnan(out) & (~np.isnan(arr))
            out[mask] = arr[mask]
        if np.isnan(out).any():
            # Domain points on boundary should still be evaluable; guard anyway
            out = np.nan_to_num(out, nan=0.0)
        out = out.reshape((ny, nx))
    else:
        out = None

    out = msh.comm.bcast(out, root=0)
    return out


def _solve_once(mesh_resolution, degree, ksp_type="cg", pc_type="hypre", rtol=1e-10):
    comm = MPI.COMM_WORLD
    msh = mesh.create_rectangle(
        comm,
        [np.array([0.0, 0.0]), np.array([1.0, 1.0])],
        [mesh_resolution, mesh_resolution],
        cell_type=mesh.CellType.quadrilateral,
    )

    gdim = msh.geometry.dim
    V = fem.functionspace(msh, ("Lagrange", degree, (gdim,)))

    u_ex, f, mu, lam = _make_exact_and_rhs(msh, E=1.0, nu=0.3)

    def eps(u):
        return ufl.sym(ufl.grad(u))

    def sigma(u):
        return 2.0 * mu * eps(u) + lam * ufl.tr(eps(u)) * ufl.Identity(gdim)

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    a = ufl.inner(sigma(u), eps(v)) * ufl.dx
    L = ufl.inner(f, v) * ufl.dx

    fdim = msh.topology.dim - 1
    bfacets = mesh.locate_entities_boundary(
        msh, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
    )
    bdofs = fem.locate_dofs_topological(V, fdim, bfacets)

    u_bc = fem.Function(V)
    expr = fem.Expression(u_ex, V.element.interpolation_points)
    u_bc.interpolate(expr)
    bc = fem.dirichletbc(u_bc, bdofs)

    problem = petsc.LinearProblem(
        a,
        L,
        bcs=[bc],
        petsc_options_prefix=f"elas_{mesh_resolution}_{degree}_",
        petsc_options={
            "ksp_type": ksp_type,
            "ksp_rtol": rtol,
            "pc_type": pc_type,
            "ksp_atol": 1e-14,
        },
    )
    uh = problem.solve()
    uh.x.scatter_forward()

    # Try to recover iteration count from embedded solver if available
    iterations = -1
    try:
        solver = problem.solver
        iterations = int(solver.getIterationNumber())
        ksp_type = solver.getType()
        pc_type = solver.getPC().getType()
    except Exception:
        pass

    err_L2 = fem.assemble_scalar(fem.form(ufl.inner(uh - u_ex, uh - u_ex) * ufl.dx))
    ref_L2 = fem.assemble_scalar(fem.form(ufl.inner(u_ex, u_ex) * ufl.dx))
    err_L2 = math.sqrt(comm.allreduce(err_L2, op=MPI.SUM))
    ref_L2 = math.sqrt(comm.allreduce(ref_L2, op=MPI.SUM))
    rel_L2 = err_L2 / ref_L2 if ref_L2 > 0 else err_L2

    return msh, uh, {
        "mesh_resolution": int(mesh_resolution),
        "element_degree": int(degree),
        "ksp_type": str(ksp_type),
        "pc_type": str(pc_type),
        "rtol": float(rtol),
        "iterations": int(max(iterations, 0)),
        "relative_l2_error": float(rel_L2),
    }


def solve(case_spec: dict) -> dict:
    t0 = time.perf_counter()
    comm = MPI.COMM_WORLD

    # Default adaptive choices tuned for target accuracy/time on this manufactured solution.
    # Since nu=0.3, locking is not severe; Q2 vector elements on quads are accurate and efficient.
    degree = 2
    mesh_resolution = 36
    ksp_type = "cg"
    pc_type = "hypre"
    rtol = 1e-10

    # First solve
    msh, uh, info = _solve_once(mesh_resolution, degree, ksp_type, pc_type, rtol)
    elapsed = time.perf_counter() - t0

    # Adaptive time-accuracy tradeoff: if well under budget, refine once proactively.
    # User time limit is ~18.67 s; keep a conservative margin.
    if elapsed < 7.0:
        try_res = 52
        try:
            msh2, uh2, info2 = _solve_once(try_res, degree, ksp_type, pc_type, rtol)
            elapsed2 = time.perf_counter() - t0
            if elapsed2 < 17.5 and info2["relative_l2_error"] <= info["relative_l2_error"]:
                msh, uh, info = msh2, uh2, info2
        except Exception:
            pass

    grid_spec = case_spec["output"]["grid"]
    u_grid = _sample_on_grid(uh, msh, grid_spec)

    return {
        "u": u_grid,
        "solver_info": {
            "mesh_resolution": info["mesh_resolution"],
            "element_degree": info["element_degree"],
            "ksp_type": info["ksp_type"],
            "pc_type": info["pc_type"],
            "rtol": info["rtol"],
            "iterations": info["iterations"],
            "relative_l2_error": info["relative_l2_error"],
            "wall_time_sec": float(time.perf_counter() - t0),
        },
    }


if __name__ == "__main__":
    case_spec = {
        "output": {
            "grid": {
                "nx": 64,
                "ny": 64,
                "bbox": [0.0, 1.0, 0.0, 1.0],
            }
        },
        "pde": {"time": None},
    }
    result = solve(case_spec)
    if MPI.COMM_WORLD.rank == 0:
        print(result["u"].shape)
        print(result["solver_info"])
