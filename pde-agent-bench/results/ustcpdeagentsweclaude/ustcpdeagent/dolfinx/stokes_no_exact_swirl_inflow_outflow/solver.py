import time
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
import ufl
from dolfinx import mesh, fem, geometry
from dolfinx.fem.petsc import LinearProblem

ScalarType = PETSc.ScalarType


def _make_velocity_space(msh, degree=2):
    return fem.functionspace(msh, ("Lagrange", degree, (msh.geometry.dim,)))


def _velocity_bc_function(V, kind):
    f = fem.Function(V)
    if kind == "inflow":
        f.interpolate(lambda x: np.vstack((np.sin(np.pi * x[1]), 0.2 * np.sin(2.0 * np.pi * x[1]))))
    elif kind == "zero":
        f.interpolate(lambda x: np.zeros((V.mesh.geometry.dim, x.shape[1]), dtype=np.float64))
    else:
        raise ValueError(kind)
    return f


def _build_bcs(msh, V):
    fdim = msh.topology.dim - 1
    left_facets = mesh.locate_entities_boundary(msh, fdim, lambda x: np.isclose(x[0], 0.0))
    bottom_facets = mesh.locate_entities_boundary(msh, fdim, lambda x: np.isclose(x[1], 0.0))
    top_facets = mesh.locate_entities_boundary(msh, fdim, lambda x: np.isclose(x[1], 1.0))

    inflow = _velocity_bc_function(V, "inflow")
    zero = _velocity_bc_function(V, "zero")

    left_dofs = fem.locate_dofs_topological(V, fdim, left_facets)
    bottom_dofs = fem.locate_dofs_topological(V, fdim, bottom_facets)
    top_dofs = fem.locate_dofs_topological(V, fdim, top_facets)

    return [
        fem.dirichletbc(inflow, left_dofs),
        fem.dirichletbc(zero, bottom_dofs),
        fem.dirichletbc(zero, top_dofs),
    ]


def _solve_once(mesh_resolution, degree=2, penalty=1.0e3, ksp_type="preonly", pc_type="lu", rtol=1.0e-10):
    msh = mesh.create_unit_square(MPI.COMM_WORLD, mesh_resolution, mesh_resolution, cell_type=mesh.CellType.triangle)
    V = _make_velocity_space(msh, degree=degree)
    bcs = _build_bcs(msh, V)

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    nu = ScalarType(0.5)
    f = fem.Constant(msh, np.array([0.0, 0.0], dtype=np.float64))

    a = (
        2.0 * nu * ufl.inner(ufl.sym(ufl.grad(u)), ufl.sym(ufl.grad(v))) * ufl.dx
        + penalty * ufl.div(u) * ufl.div(v) * ufl.dx
    )
    L = ufl.inner(f, v) * ufl.dx

    problem = LinearProblem(
        a,
        L,
        bcs=bcs,
        petsc_options_prefix=f"penalty_{mesh_resolution}_",
        petsc_options={"ksp_type": ksp_type, "pc_type": pc_type},
    )
    uh = problem.solve()
    uh.x.scatter_forward()

    its = 0
    try:
        its = int(problem.solver.getIterationNumber())
    except Exception:
        pass

    return msh, uh, its, {
        "mesh_resolution": mesh_resolution,
        "element_degree": degree,
        "ksp_type": ksp_type,
        "pc_type": pc_type,
        "rtol": float(rtol),
        "penalty": float(penalty),
    }


def _sample_vector_function(u_fun, nx, ny, bbox):
    msh = u_fun.function_space.mesh
    xs = np.linspace(bbox[0], bbox[1], nx)
    ys = np.linspace(bbox[2], bbox[3], ny)
    XX, YY = np.meshgrid(xs, ys, indexing="xy")
    pts = np.vstack([XX.ravel(), YY.ravel(), np.zeros(nx * ny, dtype=np.float64)])

    tree = geometry.bb_tree(msh, msh.topology.dim)
    candidates = geometry.compute_collisions_points(tree, pts.T)
    colliding = geometry.compute_colliding_cells(msh, candidates, pts.T)

    values = np.full((pts.shape[1], msh.geometry.dim), np.nan, dtype=np.float64)
    points_on_proc = []
    cells = []
    ids = []

    for i in range(pts.shape[1]):
        links = colliding.links(i)
        if len(links) > 0:
            points_on_proc.append(pts.T[i])
            cells.append(links[0])
            ids.append(i)

    if points_on_proc:
        vals = u_fun.eval(np.array(points_on_proc, dtype=np.float64), np.array(cells, dtype=np.int32))
        values[np.array(ids, dtype=np.int32)] = vals

    if msh.comm.size > 1:
        recv = np.empty_like(values)
        msh.comm.Allreduce(values, recv, op=MPI.SUM)
        values = recv

    mag = np.linalg.norm(np.nan_to_num(values, nan=0.0), axis=1).reshape(ny, nx)
    return mag


def _divergence_indicator(uh):
    msh = uh.function_space.mesh
    DG0 = fem.functionspace(msh, ("DG", 0))
    expr = fem.Expression(ufl.div(uh) * ufl.div(uh), DG0.element.interpolation_points)
    divsq = fem.Function(DG0)
    divsq.interpolate(expr)
    local = np.sum(divsq.x.array) if divsq.x.array.size else 0.0
    return float(msh.comm.allreduce(local, op=MPI.SUM))


def _accuracy_verification(grid):
    _, u1, _, _ = _solve_once(24)
    g1 = _sample_vector_function(u1, grid["nx"], grid["ny"], grid["bbox"])
    _, u2, _, _ = _solve_once(48)
    g2 = _sample_vector_function(u2, grid["nx"], grid["ny"], grid["bbox"])
    rel = float(np.linalg.norm(g2 - g1) / max(np.linalg.norm(g2), 1e-14))
    return {"mesh_convergence_relative_diff": rel}


def solve(case_spec: dict) -> dict:
    grid = case_spec["output"]["grid"]

    probe_resolutions = [32, 48, 64, 96]
    selected = 64
    elapsed = 0.0
    for n in probe_resolutions:
        t0 = time.time()
        _solve_once(n)
        elapsed += time.time() - t0
        selected = n
        if elapsed > 60.0:
            break

    msh, uh, its, info = _solve_once(selected)
    u_grid = _sample_vector_function(uh, int(grid["nx"]), int(grid["ny"]), grid["bbox"])
    verification = _accuracy_verification(grid)
    verification["divergence_indicator"] = _divergence_indicator(uh)

    solver_info = {
        "mesh_resolution": int(info["mesh_resolution"]),
        "element_degree": int(info["element_degree"]),
        "ksp_type": str(info["ksp_type"]),
        "pc_type": str(info["pc_type"]),
        "rtol": float(info["rtol"]),
        "iterations": int(its),
        "verification": verification,
    }
    return {"u": u_grid, "solver_info": solver_info}
