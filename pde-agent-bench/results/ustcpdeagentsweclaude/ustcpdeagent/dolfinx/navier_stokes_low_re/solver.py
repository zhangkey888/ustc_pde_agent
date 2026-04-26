import time
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc

from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from basix.ufl import element as basix_element, mixed_element as basix_mixed_element

ScalarType = PETSc.ScalarType


def _u_exact_np(x):
    px = np.pi * x[0]
    py = np.pi * x[1]
    return np.vstack(
        (
            np.pi * np.cos(py) * np.sin(px),
            -np.pi * np.cos(px) * np.sin(py),
        )
    )


def _u_mag_exact_np(x, y):
    ux = np.pi * np.cos(np.pi * y) * np.sin(np.pi * x)
    uy = -np.pi * np.cos(np.pi * x) * np.sin(np.pi * y)
    return np.sqrt(ux * ux + uy * uy)


def _default_mesh_resolution(case_spec):
    solver_spec = case_spec.get("solver", {})
    if "mesh_resolution" in solver_spec:
        return int(solver_spec["mesh_resolution"])
    grid = case_spec.get("output", {}).get("grid", {})
    nx = int(grid.get("nx", 65))
    ny = int(grid.get("ny", 65))
    target = max(24, min(40, max(nx, ny) // 2))
    return target


def _build_problem(n, newton_rtol=1.0e-10):
    comm = MPI.COMM_WORLD
    msh = mesh.create_unit_square(comm, n, n, cell_type=mesh.CellType.triangle)
    cell = msh.topology.cell_name()
    gdim = msh.geometry.dim

    vel_el = basix_element("Lagrange", cell, 2, shape=(gdim,))
    pre_el = basix_element("Lagrange", cell, 1)
    W = fem.functionspace(msh, basix_mixed_element([vel_el, pre_el]))
    V, _ = W.sub(0).collapse()
    Q, _ = W.sub(1).collapse()

    x = ufl.SpatialCoordinate(msh)
    pi = ufl.pi
    nu = ScalarType(1.0)

    u_ex = ufl.as_vector(
        (
            pi * ufl.cos(pi * x[1]) * ufl.sin(pi * x[0]),
            -pi * ufl.cos(pi * x[0]) * ufl.sin(pi * x[1]),
        )
    )
    p_ex = ufl.cos(pi * x[0]) * ufl.cos(pi * x[1])

    conv = ufl.grad(u_ex) * u_ex
    visc = -nu * ufl.div(ufl.grad(u_ex))
    gp = ufl.grad(p_ex)
    force = ufl.as_vector((conv[0] + visc[0] + gp[0], conv[1] + visc[1] + gp[1]))

    w = fem.Function(W)
    w.name = "w"
    u, p = ufl.split(w)
    v, q = ufl.TestFunctions(W)

    F = (
        nu * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
        + ufl.inner(ufl.grad(u) * u, v) * ufl.dx
        - p * ufl.div(v) * ufl.dx
        + ufl.div(u) * q * ufl.dx
        - ufl.inner(force, v) * ufl.dx
    )
    J = ufl.derivative(F, w)

    fdim = msh.topology.dim - 1
    facets = mesh.locate_entities_boundary(
        msh, fdim, lambda xx: np.ones(xx.shape[1], dtype=bool)
    )

    u_bc = fem.Function(V)
    u_bc.interpolate(_u_exact_np)
    dofs_u = fem.locate_dofs_topological((W.sub(0), V), fdim, facets)
    bc_u = fem.dirichletbc(u_bc, dofs_u, W.sub(0))

    p_dofs = fem.locate_dofs_geometrical(
        (W.sub(1), Q), lambda xx: np.isclose(xx[0], 0.0) & np.isclose(xx[1], 0.0)
    )
    bcs = [bc_u]
    if len(p_dofs) > 0:
        p0 = fem.Function(Q)
        p0.x.array[:] = 0.0
        bcs.append(fem.dirichletbc(p0, p_dofs, W.sub(1)))

    w.x.array[:] = 0.0

    opts = {
        "snes_type": "newtonls",
        "snes_linesearch_type": "bt",
        "snes_rtol": float(newton_rtol),
        "snes_atol": 1.0e-12,
        "snes_max_it": 25,
        "ksp_type": "preonly",
        "pc_type": "lu",
        "pc_factor_mat_solver_type": "mumps",
    }

    problem = petsc.NonlinearProblem(
        F, w, bcs=bcs, J=J, petsc_options_prefix="ns_", petsc_options=opts
    )
    return msh, W, V, Q, w, problem, opts


def _sample_velocity_magnitude(u_fun, msh, grid):
    nx = int(grid["nx"])
    ny = int(grid["ny"])
    xmin, xmax, ymin, ymax = grid["bbox"]

    xs = np.linspace(xmin, xmax, nx)
    ys = np.linspace(ymin, ymax, ny)
    XX, YY = np.meshgrid(xs, ys, indexing="xy")
    pts = np.column_stack([XX.ravel(), YY.ravel(), np.zeros(nx * ny, dtype=np.float64)])

    tree = geometry.bb_tree(msh, msh.topology.dim)
    cell_candidates = geometry.compute_collisions_points(tree, pts)
    colliding = geometry.compute_colliding_cells(msh, cell_candidates, pts)

    local_idx = []
    local_pts = []
    local_cells = []
    for i in range(pts.shape[0]):
        links = colliding.links(i)
        if len(links) > 0:
            local_idx.append(i)
            local_pts.append(pts[i])
            local_cells.append(links[0])

    local_values = np.full(pts.shape[0], np.nan, dtype=np.float64)
    if local_pts:
        vals = u_fun.eval(
            np.array(local_pts, dtype=np.float64),
            np.array(local_cells, dtype=np.int32),
        )
        mags = np.linalg.norm(vals, axis=1)
        for k, idx in enumerate(local_idx):
            local_values[idx] = mags[k]

    gathered = msh.comm.gather(local_values, root=0)
    if msh.comm.rank == 0:
        merged = np.full(pts.shape[0], np.nan, dtype=np.float64)
        for arr in gathered:
            mask = np.isfinite(arr)
            merged[mask] = arr[mask]
        if np.isnan(merged).any():
            raise RuntimeError("Failed to evaluate solution at some grid points.")
        return merged.reshape((ny, nx))
    return None


def _extract_velocity(w, V):
    u_h = fem.Function(V)
    u_h.interpolate(w.sub(0))
    u_h.x.scatter_forward()
    return u_h


def _compute_grid_error(u_grid, grid):
    nx = int(grid["nx"])
    ny = int(grid["ny"])
    xmin, xmax, ymin, ymax = grid["bbox"]
    xs = np.linspace(xmin, xmax, nx)
    ys = np.linspace(ymin, ymax, ny)
    XX, YY = np.meshgrid(xs, ys, indexing="xy")
    exact = _u_mag_exact_np(XX, YY)
    return float(np.sqrt(np.mean((u_grid - exact) ** 2)))


def solve(case_spec: dict) -> dict:
    n = _default_mesh_resolution(case_spec)
    newton_rtol = float(case_spec.get("solver", {}).get("newton_rtol", 1.0e-10))
    msh, W, V, Q, w, problem, opts = _build_problem(n, newton_rtol=newton_rtol)

    t0 = time.perf_counter()
    problem.solve()
    w.x.scatter_forward()
    elapsed = time.perf_counter() - t0

    u_h = _extract_velocity(w, V)
    grid = case_spec["output"]["grid"]
    u_grid = _sample_velocity_magnitude(u_h, msh, grid)

    nonlinear_iterations = -1
    linear_iterations = 0
    try:
        nonlinear_iterations = int(problem.solver.getIterationNumber())
    except Exception:
        pass
    try:
        linear_iterations = int(problem.solver.getKSP().getTotalIterations())
    except Exception:
        pass

    result = None
    if msh.comm.rank == 0:
        solver_info = {
            "mesh_resolution": n,
            "element_degree": 2,
            "ksp_type": "preonly",
            "pc_type": "lu",
            "rtol": float(newton_rtol),
            "iterations": linear_iterations,
            "nonlinear_iterations": [nonlinear_iterations],
            "verification_l2_grid_error": _compute_grid_error(u_grid, grid),
        }
        result = {"u": u_grid, "solver_info": solver_info}
    return result


def _self_test():
    case_spec = {
        "pde": {"time": None},
        "output": {"grid": {"nx": 65, "ny": 65, "bbox": [0.0, 1.0, 0.0, 1.0]}},
    }

    errs = []
    times = []
    for n in (24, 32):
        local_case = {
            "pde": case_spec["pde"],
            "solver": {"mesh_resolution": n, "newton_rtol": 1.0e-10},
            "output": case_spec["output"],
        }
        t0 = time.perf_counter()
        out = solve(local_case)
        wall = time.perf_counter() - t0
        if MPI.COMM_WORLD.rank == 0:
            errs.append(out["solver_info"]["verification_l2_grid_error"])
            times.append(wall)

    if MPI.COMM_WORLD.rank == 0:
        print(f"MESH_ERRORS: {errs}")
        print(f"MESH_TIMES: {times}")
        if len(errs) == 2:
            print(f"ERROR_REDUCTION_FACTOR: {errs[0] / errs[1]:.6e}")
        final_out = solve(case_spec)
        print(f"FINAL_L2_ERROR: {final_out['solver_info']['verification_l2_grid_error']:.8e}")
        print(final_out["solver_info"])


if __name__ == "__main__":
    _self_test()
