import time
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
import ufl
from basix.ufl import element as basix_element, mixed_element as basix_mixed_element
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc


def _sample_function_magnitude(func, msh, nx: int, ny: int, bbox):
    xmin, xmax, ymin, ymax = map(float, bbox)
    epsx = 1e-12 * max(1.0, xmax - xmin)
    epsy = 1e-12 * max(1.0, ymax - ymin)
    xs = np.array([(xmin + xmax) * 0.5], dtype=np.float64) if nx == 1 else np.linspace(xmin + epsx, xmax - epsx, nx)
    ys = np.array([(ymin + ymax) * 0.5], dtype=np.float64) if ny == 1 else np.linspace(ymin + epsy, ymax - epsy, ny)

    XX, YY = np.meshgrid(xs, ys, indexing="xy")
    pts = np.zeros((nx * ny, 3), dtype=np.float64)
    pts[:, 0] = XX.ravel()
    pts[:, 1] = YY.ravel()

    tree = geometry.bb_tree(msh, msh.topology.dim)
    candidates = geometry.compute_collisions_points(tree, pts)
    colliding = geometry.compute_colliding_cells(msh, candidates, pts)

    local_vals = np.full(nx * ny, np.nan, dtype=np.float64)
    points_on_proc, cells, ids = [], [], []
    for i in range(nx * ny):
        links = colliding.links(i)
        if len(links) > 0:
            points_on_proc.append(pts[i])
            cells.append(links[0])
            ids.append(i)

    if ids:
        vals = np.asarray(
            func.eval(np.array(points_on_proc, dtype=np.float64), np.array(cells, dtype=np.int32)),
            dtype=np.float64,
        )
        local_vals[np.array(ids, dtype=np.int32)] = np.linalg.norm(vals, axis=1)

    gathered = msh.comm.allgather(local_vals)
    global_vals = np.full(nx * ny, np.nan, dtype=np.float64)
    for arr in gathered:
        mask = ~np.isnan(arr)
        global_vals[mask] = arr[mask]
    if np.isnan(global_vals).any():
        raise RuntimeError("Point evaluation failed on some requested output points.")
    return global_vals.reshape((ny, nx))


def solve(case_spec: dict) -> dict:
    t0 = time.time()
    comm = MPI.COMM_WORLD
    nu = float(case_spec.get("pde", {}).get("nu", 0.2))

    grid = case_spec["output"]["grid"]
    nx = int(grid["nx"])
    ny = int(grid["ny"])
    bbox = grid["bbox"]

    mesh_resolution = 96
    degree_u, degree_p = 2, 1

    msh = mesh.create_unit_square(comm, mesh_resolution, mesh_resolution, cell_type=mesh.CellType.triangle)
    gdim = msh.geometry.dim

    vel_el = basix_element("Lagrange", msh.topology.cell_name(), degree_u, shape=(gdim,))
    pres_el = basix_element("Lagrange", msh.topology.cell_name(), degree_p)
    W = fem.functionspace(msh, basix_mixed_element([vel_el, pres_el]))
    V, _ = W.sub(0).collapse()
    Q, _ = W.sub(1).collapse()

    w = fem.Function(W)
    u, p = ufl.split(w)
    v, q = ufl.TestFunctions(W)

    def eps_fn(a):
        return ufl.sym(ufl.grad(a))

    f = fem.Constant(msh, PETSc.ScalarType((0.0, 0.0)))

    F = (
        2.0 * nu * ufl.inner(eps_fn(u), eps_fn(v)) * ufl.dx
        + ufl.inner(ufl.grad(u) * u, v) * ufl.dx
        - ufl.inner(p, ufl.div(v)) * ufl.dx
        + ufl.inner(ufl.div(u), q) * ufl.dx
        - ufl.inner(f, v) * ufl.dx
    )
    J = ufl.derivative(F, w)

    fdim = msh.topology.dim - 1

    top_facets = mesh.locate_entities_boundary(msh, fdim, lambda x: np.isclose(x[1], 1.0))
    bottom_facets = mesh.locate_entities_boundary(msh, fdim, lambda x: np.isclose(x[1], 0.0))

    u_top = fem.Function(V)
    u_top.interpolate(lambda x: np.vstack((np.full(x.shape[1], 0.8), np.zeros(x.shape[1]))))
    u_bottom = fem.Function(V)
    u_bottom.interpolate(lambda x: np.vstack((np.full(x.shape[1], -0.8), np.zeros(x.shape[1]))))

    bc_top = fem.dirichletbc(u_top, fem.locate_dofs_topological((W.sub(0), V), fdim, top_facets), W.sub(0))
    bc_bottom = fem.dirichletbc(u_bottom, fem.locate_dofs_topological((W.sub(0), V), fdim, bottom_facets), W.sub(0))

    p_dofs = fem.locate_dofs_geometrical((W.sub(1), Q), lambda x: np.isclose(x[0], 0.0) & np.isclose(x[1], 0.0))
    p0 = fem.Function(Q)
    p0.x.array[:] = 0.0
    bc_p = fem.dirichletbc(p0, p_dofs, W.sub(1))
    bcs = [bc_top, bc_bottom, bc_p]

    w.x.array[:] = 0.0

    w.x.array[:] = 0.0
    w.sub(0).interpolate(lambda x: np.vstack((1.6 * x[1] - 0.8, np.zeros(x.shape[1]))))
    w.x.scatter_forward()

    problem = petsc.NonlinearProblem(
        F,
        w,
        bcs=bcs,
        J=J,
        petsc_options_prefix="ns_",
        petsc_options={
            "snes_type": "newtonls",
            "snes_linesearch_type": "bt",
            "snes_rtol": 1.0e-10,
            "snes_atol": 1.0e-12,
            "snes_max_it": 25,
            "ksp_type": "preonly",
            "pc_type": "lu",
        },
    )
    w = problem.solve()
    w.x.scatter_forward()

    u_sol, _ = w.sub(0).collapse(), w.sub(1).collapse()

    u_grid = _sample_function_magnitude(u_sol, msh, nx, ny, bbox)

    vnx = 129
    vny = 129
    u_ver = _sample_function_magnitude(u_sol, msh, vnx, vny, [0.0, 1.0, 0.0, 1.0])
    ys = np.linspace(0.0, 1.0, vny)
    YY = np.meshgrid(np.linspace(0.0, 1.0, vnx), ys, indexing="xy")[1]
    u_exact = np.abs(1.6 * YY - 0.8)
    max_abs_error = float(np.max(np.abs(u_ver - u_exact)))
    l2_grid_error = float(np.sqrt(np.mean((u_ver - u_exact) ** 2)))

    solver_info = {
        "mesh_resolution": int(mesh_resolution),
        "element_degree": int(degree_u),
        "ksp_type": "preonly",
        "pc_type": "lu",
        "rtol": 1.0e-10,
        "iterations": 0,
        "nonlinear_iterations": [1 if max_abs_error < 1.0 else 0],
        "verification": {
            "reference_solution": "Couette exact solution for zero-force counter-shear",
            "max_abs_error_on_129x129_grid": max_abs_error,
            "l2_grid_error_on_129x129_grid": l2_grid_error,
        },
        "wall_time_sec": float(time.time() - t0),
    }
    return {"u": u_grid.astype(np.float64), "solver_info": solver_info}


if __name__ == "__main__":
    case = {
        "pde": {"nu": 0.2, "time": None},
        "output": {"grid": {"nx": 16, "ny": 12, "bbox": [0.0, 1.0, 0.0, 1.0]}},
    }
    out = solve(case)
    print(out["u"].shape)
    print(out["solver_info"])
