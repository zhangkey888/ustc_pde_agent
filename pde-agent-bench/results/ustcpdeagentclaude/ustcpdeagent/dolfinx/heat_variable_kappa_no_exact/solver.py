import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc


def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD

    # Time params
    t0 = 0.0
    t_end = 0.1
    dt_val = 0.0025  # finer than suggested 0.02
    try:
        tp = case_spec.get("pde", {}).get("time", {}) or case_spec.get("time", {})
        t0 = float(tp.get("t0", t0))
        t_end = float(tp.get("t_end", t_end))
    except Exception:
        pass

    n_steps = int(round((t_end - t0) / dt_val))
    if n_steps < 1:
        n_steps = 1
    dt_val = (t_end - t0) / n_steps

    # Mesh
    N = 128
    domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", 2))

    x = ufl.SpatialCoordinate(domain)
    kappa = 1.0 + 0.6 * ufl.sin(2 * ufl.pi * x[0]) * ufl.sin(2 * ufl.pi * x[1])
    f_expr = 1.0 + ufl.sin(2 * ufl.pi * x[0]) * ufl.cos(2 * ufl.pi * x[1])

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    # Initial condition
    u_n = fem.Function(V)
    u_n.interpolate(lambda xx: np.sin(np.pi * xx[0]) * np.sin(np.pi * xx[1]))

    # store initial on grid later
    u_initial_func = fem.Function(V)
    u_initial_func.x.array[:] = u_n.x.array[:]

    dt_c = fem.Constant(domain, PETSc.ScalarType(dt_val))

    # Backward Euler: (u - u_n)/dt - div(kappa grad u) = f
    a = u * v * ufl.dx + dt_c * kappa * ufl.dot(ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = u_n * v * ufl.dx + dt_c * f_expr * v * ufl.dx

    # BC: u=0 on boundary
    fdim = domain.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda xx: np.ones(xx.shape[1], dtype=bool)
    )
    dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    u_bc = fem.Function(V)
    u_bc.x.array[:] = 0.0
    bc = fem.dirichletbc(u_bc, dofs)

    problem = petsc.LinearProblem(
        a, L, bcs=[bc], u=fem.Function(V),
        petsc_options={"ksp_type": "cg", "pc_type": "hypre", "ksp_rtol": 1e-10},
        petsc_options_prefix="heat_",
    )

    total_iters = 0
    u_sol = None
    for step in range(n_steps):
        u_sol = problem.solve()
        try:
            total_iters += problem.solver.getIterationNumber()
        except Exception:
            pass
        u_n.x.array[:] = u_sol.x.array[:]

    # Sample on grid
    grid = case_spec["output"]["grid"]
    nx = int(grid["nx"])
    ny = int(grid["ny"])
    bbox = grid["bbox"]
    xs = np.linspace(bbox[0], bbox[1], nx)
    ys = np.linspace(bbox[2], bbox[3], ny)
    XX, YY = np.meshgrid(xs, ys)
    pts = np.c_[XX.ravel(), YY.ravel(), np.zeros(nx * ny)]

    tree = geometry.bb_tree(domain, domain.topology.dim)
    cand = geometry.compute_collisions_points(tree, pts)
    coll = geometry.compute_colliding_cells(domain, cand, pts)

    points_on_proc = []
    cells_on_proc = []
    idx_map = []
    for i in range(pts.shape[0]):
        links = coll.links(i)
        if len(links) > 0:
            points_on_proc.append(pts[i])
            cells_on_proc.append(links[0])
            idx_map.append(i)

    u_grid = np.zeros(nx * ny)
    u_init_grid = np.zeros(nx * ny)
    if points_on_proc:
        pa = np.array(points_on_proc)
        ca = np.array(cells_on_proc, dtype=np.int32)
        vals = u_n.eval(pa, ca).flatten()
        vals_i = u_initial_func.eval(pa, ca).flatten()
        for k, gi in enumerate(idx_map):
            u_grid[gi] = vals[k]
            u_init_grid[gi] = vals_i[k]

    u_grid = u_grid.reshape(ny, nx)
    u_init_grid = u_init_grid.reshape(ny, nx)

    return {
        "u": u_grid,
        "u_initial": u_init_grid,
        "solver_info": {
            "mesh_resolution": N,
            "element_degree": 2,
            "ksp_type": "cg",
            "pc_type": "hypre",
            "rtol": 1e-10,
            "iterations": int(total_iters),
            "dt": float(dt_val),
            "n_steps": int(n_steps),
            "time_scheme": "backward_euler",
        },
    }


if __name__ == "__main__":
    import time
    spec = {
        "pde": {"time": {"t0": 0.0, "t_end": 0.1}},
        "output": {"grid": {"nx": 64, "ny": 64, "bbox": [0, 1, 0, 1]}},
    }
    t0 = time.time()
    r = solve(spec)
    print("time:", time.time() - t0)
    print("shape:", r["u"].shape, "min/max:", r["u"].min(), r["u"].max())
    print("info:", r["solver_info"])
