import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc


def _eval_on_grid(u_sol, msh, nx, ny, bbox):
    xs = np.linspace(bbox[0], bbox[1], nx)
    ys = np.linspace(bbox[2], bbox[3], ny)
    XX, YY = np.meshgrid(xs, ys)
    pts = np.column_stack([XX.ravel(), YY.ravel(), np.zeros(nx * ny)])

    tree = geometry.bb_tree(msh, msh.topology.dim)
    cell_candidates = geometry.compute_collisions_points(tree, pts)
    colliding = geometry.compute_colliding_cells(msh, cell_candidates, pts)

    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    for i in range(pts.shape[0]):
        links = colliding.links(i)
        if len(links) > 0:
            points_on_proc.append(pts[i])
            cells_on_proc.append(links[0])
            eval_map.append(i)

    vals = np.zeros(nx * ny)
    if len(points_on_proc) > 0:
        res = u_sol.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        vals[eval_map] = res.flatten()
    return vals.reshape(ny, nx)


def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD

    out = case_spec["output"]["grid"]
    nx_out = out["nx"]
    ny_out = out["ny"]
    bbox = out["bbox"]

    pde = case_spec.get("pde", {})
    time_info = pde.get("time", None)
    params = case_spec.get("parameters", {}) if isinstance(case_spec.get("parameters", {}), dict) else {}

    t0 = 0.0
    t_end = 0.35
    dt_val = 0.01
    if time_info is not None:
        t0 = float(time_info.get("t0", 0.0))
        t_end = float(time_info.get("t_end", 0.35))
        dt_val = float(time_info.get("dt", 0.01))

    eps = float(params.get("epsilon", pde.get("epsilon", 0.01)))
    rho = float(params.get("reaction_rho", pde.get("reaction_rho", 1.0)))

    mesh_res = 64
    msh = mesh.create_unit_square(comm, mesh_res, mesh_res, cell_type=mesh.CellType.triangle)

    degree = 2
    V = fem.functionspace(msh, ("Lagrange", degree))

    dt_use = min(dt_val, 0.0025)
    n_steps = int(round((t_end - t0) / dt_use))
    dt_use = (t_end - t0) / n_steps

    x = ufl.SpatialCoordinate(msh)
    f_expr = (4.0 * ufl.exp(-200.0 * ((x[0] - 0.4)**2 + (x[1] - 0.6)**2))
              - 2.0 * ufl.exp(-200.0 * ((x[0] - 0.65)**2 + (x[1] - 0.35)**2)))

    u_n = fem.Function(V)
    u_n.interpolate(lambda X: 0.4 + 0.1 * np.sin(np.pi * X[0]) * np.sin(np.pi * X[1]))

    u_initial_grid = _eval_on_grid(u_n, msh, nx_out, ny_out, bbox)

    fdim = msh.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(
        msh, fdim, lambda X: np.ones(X.shape[1], dtype=bool)
    )
    bdofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    u_bc = fem.Function(V)
    u_bc.x.array[:] = 0.0
    bc = fem.dirichletbc(u_bc, bdofs)

    u = fem.Function(V)
    u.x.array[:] = u_n.x.array[:]
    v = ufl.TestFunction(V)

    dt_c = fem.Constant(msh, PETSc.ScalarType(dt_use))
    eps_c = fem.Constant(msh, PETSc.ScalarType(eps))
    rho_c = fem.Constant(msh, PETSc.ScalarType(rho))

    R_u = -rho_c * u * (1.0 - u)

    F = ((u - u_n) / dt_c) * v * ufl.dx \
        + eps_c * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx \
        + R_u * v * ufl.dx \
        - f_expr * v * ufl.dx

    J = ufl.derivative(F, u)

    petsc_options = {
        "snes_type": "newtonls",
        "snes_linesearch_type": "bt",
        "snes_rtol": 1e-8,
        "snes_atol": 1e-10,
        "snes_max_it": 25,
        "ksp_type": "gmres",
        "ksp_rtol": 1e-9,
        "pc_type": "hypre",
        "pc_hypre_type": "boomeramg",
    }

    problem = petsc.NonlinearProblem(
        F, u, bcs=[bc], J=J,
        petsc_options_prefix="rd_",
        petsc_options=petsc_options,
    )

    nl_iters_list = []
    total_lin_iters = 0

    t = t0
    for step in range(n_steps):
        u.x.scatter_forward()
        problem.solve()
        u.x.scatter_forward()

        try:
            its = problem.solver.getIterationNumber()
        except Exception:
            its = 0
        nl_iters_list.append(int(its))
        try:
            ksp = problem.solver.getKSP()
            total_lin_iters += int(ksp.getIterationNumber())
        except Exception:
            pass

        u_n.x.array[:] = u.x.array[:]
        t += dt_use

    u_grid = _eval_on_grid(u_n, msh, nx_out, ny_out, bbox)

    solver_info = {
        "mesh_resolution": mesh_res,
        "element_degree": degree,
        "ksp_type": "gmres",
        "pc_type": "hypre",
        "rtol": 1e-9,
        "iterations": int(total_lin_iters),
        "dt": float(dt_use),
        "n_steps": int(n_steps),
        "time_scheme": "backward_euler",
        "nonlinear_iterations": nl_iters_list,
    }

    return {
        "u": u_grid,
        "u_initial": u_initial_grid,
        "solver_info": solver_info,
    }


if __name__ == "__main__":
    import time
    case_spec = {
        "output": {"grid": {"nx": 64, "ny": 64, "bbox": [0.0, 1.0, 0.0, 1.0]}},
        "pde": {"time": {"t0": 0.0, "t_end": 0.35, "dt": 0.01, "scheme": "backward_euler"}},
        "parameters": {"epsilon": 0.01, "reaction_rho": 1.0},
    }
    t0 = time.time()
    res = solve(case_spec)
    t1 = time.time()
    print(f"Time: {t1 - t0:.2f}s")
    print(f"u shape: {res['u'].shape}")
    print(f"u range: [{res['u'].min():.4f}, {res['u'].max():.4f}]")
    print(f"nonlinear_iterations: {res['solver_info']['nonlinear_iterations']}")
    print(f"total lin iters: {res['solver_info']['iterations']}")
