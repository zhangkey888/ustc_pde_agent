import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc


def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD

    grid = case_spec["output"]["grid"]
    nx_out = grid["nx"]
    ny_out = grid["ny"]
    bbox = grid["bbox"]

    # Time parameters
    t0 = 0.0
    t_end = 0.2
    pde = case_spec.get("pde", {})
    tparams = pde.get("time", {})
    if tparams:
        t0 = tparams.get("t0", t0)
        t_end = tparams.get("t_end", t_end)

    dt_val = 0.005

    # Mesh
    N = 48
    degree = 2

    msh = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(msh, ("Lagrange", degree))

    x = ufl.SpatialCoordinate(msh)
    t_const = fem.Constant(msh, PETSc.ScalarType(t0))

    # Manufactured exact: u = exp(-t)*0.25*sin(2*pi*x)*sin(pi*y)
    u_exact_expr = ufl.exp(-t_const) * 0.25 * ufl.sin(2*ufl.pi*x[0]) * ufl.sin(ufl.pi*x[1])

    epsilon = 1.0
    # R(u) = u^3 - u  (Allen-Cahn)

    dudt_exact = -u_exact_expr  # d/dt exp(-t) = -exp(-t)
    lap_u = ufl.div(ufl.grad(u_exact_expr))
    f_expr = dudt_exact - epsilon * lap_u + (u_exact_expr**3 - u_exact_expr)

    u_n = fem.Function(V)
    u_h = fem.Function(V)
    v = ufl.TestFunction(V)

    def u0_np(xa):
        return 0.25 * np.sin(2*np.pi*xa[0]) * np.sin(np.pi*xa[1])
    u_n.interpolate(u0_np)
    u_h.interpolate(u0_np)

    u_bc = fem.Function(V)
    fdim = msh.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(
        msh, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
    )
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)

    def set_bc(t_value):
        u_bc.interpolate(
            lambda xa, tv=t_value: np.exp(-tv) * 0.25 * np.sin(2*np.pi*xa[0]) * np.sin(np.pi*xa[1])
        )

    set_bc(t0)
    bc = fem.dirichletbc(u_bc, boundary_dofs)

    dt = fem.Constant(msh, PETSc.ScalarType(dt_val))

    F = ((u_h - u_n)/dt) * v * ufl.dx \
        + epsilon * ufl.inner(ufl.grad(u_h), ufl.grad(v)) * ufl.dx \
        + (u_h**3 - u_h) * v * ufl.dx \
        - f_expr * v * ufl.dx
    J = ufl.derivative(F, u_h)

    petsc_options = {
        "snes_type": "newtonls",
        "snes_rtol": 1e-10,
        "snes_atol": 1e-12,
        "snes_max_it": 25,
        "ksp_type": "preonly",
        "pc_type": "lu",
    }

    problem = petsc.NonlinearProblem(
        F, u_h, bcs=[bc], J=J,
        petsc_options_prefix="rd_",
        petsc_options=petsc_options,
    )

    n_steps = int(round((t_end - t0) / dt_val))
    dt_val = (t_end - t0) / n_steps  # adjust
    dt.value = dt_val

    newton_iters_list = []
    total_lin_iters = 0

    t_current = t0
    for step in range(n_steps):
        t_current += dt_val
        t_const.value = t_current
        set_bc(t_current)

        problem.solve()
        try:
            snes = problem.solver
            n_it = snes.getIterationNumber()
            k_it = snes.getLinearSolveIterations()
            newton_iters_list.append(int(n_it))
            total_lin_iters += int(k_it)
        except Exception:
            newton_iters_list.append(0)

        u_h.x.scatter_forward()
        u_n.x.array[:] = u_h.x.array[:]

    # Sample on output grid
    xs = np.linspace(bbox[0], bbox[1], nx_out)
    ys = np.linspace(bbox[2], bbox[3], ny_out)
    XX, YY = np.meshgrid(xs, ys)
    pts = np.column_stack([XX.ravel(), YY.ravel(), np.zeros(nx_out * ny_out)])

    tree = geometry.bb_tree(msh, msh.topology.dim)
    cand = geometry.compute_collisions_points(tree, pts)
    coll = geometry.compute_colliding_cells(msh, cand, pts)

    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    for i in range(pts.shape[0]):
        links = coll.links(i)
        if len(links) > 0:
            points_on_proc.append(pts[i])
            cells_on_proc.append(links[0])
            eval_map.append(i)

    u_values = np.full(pts.shape[0], np.nan)
    if len(points_on_proc) > 0:
        vals = u_h.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u_values[eval_map] = vals.flatten()

    u_grid = u_values.reshape(ny_out, nx_out)

    u_init_grid = 0.25 * np.sin(2*np.pi*XX) * np.sin(np.pi*YY)

    # Debug: check error vs exact
    u_ex = np.exp(-t_end) * 0.25 * np.sin(2*np.pi*XX) * np.sin(np.pi*YY)
    err = np.sqrt(np.mean((u_grid - u_ex)**2))
    print(f"[solver] L2 err vs exact: {err:.3e}, n_steps={n_steps}, dt={dt_val}")

    solver_info = {
        "mesh_resolution": N,
        "element_degree": degree,
        "ksp_type": "preonly",
        "pc_type": "lu",
        "rtol": 1e-10,
        "iterations": int(total_lin_iters),
        "dt": dt_val,
        "n_steps": n_steps,
        "time_scheme": "backward_euler",
        "nonlinear_iterations": newton_iters_list,
    }

    return {
        "u": u_grid,
        "u_initial": u_init_grid,
        "solver_info": solver_info,
    }


if __name__ == "__main__":
    import time
    case_spec = {
        "output": {"grid": {"nx": 64, "ny": 64, "bbox": [0.0, 1.0, 0.0, 1.0]}},
        "pde": {"time": {"t0": 0.0, "t_end": 0.2, "dt": 0.005}},
    }
    t0 = time.time()
    res = solve(case_spec)
    t1 = time.time()
    print(f"Wall time: {t1-t0:.2f}s, u shape: {res['u'].shape}")
    print(f"Newton iters sample: {res['solver_info']['nonlinear_iterations'][:5]}")
