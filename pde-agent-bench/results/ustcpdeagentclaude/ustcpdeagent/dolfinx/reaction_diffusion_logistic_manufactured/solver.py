import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc as fem_petsc
import ufl
from petsc4py import PETSc


def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD

    # Parameters
    agent_params = case_spec.get("agent_params", {}) or {}
    pde = case_spec.get("pde", {}) or {}
    time_info = pde.get("time", {}) or {}

    epsilon = float(agent_params.get("epsilon", pde.get("epsilon", 0.01)))
    mesh_N = int(agent_params.get("mesh_resolution", 64))
    degree = int(agent_params.get("element_degree", 2))
    dt_val = float(agent_params.get("dt", time_info.get("dt", 0.01)))
    t0 = float(time_info.get("t0", 0.0))
    t_end = float(time_info.get("t_end", 0.3))
    time_scheme = agent_params.get("time_scheme", "backward_euler")

    # Create mesh
    msh = mesh.create_unit_square(comm, mesh_N, mesh_N, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(msh, ("Lagrange", degree))

    x = ufl.SpatialCoordinate(msh)
    t_const = fem.Constant(msh, PETSc.ScalarType(t0))
    dt_const = fem.Constant(msh, PETSc.ScalarType(dt_val))
    eps_const = fem.Constant(msh, PETSc.ScalarType(epsilon))

    # Manufactured exact solution: u = exp(-t)*(0.2 + 0.1*sin(pi*x)*sin(pi*y))
    pi = ufl.pi
    def u_exact_expr(t_c):
        return ufl.exp(-t_c) * (0.2 + 0.1 * ufl.sin(pi * x[0]) * ufl.sin(pi * x[1]))

    # R(u) = u*(1-u) (logistic)
    def R(u):
        return u * (1.0 - u)

    # Compute f from manufactured solution
    # u_t - eps*lap(u) + R(u) = f
    u_ex = u_exact_expr(t_const)
    # time derivative
    u_ex_dt = -ufl.exp(-t_const) * (0.2 + 0.1 * ufl.sin(pi * x[0]) * ufl.sin(pi * x[1]))
    lap_u = ufl.exp(-t_const) * 0.1 * (-2.0 * pi**2 * ufl.sin(pi * x[0]) * ufl.sin(pi * x[1]))
    f_expr = u_ex_dt - eps_const * lap_u + R(u_ex)

    # Boundary condition: u = u_exact on boundary
    fdim = msh.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(
        msh, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
    )
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)

    u_bc = fem.Function(V)
    bc_expr = fem.Expression(u_exact_expr(t_const), V.element.interpolation_points)
    u_bc.interpolate(bc_expr)
    bc = fem.dirichletbc(u_bc, boundary_dofs)

    # Initial condition
    u_n = fem.Function(V)  # previous time step
    u_h = fem.Function(V)  # current
    ic_expr = fem.Expression(u_exact_expr(fem.Constant(msh, PETSc.ScalarType(t0))),
                              V.element.interpolation_points)
    u_n.interpolate(ic_expr)
    u_h.interpolate(ic_expr)

    # Save initial for output
    # Variational form for backward Euler (nonlinear):
    # (u_h - u_n)/dt - eps*lap(u_h) + R(u_h) = f
    # Weak: (u_h - u_n)/dt * v + eps*grad(u_h)·grad(v) + R(u_h)*v - f*v = 0
    v = ufl.TestFunction(V)

    F = ((u_h - u_n) / dt_const) * v * ufl.dx \
        + eps_const * ufl.inner(ufl.grad(u_h), ufl.grad(v)) * ufl.dx \
        + R(u_h) * v * ufl.dx \
        - f_expr * v * ufl.dx

    J = ufl.derivative(F, u_h)

    petsc_options = {
        "snes_type": "newtonls",
        "snes_rtol": float(agent_params.get("newton_rtol", 1e-10)),
        "snes_atol": 1e-12,
        "snes_max_it": 30,
        "ksp_type": "preonly",
        "pc_type": "lu",
    }

    problem = fem_petsc.NonlinearProblem(
        F, u_h, bcs=[bc], J=J,
        petsc_options_prefix="rd_",
        petsc_options=petsc_options,
    )

    # Output grid setup
    grid_info = case_spec["output"]["grid"]
    nx = int(grid_info["nx"])
    ny = int(grid_info["ny"])
    bbox = grid_info["bbox"]
    xs = np.linspace(bbox[0], bbox[1], nx)
    ys = np.linspace(bbox[2], bbox[3], ny)
    XX, YY = np.meshgrid(xs, ys)
    pts = np.vstack([XX.ravel(), YY.ravel(), np.zeros(nx * ny)]).T

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
    points_on_proc = np.array(points_on_proc)
    cells_on_proc = np.array(cells_on_proc, dtype=np.int32)

    def sample(u_func):
        vals = u_func.eval(points_on_proc, cells_on_proc).flatten()
        arr = np.full(nx * ny, np.nan)
        arr[eval_map] = vals
        return arr.reshape(ny, nx)

    u_initial = sample(u_n)

    # Time stepping
    n_steps = int(round((t_end - t0) / dt_val))
    t_current = t0
    nonlin_iters = []

    for step in range(n_steps):
        t_current += dt_val
        t_const.value = t_current
        # Update BC
        u_bc.interpolate(bc_expr)

        try:
            result = problem.solve()
            # result is (u_h, converged_reason, iterations) in some versions
            if isinstance(result, tuple):
                niter = result[-1] if len(result) >= 2 else 1
            else:
                niter = 1
        except Exception:
            niter = -1
        nonlin_iters.append(int(niter) if isinstance(niter, (int, np.integer)) else 1)

        u_h.x.scatter_forward()
        u_n.x.array[:] = u_h.x.array

    u_grid = sample(u_h)

    solver_info = {
        "mesh_resolution": mesh_N,
        "element_degree": degree,
        "ksp_type": "preonly",
        "pc_type": "lu",
        "rtol": float(petsc_options["snes_rtol"]),
        "iterations": int(sum(nonlin_iters)),
        "dt": dt_val,
        "n_steps": n_steps,
        "time_scheme": time_scheme,
        "nonlinear_iterations": nonlin_iters,
    }

    return {
        "u": u_grid,
        "u_initial": u_initial,
        "solver_info": solver_info,
    }
