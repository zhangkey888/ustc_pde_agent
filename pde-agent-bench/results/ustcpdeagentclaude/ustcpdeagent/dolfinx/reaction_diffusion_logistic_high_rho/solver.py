import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc


def solve(case_spec: dict) -> dict:
    # Extract parameters
    pde = case_spec.get("pde", {})
    params = pde.get("parameters", {}) if isinstance(pde, dict) else {}
    # Also check top level
    for key in ["parameters", "params"]:
        if key in case_spec:
            params = {**params, **case_spec[key]}

    epsilon = float(params.get("epsilon", 0.01))
    rho = float(params.get("reaction_rho", params.get("rho", 10.0)))

    # Time params
    time_info = pde.get("time", {}) if isinstance(pde, dict) else {}
    if not time_info:
        time_info = case_spec.get("time", {})
    t0 = float(time_info.get("t0", 0.0))
    t_end = float(time_info.get("t_end", 0.2))
    dt_suggested = float(time_info.get("dt", 0.005))

    # Output grid
    grid = case_spec["output"]["grid"]
    nx_out = int(grid["nx"])
    ny_out = int(grid["ny"])
    bbox = grid["bbox"]

    # Mesh resolution / element degree
    mesh_res = int(params.get("mesh_resolution", 48))
    degree = int(params.get("element_degree", 2))

    # Choose dt
    dt_val = min(float(params.get("dt", dt_suggested)), 0.0025)
    n_steps = int(round((t_end - t0) / dt_val))
    if n_steps < 1:
        n_steps = 1
    dt_val = (t_end - t0) / n_steps

    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_square(comm, mesh_res, mesh_res, cell_type=mesh.CellType.triangle)

    V = fem.functionspace(domain, ("Lagrange", degree))

    x = ufl.SpatialCoordinate(domain)
    t_const = fem.Constant(domain, PETSc.ScalarType(t0))

    # Manufactured solution
    def u_exact_ufl(t_c):
        return ufl.exp(-t_c) * (0.35 + 0.1 * ufl.cos(2*ufl.pi*x[0]) * ufl.sin(ufl.pi*x[1]))

    # R(u) = -rho * u * (1 - u)  (logistic reaction, positive rho = growth)
    # Manufactured f: f = du/dt - eps*laplacian(u) + R(u)
    u_ex = u_exact_ufl(t_const)
    # du/dt
    dudt = -ufl.exp(-t_const) * (0.35 + 0.1 * ufl.cos(2*ufl.pi*x[0]) * ufl.sin(ufl.pi*x[1]))
    # laplacian of u
    # d2u/dx2 = exp(-t)*0.1*(-(2pi)^2)*cos(2pi x)*sin(pi y)
    # d2u/dy2 = exp(-t)*0.1*cos(2pi x)*(-(pi)^2)*sin(pi y)
    lap_u = ufl.exp(-t_const) * 0.1 * (-(2*ufl.pi)**2 - (ufl.pi)**2) * ufl.cos(2*ufl.pi*x[0]) * ufl.sin(ufl.pi*x[1])
    R_uex = -rho * u_ex * (1.0 - u_ex)
    f_src = dudt - epsilon * lap_u + R_uex

    # Boundary condition from exact
    u_bc = fem.Function(V)
    u_bc_expr = fem.Expression(u_ex, V.element.interpolation_points)
    u_bc.interpolate(u_bc_expr)

    fdim = domain.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda xx: np.ones(xx.shape[1], dtype=bool)
    )
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc = fem.dirichletbc(u_bc, boundary_dofs)

    # Initial condition
    u_n = fem.Function(V)
    u_n.interpolate(fem.Expression(u_exact_ufl(fem.Constant(domain, PETSc.ScalarType(t0))), V.element.interpolation_points))

    # Store initial
    u_initial_func = fem.Function(V)
    u_initial_func.x.array[:] = u_n.x.array[:]

    # Current solution (unknown)
    u_h = fem.Function(V)
    u_h.x.array[:] = u_n.x.array[:]

    v = ufl.TestFunction(V)
    dt_c = fem.Constant(domain, PETSc.ScalarType(dt_val))

    # Backward Euler: (u - u_n)/dt - eps*lap(u) + R(u) = f
    # Weak form: (u-u_n)/dt * v + eps*grad(u).grad(v) + R(u)*v - f*v = 0
    R_u = -rho * u_h * (1.0 - u_h)
    F = ((u_h - u_n) / dt_c) * v * ufl.dx \
        + epsilon * ufl.inner(ufl.grad(u_h), ufl.grad(v)) * ufl.dx \
        + R_u * v * ufl.dx \
        - f_src * v * ufl.dx

    J = ufl.derivative(F, u_h)

    petsc_opts = {
        "snes_type": "newtonls",
        "snes_rtol": 1e-10,
        "snes_atol": 1e-12,
        "snes_max_it": 30,
        "ksp_type": "cg",
        "pc_type": "hypre",
    }

    problem = petsc.NonlinearProblem(F, u_h, bcs=[bc], J=J,
                                      petsc_options_prefix="rd_",
                                      petsc_options=petsc_opts)

    nonlinear_iters = []
    total_lin_iters = 0

    t_cur = t0
    for step in range(n_steps):
        t_cur = t0 + (step + 1) * dt_val
        t_const.value = t_cur
        # update BC
        u_bc.interpolate(fem.Expression(u_exact_ufl(t_const), V.element.interpolation_points))

        u_h_result = problem.solve()
        u_h.x.scatter_forward()

        # try to extract SNES iteration count
        try:
            its = problem.solver.getIterationNumber()
            nonlinear_iters.append(int(its))
        except Exception:
            nonlinear_iters.append(0)

        u_n.x.array[:] = u_h.x.array[:]

    # Sample on grid
    xs = np.linspace(bbox[0], bbox[1], nx_out)
    ys = np.linspace(bbox[2], bbox[3], ny_out)
    XX, YY = np.meshgrid(xs, ys)
    pts = np.c_[XX.ravel(), YY.ravel(), np.zeros(nx_out*ny_out)]

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

    u_flat = np.full(nx_out*ny_out, np.nan)
    if points_on_proc:
        vals = u_h.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u_flat[idx_map] = vals.flatten()
    u_grid = u_flat.reshape(ny_out, nx_out)

    # Initial sample
    u_flat0 = np.full(nx_out*ny_out, np.nan)
    if points_on_proc:
        vals0 = u_initial_func.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u_flat0[idx_map] = vals0.flatten()
    u_grid0 = u_flat0.reshape(ny_out, nx_out)

    # Accuracy check vs manufactured
    u_exact_grid = np.exp(-t_end) * (0.35 + 0.1 * np.cos(2*np.pi*XX) * np.sin(np.pi*YY))
    err = float(np.sqrt(np.mean((u_grid - u_exact_grid)**2)))

    return {
        "u": u_grid,
        "u_initial": u_grid0,
        "solver_info": {
            "mesh_resolution": mesh_res,
            "element_degree": degree,
            "ksp_type": "cg",
            "pc_type": "hypre",
            "rtol": 1e-10,
            "iterations": total_lin_iters,
            "dt": dt_val,
            "n_steps": n_steps,
            "time_scheme": "backward_euler",
            "nonlinear_iterations": nonlinear_iters,
            "l2_error_vs_manufactured": err,
        },
    }


if __name__ == "__main__":
    case = {
        "pde": {
            "time": {"t0": 0.0, "t_end": 0.2, "dt": 0.005},
            "parameters": {"epsilon": 0.01, "reaction_rho": 10.0},
        },
        "output": {"grid": {"nx": 64, "ny": 64, "bbox": [0, 1, 0, 1]}},
    }
    import time
    t0 = time.time()
    res = solve(case)
    t1 = time.time()
    print("wall:", t1-t0)
    print("info:", res["solver_info"])
    print("err:", res["solver_info"]["l2_error_vs_manufactured"])
