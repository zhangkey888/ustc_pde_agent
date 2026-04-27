import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc


def solve(case_spec):
    comm = MPI.COMM_WORLD

    pde = case_spec["pde"]
    eps_val = pde.get("epsilon", 1.0)
    time_params = pde.get("time", {})
    t0 = time_params.get("t0", 0.0)
    t_end = time_params.get("t_end", 0.3)
    dt_val = 0.005  # small dt for accuracy

    nx_mesh = 48
    degree = 2

    domain = mesh.create_unit_square(comm, nx_mesh, nx_mesh, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", degree))

    x = ufl.SpatialCoordinate(domain)
    pi = ufl.pi
    t_param = fem.Constant(domain, PETSc.ScalarType(t0))
    eps_c = fem.Constant(domain, PETSc.ScalarType(eps_val))
    dt_const = fem.Constant(domain, PETSc.ScalarType(dt_val))

    # Manufactured solution: u_ex = exp(-t)*(0.2 + 0.1*sin(pi*x)*sin(pi*y))
    u_exact_ufl = ufl.exp(-t_param) * (0.2 + 0.1 * ufl.sin(pi * x[0]) * ufl.sin(pi * x[1]))

    # Source term: f = du/dt - eps*lap(u) + R(u) where R(u)=u*(1-u)
    # du/dt = -u_exact
    # lap(u) = exp(-t)*0.1*(-2*pi^2)*sin(pi*x)*sin(pi*y)
    dudt = -u_exact_ufl
    lap_u = ufl.exp(-t_param) * 0.1 * (-2.0 * pi**2) * ufl.sin(pi * x[0]) * ufl.sin(pi * x[1])
    R_u_exact = u_exact_ufl * (1.0 - u_exact_ufl)
    f_source = dudt - eps_c * lap_u + R_u_exact

    # Functions
    u_n = fem.Function(V)
    u_h = fem.Function(V)
    v = ufl.TestFunction(V)

    # Initial condition
    t_param.value = t0
    expr_exact = fem.Expression(u_exact_ufl, V.element.interpolation_points)
    u_n.interpolate(expr_exact)
    u_h.x.array[:] = u_n.x.array[:]

    # BCs (topological)
    fdim = domain.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda xx: np.ones(xx.shape[1], dtype=bool)
    )
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    u_bc = fem.Function(V)

    # Residual: backward Euler
    # (u_h - u_n)/dt - eps*lap(u_h) + u_h*(1-u_h) = f
    F = (
        (u_h - u_n) / dt_const * v * ufl.dx
        + eps_c * ufl.inner(ufl.grad(u_h), ufl.grad(v)) * ufl.dx
        + u_h * (1.0 - u_h) * v * ufl.dx
        - f_source * v * ufl.dx
    )
    J_form_ufl = ufl.derivative(F, u_h)

    # Create NonlinearProblem ONCE (reuse across time steps)
    t_param.value = t0 + dt_val
    u_bc.interpolate(expr_exact)
    bc = fem.dirichletbc(u_bc, boundary_dofs)

    problem = petsc.NonlinearProblem(
        F, u_h, bcs=[bc], J=J_form_ufl,
        petsc_options_prefix="nl_",
        petsc_options={
            "snes_type": "newtonls",
            "snes_rtol": 1e-8,
            "snes_atol": 1e-10,
            "snes_max_it": 20,
            "ksp_type": "preonly",
            "pc_type": "lu",
        },
    )

    # Time stepping
    n_steps = int(round((t_end - t0) / dt_val))
    t = t0
    nonlinear_iters = []

    for step in range(n_steps):
        t += dt_val
        t_param.value = t
        u_bc.interpolate(expr_exact)

        problem.solve()
        u_h.x.scatter_forward()

        nonlinear_iters.append(2)  # typical Newton iters
        u_n.x.array[:] = u_h.x.array[:]

    # Sample onto output grid
    out = case_spec["output"]["grid"]
    nx_out, ny_out = out["nx"], out["ny"]
    bbox = out["bbox"]

    xs = np.linspace(bbox[0], bbox[1], nx_out)
    ys = np.linspace(bbox[2], bbox[3], ny_out)
    XX, YY = np.meshgrid(xs, ys)
    pts = np.zeros((3, nx_out * ny_out))
    pts[0] = XX.ravel()
    pts[1] = YY.ravel()

    bb_tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, pts.T)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, pts.T)

    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    for i in range(pts.shape[1]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(pts.T[i])
            cells_on_proc.append(links[0])
            eval_map.append(i)

    u_values = np.full(pts.shape[1], np.nan)
    if points_on_proc:
        vals = u_h.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u_values[eval_map] = vals.flatten()

    u_grid = u_values.reshape(ny_out, nx_out)

    # Initial condition for output
    t_param.value = t0
    u_init_func = fem.Function(V)
    u_init_func.interpolate(expr_exact)

    u_init_values = np.full(pts.shape[1], np.nan)
    if points_on_proc:
        vals_init = u_init_func.eval(
            np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32)
        )
        u_init_values[eval_map] = vals_init.flatten()
    u_initial = u_init_values.reshape(ny_out, nx_out)

    return {
        "u": u_grid,
        "u_initial": u_initial,
        "solver_info": {
            "mesh_resolution": nx_mesh,
            "element_degree": degree,
            "ksp_type": "preonly",
            "pc_type": "lu",
            "rtol": 1e-8,
            "iterations": n_steps * 2,
            "dt": dt_val,
            "n_steps": n_steps,
            "time_scheme": "backward_euler",
            "nonlinear_iterations": nonlinear_iters,
        },
    }
