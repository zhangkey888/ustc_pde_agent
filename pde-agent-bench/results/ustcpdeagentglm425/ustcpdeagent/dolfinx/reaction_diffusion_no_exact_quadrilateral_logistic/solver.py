import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc as petsc_fem
import ufl
from petsc4py import PETSc

ScalarType = PETSc.ScalarType

def solve(case_spec: dict) -> dict:
    pde = case_spec.get("pde", {})
    time_params = pde.get("time", {})
    is_transient = bool(time_params)

    domain_info = pde.get("domain", {})
    bbox_domain = domain_info.get("bbox", [0.0, 1.0, 0.0, 1.0])
    xmin_d, xmax_d, ymin_d, ymax_d = bbox_domain

    epsilon = pde.get("epsilon", 1.0)
    if epsilon is None:
        epsilon = 1.0
    f_val = pde.get("f", 1.0)
    if f_val is None:
        f_val = 1.0
    rho = pde.get("rho", 1.0)
    if rho is None:
        rho = 1.0

    bc_val = 0.0
    for bc_info in pde.get("bcs", []):
        if bc_info.get("type", "dirichlet") == "dirichlet":
            bc_val = bc_info.get("value", 0.0)

    t0 = time_params.get("t0", 0.0) if is_transient else 0.0
    t_end = time_params.get("t_end", 0.3) if is_transient else 0.3
    dt_suggested = time_params.get("dt", 0.01) if is_transient else 0.01

    output = case_spec.get("output", {})
    grid = output.get("grid", {})
    nx_out = grid.get("nx", 50)
    ny_out = grid.get("ny", 50)
    bbox_out = grid.get("bbox", [xmin_d, xmax_d, ymin_d, ymax_d])

    mesh_res = 96
    element_degree = 2
    dt = dt_suggested

    comm = MPI.COMM_WORLD
    p0 = np.array([xmin_d, ymin_d], dtype=np.float64)
    p1 = np.array([xmax_d, ymax_d], dtype=np.float64)
    domain = mesh.create_rectangle(comm, [p0, p1], [mesh_res, mesh_res],
                                   cell_type=mesh.CellType.quadrilateral)

    tdim = domain.topology.dim
    fdim = tdim - 1

    V = fem.functionspace(domain, ("Lagrange", element_degree))

    boundary_facets = mesh.locate_entities_boundary(domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool))
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc_func = fem.Function(V)
    bc_func.interpolate(lambda x: np.full(x.shape[1], bc_val, dtype=ScalarType))
    bc = fem.dirichletbc(bc_func, boundary_dofs)

    u = fem.Function(V)
    u_n = fem.Function(V)
    v = ufl.TestFunction(V)

    def u0_func(x):
        return (0.25 + 0.15 * np.sin(np.pi * x[0]) * np.sin(np.pi * x[1])).astype(ScalarType)

    u_n.interpolate(u0_func)
    u.x.array[:] = u_n.x.array[:]

    f_const = fem.Constant(domain, ScalarType(f_val))
    dt_const = fem.Constant(domain, ScalarType(dt))

    F = (((u - u_n) / dt_const * v
          + epsilon * ufl.inner(ufl.grad(u), ufl.grad(v))
          + rho * u * (1.0 - u) * v
          - f_const * v) * ufl.dx)

    J = ufl.derivative(F, u)

    problem = petsc_fem.NonlinearProblem(
        F, u, bcs=[bc], J=J,
        petsc_options_prefix="rd_",
        petsc_options={
            "snes_type": "newtonls",
            "snes_linesearch_type": "bt",
            "snes_rtol": 1e-10,
            "snes_atol": 1e-12,
            "snes_max_it": 25,
            "ksp_type": "preonly",
            "pc_type": "lu",
        }
    )

    n_steps = int(round((t_end - t0) / dt))
    if n_steps < 1:
        n_steps = 1
        dt = (t_end - t0) / n_steps
        dt_const.value = ScalarType(dt)

    nonlinear_iterations = []
    total_linear_iterations = 0
    snes = problem.solver

    for step in range(n_steps):
        u.x.scatter_forward()
        problem.solve()
        u.x.scatter_forward()
        nonlinear_iterations.append(snes.getIterationNumber())
        total_linear_iterations += snes.getLinearSolveIterations()
        u_n.x.array[:] = u.x.array[:]
        u_n.x.scatter_forward()

    # Sample solution onto output grid
    xs = np.linspace(bbox_out[0], bbox_out[1], nx_out)
    ys = np.linspace(bbox_out[2], bbox_out[3], ny_out)
    XX, YY = np.meshgrid(xs, ys)

    points = np.zeros((3, nx_out * ny_out), dtype=np.float64)
    points[0, :] = XX.ravel()
    points[1, :] = YY.ravel()

    bb_tree = geometry.bb_tree(domain, tdim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, points.T)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points.T)

    u_values = np.full(nx_out * ny_out, np.nan, dtype=np.float64)
    pts_list, cells_list, idx_list = [], [], []
    for i in range(points.shape[1]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            pts_list.append(points[:, i])
            cells_list.append(links[0])
            idx_list.append(i)

    pts_arr = None
    cells_arr = None
    if pts_list:
        pts_arr = np.array(pts_list, dtype=np.float64)
        cells_arr = np.array(cells_list, dtype=np.int32)
        u_values[idx_list] = u.eval(pts_arr, cells_arr).flatten()

    u_grid = u_values.reshape(ny_out, nx_out)

    # Sample initial condition
    u_init = fem.Function(V)
    u_init.interpolate(u0_func)
    u_initial_grid = np.full((ny_out, nx_out), np.nan, dtype=np.float64)
    if pts_arr is not None:
        u_init_flat = np.full(nx_out * ny_out, np.nan, dtype=np.float64)
        u_init_flat[idx_list] = u_init.eval(pts_arr, cells_arr).flatten()
        u_initial_grid = u_init_flat.reshape(ny_out, nx_out)

    # Accuracy verification
    print(f"Solution range: [{np.nanmin(u_grid):.6e}, {np.nanmax(u_grid):.6e}]")
    print(f"Newton iters: {nonlinear_iterations}")

    return {
        "u": u_grid,
        "u_initial": u_initial_grid,
        "solver_info": {
            "mesh_resolution": mesh_res,
            "element_degree": element_degree,
            "ksp_type": "preonly",
            "pc_type": "lu",
            "rtol": 1e-10,
            "iterations": total_linear_iterations,
            "dt": dt,
            "n_steps": n_steps,
            "time_scheme": "backward_euler",
            "nonlinear_iterations": nonlinear_iterations,
        }
    }
