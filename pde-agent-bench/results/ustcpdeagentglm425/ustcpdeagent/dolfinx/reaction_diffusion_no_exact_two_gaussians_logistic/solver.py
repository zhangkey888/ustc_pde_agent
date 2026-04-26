import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc
import time as time_module


def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD

    pde = case_spec.get("pde", {})
    output = case_spec.get("output", {})
    grid = output.get("grid", {})

    nx_out = grid.get("nx", 100)
    ny_out = grid.get("ny", 100)
    bbox = grid.get("bbox", [0, 1, 0, 1])

    time_params = pde.get("time", {})
    t0 = float(time_params.get("t0", 0.0))
    t_end = float(time_params.get("t_end", 0.4))

    epsilon = None
    for key in ["epsilon", "eps", "diffusion_coefficient", "diffusivity"]:
        if key in pde:
            epsilon = float(pde[key])
            break
    if epsilon is None:
        params = pde.get("parameters", {})
        for key in ["epsilon", "eps", "diffusion_coefficient", "diffusivity"]:
            if key in params:
                epsilon = float(params[key])
                break
    if epsilon is None:
        epsilon = 1.0

    alpha = None
    for key in ["reaction_coefficient", "alpha", "reaction_coeff"]:
        if key in pde:
            alpha = float(pde[key])
            break
    if alpha is None:
        params = pde.get("parameters", {})
        for key in ["reaction_coefficient", "alpha", "reaction_coeff"]:
            if key in params:
                alpha = float(params[key])
                break
    if alpha is None:
        alpha = 1.0

    mesh_res = 48
    elem_degree = 2
    dt = 0.001

    n_steps = int(round((t_end - t0) / dt))
    dt = (t_end - t0) / n_steps

    domain = mesh.create_unit_square(comm, mesh_res, mesh_res, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", elem_degree))

    u = fem.Function(V)
    u_prev = fem.Function(V)
    f_func = fem.Function(V)
    v = ufl.TestFunction(V)
    x = ufl.SpatialCoordinate(domain)

    f_expr = 6.0 * (
        ufl.exp(-160.0 * ((x[0] - 0.3)**2 + (x[1] - 0.7)**2))
        + 0.8 * ufl.exp(-160.0 * ((x[0] - 0.75)**2 + (x[1] - 0.35)**2))
    )
    f_func.interpolate(fem.Expression(f_expr, V.element.interpolation_points))

    u0_expr = (
        0.3 * ufl.exp(-50.0 * ((x[0] - 0.3)**2 + (x[1] - 0.5)**2))
        + 0.3 * ufl.exp(-50.0 * ((x[0] - 0.7)**2 + (x[1] - 0.5)**2))
    )
    u.interpolate(fem.Expression(u0_expr, V.element.interpolation_points))
    u_prev.interpolate(fem.Expression(u0_expr, V.element.interpolation_points))

    fdim = domain.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool))
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc = fem.dirichletbc(PETSc.ScalarType(0.0), boundary_dofs, V)

    F = (u - u_prev) * v * ufl.dx
    F += dt * epsilon * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
    F += dt * alpha * u * (1.0 - u) * v * ufl.dx
    F -= dt * f_func * v * ufl.dx

    J = ufl.derivative(F, u)

    po = {
        "snes_type": "newtonls",
        "snes_linesearch_type": "bt",
        "snes_rtol": 1e-8,
        "snes_atol": 1e-10,
        "snes_max_it": 25,
        "ksp_type": "gmres",
        "pc_type": "ilu",
        "ksp_rtol": 1e-8,
        "ksp_atol": 1e-10,
        "ksp_max_it": 200,
    }

    problem = petsc.NonlinearProblem(F, u, bcs=[bc], J=J, petsc_options_prefix="rd_", petsc_options=po)

    nonlinear_iterations = []
    total_linear_iterations = 0

    u_initial_grid = _sample_on_grid(u, domain, nx_out, ny_out, bbox)

    for step in range(n_steps):
        problem.solve()
        u.x.scatter_forward()
        snes = problem._snes
        nonlinear_iterations.append(snes.getIterationNumber())
        total_linear_iterations += snes.getLinearSolveIterations()
        u_prev.x.array[:] = u.x.array[:]
        u_prev.x.scatter_forward()

    u_grid = _sample_on_grid(u, domain, nx_out, ny_out, bbox)

    solver_info = {
        "mesh_resolution": mesh_res,
        "element_degree": elem_degree,
        "ksp_type": "gmres",
        "pc_type": "ilu",
        "rtol": 1e-8,
        "iterations": total_linear_iterations,
        "dt": dt,
        "n_steps": n_steps,
        "time_scheme": "backward_euler",
        "nonlinear_iterations": nonlinear_iterations,
    }

    return {"u": u_grid, "u_initial": u_initial_grid, "solver_info": solver_info}


def _sample_on_grid(u_func, domain, nx, ny, bbox):
    xs = np.linspace(bbox[0], bbox[1], nx)
    ys = np.linspace(bbox[2], bbox[3], ny)
    XX, YY = np.meshgrid(xs, ys)
    points = np.vstack([XX.ravel(), YY.ravel(), np.zeros(nx * ny)])

    bb_tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, points.T)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points.T)

    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    for i in range(points.shape[1]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(points.T[i])
            cells_on_proc.append(links[0])
            eval_map.append(i)

    u_values = np.full((points.shape[1],), np.nan)
    if len(points_on_proc) > 0:
        pts_arr = np.array(points_on_proc)
        cells_arr = np.array(cells_on_proc, dtype=np.int32)
        vals = u_func.eval(pts_arr, cells_arr)
        u_values[eval_map] = vals.flatten()

    if domain.comm.size > 1:
        all_values = domain.comm.allgather(u_values)
        result = np.full_like(u_values, np.nan)
        for arr in all_values:
            mask = ~np.isnan(arr)
            result[mask] = arr[mask]
        u_values = result

    return u_values.reshape(ny, nx)
