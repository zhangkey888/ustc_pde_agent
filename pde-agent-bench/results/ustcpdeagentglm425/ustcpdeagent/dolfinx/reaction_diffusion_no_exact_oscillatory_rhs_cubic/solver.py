import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc

def solve(case_spec: dict) -> dict:
    pde = case_spec.get("pde", {})
    time_params = pde.get("time", {})
    output_spec = case_spec.get("output", {})
    grid_spec = output_spec.get("grid", {})

    t0 = float(time_params.get("t0", 0.0))
    t_end = float(time_params.get("t_end", 0.3))
    dt_suggested = float(time_params.get("dt", 0.005))

    nx_grid = int(grid_spec.get("nx", 64))
    ny_grid = int(grid_spec.get("ny", 64))
    bbox = grid_spec.get("bbox", [0.0, 1.0, 0.0, 1.0])

    epsilon = float(pde.get("epsilon", 1.0))

    dt = dt_suggested
    n_steps = int(round((t_end - t0) / dt))

    # --- Mesh and Function Space ---
    mesh_res = 48
    domain = mesh.create_unit_square(MPI.COMM_WORLD, mesh_res, mesh_res, cell_type=mesh.CellType.triangle)

    elem_degree = 2
    V = fem.functionspace(domain, ("Lagrange", elem_degree))

    # --- Boundary Conditions ---
    fdim = domain.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool))
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    u_bc_func = fem.Function(V)
    u_bc_func.x.array[:] = 0.0
    bc = fem.dirichletbc(u_bc_func, boundary_dofs)

    # --- Define functions ---
    u = fem.Function(V)
    u_prev = fem.Function(V)

    def u0_expr(x):
        return 0.2 * np.sin(3.0 * np.pi * x[0]) * np.sin(2.0 * np.pi * x[1])

    u.interpolate(u0_expr)
    u_prev.interpolate(u0_expr)

    # Source term
    x_coord = ufl.SpatialCoordinate(domain)
    f_expr = ufl.sin(6.0 * ufl.pi * x_coord[0]) * ufl.sin(5.0 * ufl.pi * x_coord[1])
    f = fem.Function(V)
    f.interpolate(fem.Expression(f_expr, V.element.interpolation_points))

    # --- Nonlinear residual: Backward Euler ---
    v = ufl.TestFunction(V)
    F = (
        (u - u_prev) / dt * v * ufl.dx
        + epsilon * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
        + u**3 * v * ufl.dx
        - f * v * ufl.dx
    )
    J = ufl.derivative(F, u)

    # --- SNES solver ---
    petsc_options = {
        "snes_type": "newtonls",
        "snes_linesearch_type": "bt",
        "snes_rtol": 1e-10,
        "snes_atol": 1e-12,
        "snes_max_it": 20,
        "ksp_type": "preonly",
        "pc_type": "lu",
    }

    problem = petsc.NonlinearProblem(
        F, u, bcs=[bc], J=J,
        petsc_options_prefix="rd_",
        petsc_options=petsc_options
    )

    # --- Time stepping ---
    total_linear_iterations = 0
    nonlinear_iterations_list = []

    for step in range(n_steps):
        problem.solve()
        u.x.scatter_forward()

        snes = problem.solver
        newton_iters = int(snes.getIterationNumber())
        nonlinear_iterations_list.append(newton_iters)
        ksp = snes.getKSP()
        total_linear_iterations += int(ksp.getIterationNumber())

        u_prev.x.array[:] = u.x.array[:]
        u_prev.x.scatter_forward()

    # --- Sample solution onto output grid ---
    xs = np.linspace(bbox[0], bbox[1], nx_grid)
    ys = np.linspace(bbox[2], bbox[3], ny_grid)
    XX, YY = np.meshgrid(xs, ys)
    pts = np.zeros((3, nx_grid * ny_grid))
    pts[0, :] = XX.ravel()
    pts[1, :] = YY.ravel()

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

    u_grid = np.zeros((ny_grid, nx_grid))
    if len(points_on_proc) > 0:
        vals = u.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u_flat = np.zeros(nx_grid * ny_grid)
        u_flat[eval_map] = vals.flatten()
        u_grid = u_flat.reshape(ny_grid, nx_grid)

    # --- Initial condition on grid ---
    u_init_func = fem.Function(V)
    u_init_func.interpolate(u0_expr)
    u_initial_grid = np.zeros((ny_grid, nx_grid))
    if len(points_on_proc) > 0:
        vals0 = u_init_func.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u0_flat = np.zeros(nx_grid * ny_grid)
        u0_flat[eval_map] = vals0.flatten()
        u_initial_grid = u0_flat.reshape(ny_grid, nx_grid)

    solver_info = {
        "mesh_resolution": mesh_res,
        "element_degree": elem_degree,
        "ksp_type": "preonly",
        "pc_type": "lu",
        "rtol": 1e-10,
        "iterations": total_linear_iterations,
        "dt": dt,
        "n_steps": n_steps,
        "time_scheme": "backward_euler",
        "nonlinear_iterations": nonlinear_iterations_list,
    }

    return {
        "u": u_grid,
        "u_initial": u_initial_grid,
        "solver_info": solver_info,
    }
