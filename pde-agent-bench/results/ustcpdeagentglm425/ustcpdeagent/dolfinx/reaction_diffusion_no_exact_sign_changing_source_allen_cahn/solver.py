import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc

def solve(case_spec: dict) -> dict:
    pde = case_spec["pde"]
    time_params = pde.get("time", {})
    t0 = time_params.get("t0", 0.0)
    t_end = time_params.get("t_end", 0.2)
    dt_suggested = time_params.get("dt", 0.005)

    output = case_spec["output"]
    grid = output["grid"]
    nx_out = grid["nx"]
    ny_out = grid["ny"]
    bbox = grid["bbox"]

    epsilon = 1.0
    mesh_res = 96
    elem_degree = 2
    dt = dt_suggested
    n_steps = int(round((t_end - t0) / dt))

    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_square(comm, mesh_res, mesh_res, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", elem_degree))

    tdim = domain.topology.dim
    fdim = tdim - 1
    boundary_facets = mesh.locate_entities_boundary(domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool))
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc_val = fem.Function(V)
    bc_val.x.array[:] = 0.0
    bc = fem.dirichletbc(bc_val, boundary_dofs)

    u = fem.Function(V)
    u_n = fem.Function(V)
    v = ufl.TestFunction(V)

    x = ufl.SpatialCoordinate(domain)
    u0_expr = 0.2 * ufl.sin(3 * ufl.pi * x[0]) * ufl.sin(2 * ufl.pi * x[1])
    u_n.interpolate(fem.Expression(u0_expr, V.element.interpolation_points))
    u.x.array[:] = u_n.x.array[:]

    u_init = fem.Function(V)
    u_init.interpolate(fem.Expression(u0_expr, V.element.interpolation_points))

    f_expr = 3.0 * ufl.cos(3 * ufl.pi * x[0]) * ufl.sin(2 * ufl.pi * x[1])
    f = fem.Function(V)
    f.interpolate(fem.Expression(f_expr, V.element.interpolation_points))

    F = (
        (u - u_n) / dt * v * ufl.dx
        + epsilon * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
        + (u**3 - u) * v * ufl.dx
        - f * v * ufl.dx
    )
    J = ufl.derivative(F, u)

    ksp_type = "preonly"
    pc_type = "lu"
    rtol_val = 1e-8

    petsc_opts = {
        "snes_type": "newtonls",
        "snes_linesearch_type": "bt",
        "snes_rtol": 1e-8,
        "snes_atol": 1e-10,
        "snes_max_it": 25,
        "ksp_type": ksp_type,
        "pc_type": pc_type,
    }

    problem = petsc.NonlinearProblem(F, u, bcs=[bc], J=J,
                                      petsc_options_prefix="ac_",
                                      petsc_options=petsc_opts)

    nonlin_iters_list = []
    total_ksp_iters = 0

    for step in range(n_steps):
        u.x.array[:] = u_n.x.array[:]
        problem.solve()
        u.x.scatter_forward()

        snes = problem._snes
        nit = snes.getIterationNumber()
        nonlin_iters_list.append(int(nit))
        total_ksp_iters += snes.getKSP().getIterationNumber()

        u_n.x.array[:] = u.x.array[:]

    xmin, xmax, ymin, ymax = bbox
    xs = np.linspace(xmin, xmax, nx_out)
    ys = np.linspace(ymin, ymax, ny_out)
    XX, YY = np.meshgrid(xs, ys)
    pts_flat = np.column_stack([XX.ravel(), YY.ravel(), np.zeros(nx_out * ny_out)])

    bb_tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, pts_flat)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, pts_flat)

    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    for i in range(pts_flat.shape[0]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(pts_flat[i])
            cells_on_proc.append(links[0])
            eval_map.append(i)

    u_grid = np.zeros((ny_out, nx_out))
    u_init_grid = np.zeros((ny_out, nx_out))

    if len(points_on_proc) > 0:
        pts_arr = np.array(points_on_proc)
        cells_arr = np.array(cells_on_proc, dtype=np.int32)
        vals = u.eval(pts_arr, cells_arr).flatten()
        vals_init = u_init.eval(pts_arr, cells_arr).flatten()
        for idx, gi in enumerate(eval_map):
            row = gi // nx_out
            col = gi % nx_out
            u_grid[row, col] = vals[idx]
            u_init_grid[row, col] = vals_init[idx]

    u_grid_global = np.zeros_like(u_grid)
    comm.Allreduce(u_grid, u_grid_global, op=MPI.SUM)
    u_init_global = np.zeros_like(u_init_grid)
    comm.Allreduce(u_init_grid, u_init_global, op=MPI.SUM)

    solver_info = {
        "mesh_resolution": mesh_res,
        "element_degree": elem_degree,
        "ksp_type": ksp_type,
        "pc_type": pc_type,
        "rtol": rtol_val,
        "iterations": total_ksp_iters,
        "dt": dt,
        "n_steps": n_steps,
        "time_scheme": "backward_euler",
        "nonlinear_iterations": nonlin_iters_list,
    }

    return {
        "u": u_grid_global,
        "u_initial": u_init_global,
        "solver_info": solver_info,
    }
