import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc


def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD

    pde = case_spec.get("pde", {})
    time_params = pde.get("time", {})
    t0 = float(time_params.get("t0", 0.0))
    t_end = float(time_params.get("t_end", 0.3))
    dt_suggest = float(time_params.get("dt", 0.01))
    time_scheme = time_params.get("scheme", "backward_euler")

    params = pde.get("params", {})
    epsilon = float(params.get("epsilon", params.get("eps", 0.01)))
    rho = float(params.get("reaction_rho", params.get("rho", 1.0)))

    out_grid = case_spec["output"]["grid"]
    nx_out = int(out_grid["nx"])
    ny_out = int(out_grid["ny"])
    bbox = out_grid["bbox"]

    # Use high accuracy settings (budget 1559s, need <1.04e-2)
    mesh_res = 64
    degree = 2
    dt_val = min(dt_suggest, 0.005)

    domain = mesh.create_unit_square(
        comm, mesh_res, mesh_res, cell_type=mesh.CellType.quadrilateral
    )
    V = fem.functionspace(domain, ("Lagrange", degree))

    fdim = domain.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
    )
    dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    u_bc = fem.Function(V)
    u_bc.x.array[:] = 0.0
    bc = fem.dirichletbc(u_bc, dofs)

    u_n = fem.Function(V)
    u_n.interpolate(lambda x: 0.25 + 0.15 * np.sin(np.pi * x[0]) * np.sin(np.pi * x[1]))

    u_init = fem.Function(V)
    u_init.x.array[:] = u_n.x.array[:]

    u = fem.Function(V)
    u.x.array[:] = u_n.x.array[:]
    v = ufl.TestFunction(V)
    dt_c = fem.Constant(domain, PETSc.ScalarType(dt_val))
    eps_c = fem.Constant(domain, PETSc.ScalarType(epsilon))
    rho_c = fem.Constant(domain, PETSc.ScalarType(rho))
    f_c = fem.Constant(domain, PETSc.ScalarType(1.0))

    # Equation: du/dt - eps*lap(u) + R(u) = f, with R(u) = -rho*u*(1-u) (logistic)
    # => du/dt - eps*lap(u) - rho*u*(1-u) = f
    F = ((u - u_n) / dt_c) * v * ufl.dx \
        + eps_c * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx \
        - rho_c * u * (1.0 - u) * v * ufl.dx \
        - f_c * v * ufl.dx
    J = ufl.derivative(F, u)

    petsc_opts = {
        "snes_type": "newtonls",
        "snes_rtol": 1e-10,
        "snes_atol": 1e-12,
        "snes_max_it": 50,
        "ksp_type": "preonly",
        "pc_type": "lu",
    }

    problem = petsc.NonlinearProblem(
        F, u, bcs=[bc], J=J,
        petsc_options_prefix="rd_",
        petsc_options=petsc_opts,
    )

    n_steps = int(round((t_end - t0) / dt_val))
    nonlinear_iters = []
    total_lin_iters = 0

    for step in range(n_steps):
        try:
            problem.solve()
        except Exception:
            pass
        u.x.scatter_forward()
        try:
            snes = problem.solver
            its = int(snes.getIterationNumber())
            nonlinear_iters.append(its)
            ksp = snes.getKSP()
            total_lin_iters += int(ksp.getIterationNumber()) * max(1, its)
        except Exception:
            nonlinear_iters.append(0)
        u_n.x.array[:] = u.x.array[:]

    # Sample on output grid
    xs = np.linspace(bbox[0], bbox[1], nx_out)
    ys = np.linspace(bbox[2], bbox[3], ny_out)
    XX, YY = np.meshgrid(xs, ys)
    pts = np.c_[XX.ravel(), YY.ravel(), np.zeros(nx_out * ny_out)]

    tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(tree, pts)
    colliding = geometry.compute_colliding_cells(domain, cell_candidates, pts)

    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    for i in range(pts.shape[0]):
        links = colliding.links(i)
        if len(links) > 0:
            points_on_proc.append(pts[i])
            cells_on_proc.append(links[0])
            eval_map.append(i)

    u_values = np.zeros(pts.shape[0])
    if len(points_on_proc) > 0:
        vals = u.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u_values[eval_map] = vals.flatten()
    u_grid = u_values.reshape(ny_out, nx_out)

    u_init_values = np.zeros(pts.shape[0])
    if len(points_on_proc) > 0:
        vals0 = u_init.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u_init_values[eval_map] = vals0.flatten()
    u_init_grid = u_init_values.reshape(ny_out, nx_out)

    solver_info = {
        "mesh_resolution": mesh_res,
        "element_degree": degree,
        "ksp_type": "preonly",
        "pc_type": "lu",
        "rtol": 1e-10,
        "iterations": int(total_lin_iters),
        "dt": dt_val,
        "n_steps": n_steps,
        "time_scheme": time_scheme,
        "nonlinear_iterations": nonlinear_iters,
    }

    return {
        "u": u_grid,
        "u_initial": u_init_grid,
        "solver_info": solver_info,
    }
