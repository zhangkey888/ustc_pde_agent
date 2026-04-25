import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc
import uuid


def solve(case_spec: dict) -> dict:
    pde = case_spec.get("pde", {})
    time_cfg = pde.get("time", {}) if pde else {}
    params = case_spec.get("parameters", {})
    out = case_spec["output"]
    grid = out["grid"]
    nx_out = grid["nx"]
    ny_out = grid["ny"]
    bbox = grid["bbox"]

    epsilon = float(params.get("epsilon", 0.01))
    mesh_N = int(params.get("mesh_resolution", 128))
    deg = int(params.get("element_degree", 1))
    dt_val = float(params.get("dt", time_cfg.get("dt", 0.01)))
    t0 = float(time_cfg.get("t0", 0.0))
    t_end = float(time_cfg.get("t_end", 0.4))
    time_scheme = params.get("time_scheme", time_cfg.get("scheme", "backward_euler"))
    newton_max_it = int(params.get("newton_max_it", 25))
    pc_choice = params.get("pc_type", "hypre")

    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_square(comm, mesh_N, mesh_N, cell_type=mesh.CellType.triangle)

    V = fem.functionspace(domain, ("Lagrange", deg))

    # Boundary condition: u=0 on all boundaries
    fdim = domain.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
    )
    dofs_bc = fem.locate_dofs_topological(V, fdim, boundary_facets)
    u_bc = fem.Function(V)
    u_bc.x.array[:] = 0.0
    bc = fem.dirichletbc(u_bc, dofs_bc)

    # Source term f (time-independent)
    x = ufl.SpatialCoordinate(domain)
    f_expr = 6.0 * (ufl.exp(-160.0 * ((x[0] - 0.3)**2 + (x[1] - 0.7)**2))
                    + 0.8 * ufl.exp(-160.0 * ((x[0] - 0.75)**2 + (x[1] - 0.35)**2)))

    # Initial condition
    u_n = fem.Function(V)
    def u0_func(x):
        return 0.3 * np.exp(-50.0 * ((x[0] - 0.3)**2 + (x[1] - 0.5)**2)) \
             + 0.3 * np.exp(-50.0 * ((x[0] - 0.7)**2 + (x[1] - 0.5)**2))
    u_n.interpolate(u0_func)

    u = fem.Function(V)
    u.x.array[:] = u_n.x.array[:]

    v = ufl.TestFunction(V)
    dt_c = fem.Constant(domain, PETSc.ScalarType(dt_val))
    eps_c = fem.Constant(domain, PETSc.ScalarType(epsilon))

    # Logistic reaction R(u) = u*(1-u)
    F = ((u - u_n) / dt_c) * v * ufl.dx \
        + eps_c * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx \
        + u * (1.0 - u) * v * ufl.dx \
        - f_expr * v * ufl.dx

    J = ufl.derivative(F, u)

    petsc_options = {
        "snes_type": "newtonls",
        "snes_linesearch_type": "bt",
        "snes_rtol": 1e-8,
        "snes_atol": 1e-10,
        "snes_max_it": newton_max_it,
        "ksp_type": "cg",
        "pc_type": pc_choice,
        "ksp_rtol": 1e-10,
    }

    prefix = "rd_" + uuid.uuid4().hex[:8] + "_"
    problem = petsc.NonlinearProblem(
        F, u, bcs=[bc], J=J,
        petsc_options_prefix=prefix,
        petsc_options=petsc_options,
    )

    u_initial_grid = _sample_grid(u_n, domain, nx_out, ny_out, bbox)

    n_steps = int(round((t_end - t0) / dt_val))
    nonlinear_iters = []
    total_lin_iters = 0

    for step in range(n_steps):
        try:
            problem.solve()
            try:
                n_it = problem.solver.getIterationNumber()
            except Exception:
                n_it = 1
            nonlinear_iters.append(int(n_it))
            try:
                ksp = problem.solver.getKSP()
                total_lin_iters += int(ksp.getIterationNumber())
            except Exception:
                pass
        except Exception:
            nonlinear_iters.append(-1)

        u.x.scatter_forward()
        u_n.x.array[:] = u.x.array[:]

    u_grid = _sample_grid(u, domain, nx_out, ny_out, bbox)

    return {
        "u": u_grid,
        "u_initial": u_initial_grid,
        "solver_info": {
            "mesh_resolution": mesh_N,
            "element_degree": deg,
            "ksp_type": "cg",
            "pc_type": pc_choice,
            "rtol": 1e-10,
            "iterations": int(total_lin_iters),
            "dt": dt_val,
            "n_steps": n_steps,
            "time_scheme": time_scheme,
            "nonlinear_iterations": nonlinear_iters,
        },
    }


def _sample_grid(u_func, domain, nx_out, ny_out, bbox):
    xmin, xmax, ymin, ymax = bbox
    xs = np.linspace(xmin, xmax, nx_out)
    ys = np.linspace(ymin, ymax, ny_out)
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

    vals = np.full(pts.shape[0], np.nan)
    if len(points_on_proc) > 0:
        res = u_func.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        vals[eval_map] = res.flatten()
    return vals.reshape(ny_out, nx_out)
