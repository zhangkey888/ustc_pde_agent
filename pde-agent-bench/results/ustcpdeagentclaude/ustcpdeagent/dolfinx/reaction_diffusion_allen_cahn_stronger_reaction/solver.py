import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc


def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD

    params = case_spec.get("parameters", {}) if isinstance(case_spec.get("parameters", {}), dict) else {}
    epsilon = float(params.get("epsilon", 0.01))
    reaction_lambda = float(params.get("reaction_lambda", 1.0))

    tparams = None
    if "pde" in case_spec and isinstance(case_spec["pde"], dict) and "time" in case_spec["pde"]:
        tparams = case_spec["pde"]["time"]
    elif "time" in case_spec:
        tparams = case_spec["time"]
    if tparams is None:
        tparams = {}
    t0 = float(tparams.get("t0", 0.0))
    t_end = float(tparams.get("t_end", 0.1))
    dt_val = float(tparams.get("dt", 0.002))

    out = case_spec["output"]["grid"]
    nx_out = int(out["nx"])
    ny_out = int(out["ny"])
    bbox = out["bbox"]

    # Mesh & space
    N = 96
    degree = 1
    domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", degree))

    x = ufl.SpatialCoordinate(domain)
    t_const = fem.Constant(domain, PETSc.ScalarType(t0))
    dt_c = fem.Constant(domain, PETSc.ScalarType(dt_val))
    eps_c = fem.Constant(domain, PETSc.ScalarType(epsilon))
    lam_c = fem.Constant(domain, PETSc.ScalarType(reaction_lambda))

    # Manufactured solution: u = exp(-t)*(0.15 + 0.12*sin(2πx)*sin(2πy))
    E = ufl.exp(-t_const)
    S = ufl.sin(2 * ufl.pi * x[0]) * ufl.sin(2 * ufl.pi * x[1])
    a_const = 0.15
    b_const = 0.12
    u_exact_expr = E * (a_const + b_const * S)
    dudt_exact = -E * (a_const + b_const * S)
    laplace_u = E * b_const * (-8.0 * ufl.pi * ufl.pi) * S
    reaction_exact = lam_c * (u_exact_expr ** 3 - u_exact_expr)
    f_expr_src = dudt_exact - eps_c * laplace_u + reaction_exact

    bc_interp_expr = fem.Expression(u_exact_expr, V.element.interpolation_points)

    # Boundary DOFs
    fdim = domain.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
    )
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)

    u_bc = fem.Function(V)
    u_bc.interpolate(bc_interp_expr)
    bc = fem.dirichletbc(u_bc, boundary_dofs)

    # Initial condition
    u_n = fem.Function(V)
    t_const.value = t0
    u_n.interpolate(bc_interp_expr)
    u_initial_array = u_n.x.array.copy()

    # Current unknown
    u = fem.Function(V)
    u.x.array[:] = u_n.x.array[:]
    v = ufl.TestFunction(V)

    # Backward Euler residual
    F_form = (
        ((u - u_n) / dt_c) * v * ufl.dx
        + eps_c * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
        + lam_c * (u ** 3 - u) * v * ufl.dx
        - f_expr_src * v * ufl.dx
    )
    J = ufl.derivative(F_form, u)

    petsc_options = {
        "snes_type": "newtonls",
        "snes_rtol": 1e-10,
        "snes_atol": 1e-12,
        "snes_max_it": 30,
        "ksp_type": "preonly",
        "pc_type": "lu",
    }

    problem = petsc.NonlinearProblem(
        F_form, u, bcs=[bc], J=J,
        petsc_options_prefix="rd_",
        petsc_options=petsc_options,
    )

    # Time stepping
    n_steps = int(round((t_end - t0) / dt_val))
    if n_steps < 1:
        n_steps = 1
    # adjust dt to exactly hit t_end
    dt_exact = (t_end - t0) / n_steps
    dt_c.value = dt_exact
    dt_val = dt_exact

    nonlinear_iters = []
    total_linear_iters = 0

    t_cur = t0
    for step in range(n_steps):
        t_new = t_cur + dt_val
        t_const.value = t_new
        u_bc.interpolate(bc_interp_expr)
        try:
            problem.solve()
            snes = problem.solver
            try:
                nonlinear_iters.append(int(snes.getIterationNumber()))
                total_linear_iters += int(snes.getKSP().getIterationNumber())
            except Exception:
                nonlinear_iters.append(0)
        except Exception:
            # fallback: reduce not implemented, just record
            nonlinear_iters.append(-1)

        u_n.x.array[:] = u.x.array[:]
        t_cur = t_new

    # Sample onto uniform grid
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

    u_vals = np.full(pts.shape[0], np.nan)
    if len(points_on_proc) > 0:
        vals = u.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u_vals[eval_map] = vals.flatten()
    u_grid = u_vals.reshape(ny_out, nx_out)

    # Initial grid
    u_init_func = fem.Function(V)
    u_init_func.x.array[:] = u_initial_array
    u_init_vals = np.full(pts.shape[0], np.nan)
    if len(points_on_proc) > 0:
        vals_i = u_init_func.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u_init_vals[eval_map] = vals_i.flatten()
    u_init_grid = u_init_vals.reshape(ny_out, nx_out)

    return {
        "u": u_grid,
        "u_initial": u_init_grid,
        "solver_info": {
            "mesh_resolution": N,
            "element_degree": degree,
            "ksp_type": "preonly",
            "pc_type": "lu",
            "rtol": 1e-10,
            "iterations": int(total_linear_iters),
            "dt": float(dt_val),
            "n_steps": int(n_steps),
            "time_scheme": "backward_euler",
            "nonlinear_iterations": nonlinear_iters,
        },
    }


if __name__ == "__main__":
    import time
    case_spec = {
        "pde": {"time": {"t0": 0.0, "t_end": 0.1, "dt": 0.002}},
        "parameters": {"epsilon": 0.01, "reaction_lambda": 1.0},
        "output": {"grid": {"nx": 64, "ny": 64, "bbox": [0.0, 1.0, 0.0, 1.0]}},
    }
    t_start = time.time()
    result = solve(case_spec)
    wall = time.time() - t_start
    print("Wall time:", wall)
    print("u shape:", result["u"].shape)
    xs = np.linspace(0, 1, 64)
    ys = np.linspace(0, 1, 64)
    XX, YY = np.meshgrid(xs, ys)
    u_ex = np.exp(-0.1) * (0.15 + 0.12 * np.sin(2 * np.pi * XX) * np.sin(2 * np.pi * YY))
    err = np.sqrt(np.mean((result["u"] - u_ex) ** 2))
    print("RMSE:", err)
    print("solver_info:", result["solver_info"])
