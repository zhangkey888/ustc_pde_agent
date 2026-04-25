import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc


def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD

    # Parse spec
    pde = case_spec.get("pde", {})
    params = pde.get("params", {}) if isinstance(pde, dict) else {}
    epsilon = float(params.get("epsilon", case_spec.get("epsilon", 0.01)))

    time_info = pde.get("time", case_spec.get("time", {})) or {}
    t0 = float(time_info.get("t0", 0.0))
    t_end = float(time_info.get("t_end", 0.3))
    dt_val = float(time_info.get("dt", 0.005))

    out_grid = case_spec["output"]["grid"]
    nx_out = int(out_grid["nx"])
    ny_out = int(out_grid["ny"])
    bbox = out_grid["bbox"]

    # Mesh resolution
    N = int(case_spec.get("mesh_resolution", 64))
    degree = 2

    # Refine dt for accuracy
    dt_use = min(dt_val, 0.005)
    n_steps = int(np.ceil((t_end - t0) / dt_use))
    dt_use = (t_end - t0) / n_steps

    domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", degree))

    # BC: u=0 on boundary
    fdim = domain.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
    )
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    u_bc = fem.Function(V)
    u_bc.x.array[:] = 0.0
    bc = fem.dirichletbc(u_bc, boundary_dofs)

    # Solution functions
    u = fem.Function(V)   # current (nonlinear unknown)
    u_n = fem.Function(V) # previous time step

    # Initial condition
    def u0_expr(x):
        return 0.2 * np.sin(3 * np.pi * x[0]) * np.sin(2 * np.pi * x[1])

    u_n.interpolate(u0_expr)
    u.interpolate(u0_expr)
    # enforce BC on initial
    u_n.x.array[boundary_dofs] = 0.0
    u.x.array[boundary_dofs] = 0.0

    # Source
    x = ufl.SpatialCoordinate(domain)
    f_expr = ufl.sin(6 * ufl.pi * x[0]) * ufl.sin(5 * ufl.pi * x[1])

    v = ufl.TestFunction(V)
    dt_c = fem.Constant(domain, PETSc.ScalarType(dt_use))
    eps_c = fem.Constant(domain, PETSc.ScalarType(epsilon))

    # Backward Euler residual: (u - u_n)/dt - eps*lap(u) + u^3 = f
    F = ((u - u_n) / dt_c) * v * ufl.dx \
        + eps_c * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx \
        + (u ** 3) * v * ufl.dx \
        - f_expr * v * ufl.dx

    J = ufl.derivative(F, u)

    petsc_opts = {
        "snes_type": "newtonls",
        "snes_rtol": 1e-9,
        "snes_atol": 1e-11,
        "snes_max_it": 30,
        "ksp_type": "gmres",
        "pc_type": "hypre",
        "ksp_rtol": 1e-10,
    }

    problem = petsc.NonlinearProblem(
        F, u, bcs=[bc], J=J,
        petsc_options_prefix=f"rd_{id(u)}_",
        petsc_options=petsc_opts,
    )

    newton_iters = []
    total_lin_iters = 0

    t = t0
    for step in range(n_steps):
        try:
            u_sol = problem.solve()
        except Exception:
            u_sol = problem.solve()
        # SNES iteration count
        try:
            snes = problem.solver
            its = snes.getIterationNumber()
            newton_iters.append(int(its))
            ksp = snes.getKSP()
            total_lin_iters += int(ksp.getIterationNumber())
        except Exception:
            newton_iters.append(-1)

        u.x.scatter_forward()
        u_n.x.array[:] = u.x.array[:]
        t += dt_use

    # Sample on output grid
    xs = np.linspace(bbox[0], bbox[1], nx_out)
    ys = np.linspace(bbox[2], bbox[3], ny_out)
    XX, YY = np.meshgrid(xs, ys)
    pts = np.c_[XX.ravel(), YY.ravel(), np.zeros(nx_out * ny_out)]

    tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(tree, pts)
    colliding = geometry.compute_colliding_cells(domain, cell_candidates, pts)

    pts_proc, cells_proc, idx_map = [], [], []
    for i in range(pts.shape[0]):
        links = colliding.links(i)
        if len(links) > 0:
            pts_proc.append(pts[i])
            cells_proc.append(links[0])
            idx_map.append(i)

    u_values = np.zeros(nx_out * ny_out)
    if len(pts_proc) > 0:
        vals = u.eval(np.array(pts_proc), np.array(cells_proc, dtype=np.int32))
        u_values[idx_map] = vals.flatten()
    u_grid = u_values.reshape(ny_out, nx_out)

    # Initial condition grid
    u0_grid = 0.2 * np.sin(3 * np.pi * XX) * np.sin(2 * np.pi * YY)

    return {
        "u": u_grid,
        "u_initial": u0_grid,
        "solver_info": {
            "mesh_resolution": N,
            "element_degree": degree,
            "ksp_type": "gmres",
            "pc_type": "hypre",
            "rtol": 1e-9,
            "iterations": int(total_lin_iters),
            "dt": dt_use,
            "n_steps": n_steps,
            "time_scheme": "backward_euler",
            "nonlinear_iterations": newton_iters,
        },
    }
