import os
os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc


def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD

    # Output grid spec
    out = case_spec.get("output", {}).get("grid", {})
    nx_out = out.get("nx", 64)
    ny_out = out.get("ny", 64)
    bbox = out.get("bbox", [0.0, 1.0, 0.0, 1.0])

    # Time params
    pde = case_spec.get("pde", {})
    time_info = pde.get("time", {"t0": 0.0, "t_end": 0.25, "dt": 0.005})
    t0 = float(time_info.get("t0", 0.0))
    t_end = float(time_info.get("t_end", 0.25))
    dt_val = float(time_info.get("dt", 0.005))

    # Agent-selectable params (defaults tuned for accuracy)
    mesh_resolution = 128
    element_degree = 2
    dt_val = min(dt_val, 0.005)
    newton_rtol = 1e-8

    # Read epsilon from multiple possible locations in case_spec
    epsilon = 1e-3  # Default for Allen-Cahn
    for src in [case_spec.get("params", {}),
                pde.get("params", {}),
                pde.get("coefficients", {}),
                pde.get("coeffs", {}),
                case_spec.get("coefficients", {}),
                case_spec,
                pde]:
        if isinstance(src, dict):
            if "epsilon" in src:
                epsilon = float(src["epsilon"]); break
            if "eps" in src:
                epsilon = float(src["eps"]); break
            if "diffusion" in src:
                try: epsilon = float(src["diffusion"]); break
                except: pass
    import sys
    print(f"[solver] Using epsilon={epsilon}", file=sys.stderr)

    # Build mesh
    domain = mesh.create_unit_square(
        comm, mesh_resolution, mesh_resolution, cell_type=mesh.CellType.triangle
    )

    V = fem.functionspace(domain, ("Lagrange", element_degree))

    # Boundary conditions: u = 0 on all
    fdim = domain.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
    )
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    u_bc = fem.Function(V)
    u_bc.x.array[:] = 0.0
    bc = fem.dirichletbc(u_bc, boundary_dofs)

    # Source term f = 5*exp(-180*((x-0.35)^2 + (y-0.55)^2))
    x = ufl.SpatialCoordinate(domain)
    f_expr = 5.0 * ufl.exp(-180.0 * ((x[0] - 0.35) ** 2 + (x[1] - 0.55) ** 2))

    # Initial condition
    u_n = fem.Function(V)  # previous time step
    u_n.interpolate(
        lambda X: 0.1 * np.exp(-50.0 * ((X[0] - 0.5) ** 2 + (X[1] - 0.5) ** 2))
    )

    # Current solution
    u = fem.Function(V)
    u.x.array[:] = u_n.x.array[:]

    v = ufl.TestFunction(V)

    dt_c = fem.Constant(domain, PETSc.ScalarType(dt_val))
    eps_c = fem.Constant(domain, PETSc.ScalarType(epsilon))

    # Allen-Cahn reaction: R(u) = u^3 - u
    # Residual for backward Euler:
    # (u - u_n)/dt - eps * laplace(u) + (u^3 - u) = f
    F = (
        ufl.inner((u - u_n) / dt_c, v) * ufl.dx
        + eps_c * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
        + ufl.inner(u**3 - u, v) * ufl.dx
        - ufl.inner(f_expr, v) * ufl.dx
    )
    J = ufl.derivative(F, u)

    petsc_options = {
        "snes_type": "newtonls",
        "snes_linesearch_type": "bt",
        "snes_rtol": newton_rtol,
        "snes_atol": 1e-10,
        "snes_max_it": 30,
        "ksp_type": "gmres",
        "pc_type": "hypre",
        "pc_hypre_type": "boomeramg",
        "ksp_rtol": 1e-10,
    }

    problem = petsc.NonlinearProblem(
        F, u, bcs=[bc], J=J,
        petsc_options_prefix="rd_",
        petsc_options=petsc_options,
    )

    # Sample initial condition on the output grid (before stepping)
    xs = np.linspace(bbox[0], bbox[1], nx_out)
    ys = np.linspace(bbox[2], bbox[3], ny_out)
    XX, YY = np.meshgrid(xs, ys)
    pts = np.c_[XX.ravel(), YY.ravel(), np.zeros(nx_out * ny_out)]

    def sample_grid(func):
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
        vals = np.full((pts.shape[0],), np.nan)
        if len(points_on_proc) > 0:
            out = func.eval(np.array(points_on_proc),
                            np.array(cells_on_proc, dtype=np.int32))
            vals[eval_map] = out.flatten()
        return vals.reshape(ny_out, nx_out)

    u_initial_grid = sample_grid(u_n)

    # Time stepping
    t = t0
    n_steps = int(round((t_end - t0) / dt_val))
    nl_iters_list = []

    for step in range(n_steps):
        # Solve
        try:
            u_h = problem.solve()
        except Exception as e:
            # Retry with smaller dt could be implemented; for now raise
            raise
        # Try to grab iteration count from SNES
        try:
            it = problem.solver.getIterationNumber()
        except Exception:
            it = -1
        nl_iters_list.append(int(it))

        # Update previous solution
        u_n.x.array[:] = u.x.array[:]
        t += dt_val

    u_grid = sample_grid(u)

    solver_info = {
        "mesh_resolution": mesh_resolution,
        "element_degree": element_degree,
        "ksp_type": "gmres",
        "pc_type": "hypre",
        "pc_hypre_type": "boomeramg",
        "rtol": 1e-10,
        "iterations": int(sum(max(i, 0) for i in nl_iters_list)),
        "dt": dt_val,
        "n_steps": n_steps,
        "time_scheme": "backward_euler",
        "nonlinear_iterations": nl_iters_list,
        "epsilon": epsilon,
    }

    return {
        "u": u_grid,
        "u_initial": u_initial_grid,
        "solver_info": solver_info,
    }


if __name__ == "__main__":
    import time
    spec = {
        "pde": {
            "time": {"t0": 0.0, "t_end": 0.25, "dt": 0.005},
        },
        "output": {
            "grid": {"nx": 64, "ny": 64, "bbox": [0.0, 1.0, 0.0, 1.0]},
        },
    }
    t0 = time.time()
    result = solve(spec)
    print(f"Wall time: {time.time()-t0:.2f}s")
    print(f"u shape: {result['u'].shape}")
    print(f"u min/max: {result['u'].min():.4e} / {result['u'].max():.4e}")
    print(f"solver_info: {result['solver_info']}")

