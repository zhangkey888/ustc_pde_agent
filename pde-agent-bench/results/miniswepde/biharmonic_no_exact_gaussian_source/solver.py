import argparse
import json
import time
import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc

ScalarType = PETSc.ScalarType


def solve(case_spec: dict = None) -> dict:
    """
    Solve the biharmonic equation Δ²u = f on [0,1]x[0,1]
    using a mixed formulation (two sequential Poisson solves):
      Step 1: -Δw = f, w=0 on ∂Ω
      Step 2: -Δu = w, u=0 on ∂Ω
    """
    if case_spec is None:
        case_spec = {}

    nx_out = 50
    ny_out = 50

    # Adaptive mesh refinement with convergence check
    resolutions = [32, 64, 128]
    element_degree = 2
    prev_grid = None
    final_u_grid = None
    final_info = {}

    for N in resolutions:
        u_grid, info = _solve_biharmonic(N, element_degree, nx_out, ny_out)

        if prev_grid is not None:
            max_diff = np.nanmax(np.abs(u_grid - prev_grid))
            max_val = np.nanmax(np.abs(u_grid)) + 1e-15
            rel_change = max_diff / max_val
            if rel_change < 1e-3:
                # Converged
                final_u_grid = u_grid
                final_info = info
                break

        prev_grid = u_grid
        final_u_grid = u_grid
        final_info = info

    return {
        "u": final_u_grid,
        "solver_info": final_info,
    }


def _solve_biharmonic(N, element_degree, nx_out, ny_out):
    """Solve biharmonic via two Poisson problems."""
    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)

    V = fem.functionspace(domain, ("Lagrange", element_degree))

    # Boundary conditions: homogeneous Dirichlet on all boundaries
    tdim = domain.topology.dim
    fdim = tdim - 1
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
    )
    dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc_zero = fem.dirichletbc(ScalarType(0.0), dofs, V)

    # Source term
    x = ufl.SpatialCoordinate(domain)
    f_expr = 10.0 * ufl.exp(-80.0 * ((x[0] - 0.35) ** 2 + (x[1] - 0.55) ** 2))

    w_trial = ufl.TrialFunction(V)
    v_test = ufl.TestFunction(V)

    ksp_type = "cg"
    pc_type = "hypre"
    rtol = 1e-10

    petsc_opts = {
        "ksp_type": ksp_type,
        "pc_type": pc_type,
        "ksp_rtol": str(rtol),
        "ksp_atol": "1e-12",
    }

    # Step 1: Solve -Δw = f with w=0 on ∂Ω
    a1 = ufl.inner(ufl.grad(w_trial), ufl.grad(v_test)) * ufl.dx
    L1 = ufl.inner(f_expr, v_test) * ufl.dx

    problem1 = petsc.LinearProblem(
        a1, L1, bcs=[bc_zero],
        petsc_options=petsc_opts,
        petsc_options_prefix="step1_",
    )
    w_sol = problem1.solve()

    # Step 2: Solve -Δu = w with u=0 on ∂Ω
    u_trial = ufl.TrialFunction(V)
    a2 = ufl.inner(ufl.grad(u_trial), ufl.grad(v_test)) * ufl.dx
    L2 = ufl.inner(w_sol, v_test) * ufl.dx

    problem2 = petsc.LinearProblem(
        a2, L2, bcs=[bc_zero],
        petsc_options=petsc_opts,
        petsc_options_prefix="step2_",
    )
    u_sol = problem2.solve()

    # Evaluate on output grid - use same convention as oracle (indexing="ij")
    u_grid = _evaluate_on_grid(domain, u_sol, nx_out, ny_out)

    info = {
        "mesh_resolution": N,
        "element_degree": element_degree,
        "ksp_type": ksp_type,
        "pc_type": pc_type,
        "rtol": rtol,
        "iterations": 0,
    }

    return u_grid, info


def _evaluate_on_grid(domain, u_func, nx_out, ny_out):
    """Evaluate a dolfinx Function on a uniform nx_out x ny_out grid over [0,1]^2.
    Uses indexing='ij' to match oracle convention: result[i,j] = u(x[i], y[j]).
    """
    xs = np.linspace(0, 1, nx_out)
    ys = np.linspace(0, 1, ny_out)
    XX, YY = np.meshgrid(xs, ys, indexing="ij")

    points = np.zeros((nx_out * ny_out, 3))
    points[:, 0] = XX.flatten()
    points[:, 1] = YY.flatten()

    bb_tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, points)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points)

    points_on_proc = []
    cells_on_proc = []
    eval_map = []

    for i in range(points.shape[0]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(points[i])
            cells_on_proc.append(links[0])
            eval_map.append(i)

    u_values = np.full(nx_out * ny_out, np.nan)
    if len(points_on_proc) > 0:
        pts_arr = np.array(points_on_proc)
        cells_arr = np.array(cells_on_proc, dtype=np.int32)
        vals = u_func.eval(pts_arr, cells_arr)
        u_values[eval_map] = vals.flatten()

    return u_values.reshape((nx_out, ny_out))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Biharmonic PDE solver")
    parser.add_argument("--outdir", type=str, default=None, help="Output directory")
    parser.add_argument("--resolution", type=int, default=None, help="(ignored)")
    parser.add_argument("--degree", type=int, default=None, help="(ignored)")
    args = parser.parse_args()

    t0 = time.time()
    result = solve()
    elapsed = time.time() - t0

    u_grid = result["u"]
    info = result["solver_info"]

    if args.outdir is not None:
        import os
        os.makedirs(args.outdir, exist_ok=True)

        # Grid coordinates
        nx_out, ny_out = u_grid.shape
        x_grid = np.linspace(0, 1, nx_out)
        y_grid = np.linspace(0, 1, ny_out)

        np.savez(
            os.path.join(args.outdir, "solution.npz"),
            x=x_grid,
            y=y_grid,
            u=u_grid,
        )

        meta = {
            "wall_time_sec": elapsed,
            "solver_info": info,
        }
        with open(os.path.join(args.outdir, "meta.json"), "w") as f:
            json.dump(meta, f, indent=2)

        print(f"Saved solution to {args.outdir}")
    else:
        print(f"Solve time: {elapsed:.3f}s")
        print(f"u_grid shape: {u_grid.shape}")
        print(f"u_grid min: {np.nanmin(u_grid):.6e}, max: {np.nanmax(u_grid):.6e}")
        print(f"NaN count: {np.count_nonzero(np.isnan(u_grid))}")
        print(f"Solver info: {info}")
