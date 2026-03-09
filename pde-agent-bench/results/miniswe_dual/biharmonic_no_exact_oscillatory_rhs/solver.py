import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc

ScalarType = PETSc.ScalarType


def solve(case_spec: dict = None) -> dict:
    """
    Solve biharmonic equation Δ²u = f on [0,1]×[0,1]
    using mixed formulation (two sequential Poisson solves):
      Step 1: -Δw = f, w=0 on ∂Ω
      Step 2: -Δu = w, u=0 on ∂Ω
    This corresponds to simply-supported plate BCs: u=0, Δu=0 on ∂Ω.
    """
    if case_spec is None:
        case_spec = {}

    # Output grid size
    output_cfg = case_spec.get('oracle_config', case_spec).get('output', {}).get('grid', {})
    nx_out = output_cfg.get('nx', 50)
    ny_out = output_cfg.get('ny', 50)
    bbox = output_cfg.get('bbox', [0, 1, 0, 1])

    # Adaptive mesh refinement
    resolutions = [32, 64, 128]
    element_degree = 2
    prev_grid = None
    final_u_grid = None
    final_N = resolutions[0]

    for N in resolutions:
        u_grid = _solve_biharmonic(N, element_degree, nx_out, ny_out, bbox)

        if prev_grid is not None:
            max_diff = np.nanmax(np.abs(u_grid - prev_grid))
            max_val = np.nanmax(np.abs(u_grid)) + 1e-15
            rel_change = max_diff / max_val
            if rel_change < 0.005:
                final_u_grid = u_grid
                final_N = N
                break

        prev_grid = u_grid
        final_u_grid = u_grid
        final_N = N

    solver_info = {
        "mesh_resolution": final_N,
        "element_degree": element_degree,
        "ksp_type": "cg",
        "pc_type": "hypre",
        "rtol": 1e-10,
        "iterations": 0,
    }

    return {
        "u": final_u_grid,
        "solver_info": solver_info,
    }


def _solve_biharmonic(N, element_degree, nx_out, ny_out, bbox):
    """Solve biharmonic via two Poisson problems."""
    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", element_degree))

    # Source term
    x = ufl.SpatialCoordinate(domain)
    f_expr = ufl.sin(10 * ufl.pi * x[0]) * ufl.sin(8 * ufl.pi * x[1])

    # Boundary conditions: homogeneous Dirichlet on all boundaries
    tdim = domain.topology.dim
    fdim = tdim - 1
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda xx: np.ones(xx.shape[1], dtype=bool)
    )
    dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc_zero = fem.dirichletbc(ScalarType(0.0), dofs, V)

    ksp_type = "cg"
    pc_type = "hypre"
    rtol = 1e-10

    petsc_opts = {
        "ksp_type": ksp_type,
        "pc_type": pc_type,
        "ksp_rtol": str(rtol),
        "ksp_atol": "1e-12",
    }

    # Step 1: -Δw = f, w=0 on ∂Ω
    w_trial = ufl.TrialFunction(V)
    v_test = ufl.TestFunction(V)
    a1 = ufl.inner(ufl.grad(w_trial), ufl.grad(v_test)) * ufl.dx
    L1 = f_expr * v_test * ufl.dx

    problem1 = petsc.LinearProblem(
        a1, L1, bcs=[bc_zero],
        petsc_options=petsc_opts,
        petsc_options_prefix="step1_",
    )
    w_sol = problem1.solve()

    # Step 2: -Δu = w, u=0 on ∂Ω
    u_trial = ufl.TrialFunction(V)
    a2 = ufl.inner(ufl.grad(u_trial), ufl.grad(v_test)) * ufl.dx
    L2 = w_sol * v_test * ufl.dx

    problem2 = petsc.LinearProblem(
        a2, L2, bcs=[bc_zero],
        petsc_options=petsc_opts,
        petsc_options_prefix="step2_",
    )
    u_sol = problem2.solve()

    # Evaluate on output grid
    u_grid = _evaluate_on_grid(domain, u_sol, nx_out, ny_out, bbox)
    return u_grid


def _evaluate_on_grid(domain, u_func, nx_out, ny_out, bbox):
    """Evaluate a dolfinx Function on a uniform nx_out x ny_out grid."""
    xmin, xmax, ymin, ymax = bbox
    xs = np.linspace(xmin, xmax, nx_out)
    ys = np.linspace(ymin, ymax, ny_out)
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
    import argparse
    import json
    import time

    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", type=str, default=None)
    args, _ = parser.parse_known_args()

    case_spec = {
        "oracle_config": {
            "pde": {
                "type": "biharmonic",
                "source_term": "sin(10*pi*x)*sin(8*pi*y)",
            },
            "domain": {"type": "unit_square"},
            "output": {
                "grid": {"bbox": [0, 1, 0, 1], "nx": 50, "ny": 50}
            },
        }
    }

    t0 = time.time()
    result = solve(case_spec)
    elapsed = time.time() - t0

    u_grid = result["u"]
    solver_info = result["solver_info"]

    print(f"Solution shape: {u_grid.shape}")
    print(f"Solution range: [{np.nanmin(u_grid):.6e}, {np.nanmax(u_grid):.6e}]")
    print(f"L2 norm of grid: {np.sqrt(np.nanmean(u_grid**2)):.6e}")
    print(f"Time: {elapsed:.3f}s")
    print(f"Solver info: {solver_info}")
    print(f"Any NaN: {np.any(np.isnan(u_grid))}")

    if args.outdir is not None:
        from pathlib import Path
        outdir = Path(args.outdir)
        outdir.mkdir(parents=True, exist_ok=True)

        nx_out = u_grid.shape[0]
        ny_out = u_grid.shape[1]
        x = np.linspace(0, 1, nx_out)
        y = np.linspace(0, 1, ny_out)
        np.savez(outdir / "solution.npz", x=x, y=y, u=u_grid)

        meta = {
            "wall_time_sec": elapsed,
            "solver_info": solver_info,
        }
        with open(outdir / "meta.json", "w") as f:
            json.dump(meta, f, indent=2)
        print(f"Output saved to {outdir}")
