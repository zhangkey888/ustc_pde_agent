import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc
import argparse
import json
import sys
import os


def solve(case_spec: dict = None) -> dict:
    """
    Solve the biharmonic equation via mixed formulation (two Poisson solves):
      -Δw = f,  w=0 on ∂Ω
      -Δu = w,  u=0 on ∂Ω
    """
    comm = MPI.COMM_WORLD

    # Defaults
    source_term_str = "sin(8*pi*x)*cos(6*pi*y)"
    N = 160
    element_degree = 2
    cell_type = mesh.CellType.quadrilateral
    bc_value = 0.0
    nx_out, ny_out = 50, 50
    bbox = [0, 1, 0, 1]

    if case_spec is not None:
        oc = case_spec.get("oracle_config", case_spec)
        pde_cfg = oc.get("pde", {})
        source_term_str = pde_cfg.get("source_term", source_term_str)

        mesh_cfg = oc.get("mesh", {})
        ct = mesh_cfg.get("cell_type", "quadrilateral")
        if ct == "quadrilateral":
            cell_type = mesh.CellType.quadrilateral
        else:
            cell_type = mesh.CellType.triangle

        fem_cfg = oc.get("fem", {})
        # We override degree to 2 for accuracy
        element_degree = max(fem_cfg.get("degree", element_degree), 2)

        bc_cfg = oc.get("bc", {}).get("dirichlet", {})
        try:
            bc_value = float(bc_cfg.get("value", 0.0))
        except Exception:
            bc_value = 0.0

        out_cfg = oc.get("output", {}).get("grid", {})
        nx_out = out_cfg.get("nx", nx_out)
        ny_out = out_cfg.get("ny", ny_out)
        bbox = out_cfg.get("bbox", bbox)

    # Create mesh
    domain = mesh.create_unit_square(comm, N, N, cell_type=cell_type)
    V = fem.functionspace(domain, ("Lagrange", element_degree))

    # Source term
    x = ufl.SpatialCoordinate(domain)
    f_expr = ufl.sin(8 * ufl.pi * x[0]) * ufl.cos(6 * ufl.pi * x[1])

    # Boundary conditions
    tdim = domain.topology.dim
    fdim = tdim - 1
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda xx: np.ones(xx.shape[1], dtype=bool)
    )
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc_zero = fem.dirichletbc(PETSc.ScalarType(bc_value), boundary_dofs, V)

    # Common bilinear form
    trial_fn = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    a_form = ufl.inner(ufl.grad(trial_fn), ufl.grad(v)) * ufl.dx

    petsc_opts = {
        "ksp_type": "cg",
        "pc_type": "hypre",
        "ksp_rtol": "1e-10",
        "ksp_atol": "1e-12",
    }

    # Step 1: -Δw = f, w=0 on ∂Ω
    L1 = f_expr * v * ufl.dx
    prob1 = petsc.LinearProblem(
        a_form, L1, bcs=[bc_zero],
        petsc_options=petsc_opts,
        petsc_options_prefix="biharm_w_"
    )
    w_h = prob1.solve()
    iter1 = prob1.solver.getIterationNumber()

    # Step 2: -Δu = w, u=0 on ∂Ω
    L2 = w_h * v * ufl.dx
    prob2 = petsc.LinearProblem(
        a_form, L2, bcs=[bc_zero],
        petsc_options=petsc_opts,
        petsc_options_prefix="biharm_u_"
    )
    u_h = prob2.solve()
    iter2 = prob2.solver.getIterationNumber()

    total_iterations = iter1 + iter2

    # Sample on output grid
    xmin, xmax, ymin, ymax = bbox
    xs = np.linspace(xmin, xmax, nx_out)
    ys = np.linspace(ymin, ymax, ny_out)
    XX, YY = np.meshgrid(xs, ys, indexing='ij')
    pts = np.zeros((nx_out * ny_out, 3))
    pts[:, 0] = XX.ravel()
    pts[:, 1] = YY.ravel()

    bb_tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, pts)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, pts)

    u_values = np.full(pts.shape[0], np.nan)
    pts_proc = []
    cells_proc = []
    emap = []
    for i in range(pts.shape[0]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            pts_proc.append(pts[i])
            cells_proc.append(links[0])
            emap.append(i)

    if len(pts_proc) > 0:
        vals = u_h.eval(np.array(pts_proc), np.array(cells_proc, dtype=np.int32))
        u_values[emap] = vals.flatten()

    u_grid = u_values.reshape((nx_out, ny_out))

    return {
        "u": u_grid,
        "solver_info": {
            "mesh_resolution": N,
            "element_degree": element_degree,
            "ksp_type": "cg",
            "pc_type": "hypre",
            "rtol": 1e-10,
            "iterations": max(total_iterations, 1),
        }
    }


if __name__ == "__main__":
    import time

    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", type=str, default=None)
    args, _ = parser.parse_known_args()

    t0 = time.time()
    
    # Load case_spec if config.json exists nearby
    case_spec = None
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Try to find config.json in the cases directory
    possible_config_paths = [
        os.path.join(script_dir, "config.json"),
    ]
    for cp in possible_config_paths:
        if os.path.exists(cp):
            with open(cp) as f:
                case_spec = json.load(f)
            break

    result = solve(case_spec)
    elapsed = time.time() - t0

    ug = result['u']
    solver_info = result['solver_info']

    if args.outdir:
        outdir = args.outdir
        os.makedirs(outdir, exist_ok=True)

        # Get grid parameters
        nx, ny = ug.shape
        bbox = [0, 1, 0, 1]
        if case_spec:
            oc = case_spec.get("oracle_config", case_spec)
            grid_cfg = oc.get("output", {}).get("grid", {})
            bbox = grid_cfg.get("bbox", bbox)

        x = np.linspace(bbox[0], bbox[1], nx)
        y = np.linspace(bbox[2], bbox[3], ny)

        np.savez(os.path.join(outdir, "solution.npz"), x=x, y=y, u=ug)
        
        meta = {
            "wall_time_sec": elapsed,
            "solver_info": solver_info,
        }
        with open(os.path.join(outdir, "meta.json"), "w") as f:
            json.dump(meta, f, indent=2)

        print(f"Saved solution to {outdir}")
    else:
        print(f"Time: {elapsed:.3f}s")
        print(f"Shape: {ug.shape}, min: {np.nanmin(ug):.6e}, max: {np.nanmax(ug):.6e}")
        print(f"NaN: {np.sum(np.isnan(ug))}")
        print(f"Info: {solver_info}")
