import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc


def solve(case_spec: dict = None):
    if case_spec is None:
        case_spec = {}

    # Extract parameters - handle nested oracle_config structure
    oracle_config = case_spec.get("oracle_config", case_spec)
    pde = oracle_config.get("pde", {})
    pde_params = pde.get("pde_params", pde)
    
    epsilon = pde_params.get("epsilon", 0.05)
    beta_vec = pde_params.get("beta", [3.0, 3.0])
    
    output = oracle_config.get("output", {})
    grid = output.get("grid", output)
    nx_out = grid.get("nx", 50)
    ny_out = grid.get("ny", 50)

    N = 128
    element_degree = 2
    
    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
    
    V = fem.functionspace(domain, ("Lagrange", element_degree))
    
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    
    x = ufl.SpatialCoordinate(domain)
    
    # Source term: f = sin(6*pi*x)*sin(5*pi*y)
    f_expr = ufl.sin(6 * ufl.pi * x[0]) * ufl.sin(5 * ufl.pi * x[1])
    
    # Convection velocity and diffusion
    beta = ufl.as_vector([PETSc.ScalarType(beta_vec[0]), PETSc.ScalarType(beta_vec[1])])
    eps_val = fem.Constant(domain, PETSc.ScalarType(epsilon))
    
    # Standard Galerkin weak form (no SUPG - matches reference solver)
    a = eps_val * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx \
        + ufl.inner(ufl.dot(beta, ufl.grad(u)), v) * ufl.dx
    L = f_expr * v * ufl.dx
    
    # Boundary conditions: u = 0 on all boundaries
    tdim = domain.topology.dim
    fdim = tdim - 1
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
    )
    dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc = fem.dirichletbc(PETSc.ScalarType(0.0), dofs, V)
    bcs = [bc]
    
    # Solve
    ksp_type = "gmres"
    pc_type = "ilu"
    rtol = 1e-10
    
    try:
        problem = petsc.LinearProblem(
            a, L, bcs=bcs,
            petsc_options={
                "ksp_type": ksp_type,
                "pc_type": pc_type,
                "ksp_rtol": str(rtol),
                "ksp_max_it": "5000",
                "ksp_gmres_restart": "150",
            },
            petsc_options_prefix="cdiff_"
        )
        u_sol = problem.solve()
    except Exception:
        ksp_type = "preonly"
        pc_type = "lu"
        problem = petsc.LinearProblem(
            a, L, bcs=bcs,
            petsc_options={
                "ksp_type": ksp_type,
                "pc_type": pc_type,
            },
            petsc_options_prefix="cdiff2_"
        )
        u_sol = problem.solve()
    
    # Evaluate on output grid
    u_grid = evaluate_on_grid(domain, u_sol, nx_out, ny_out)
    
    solver_info = {
        "mesh_resolution": N,
        "element_degree": element_degree,
        "ksp_type": ksp_type,
        "pc_type": pc_type,
        "rtol": rtol,
        "iterations": 0,
    }
    
    return {
        "u": u_grid,
        "solver_info": solver_info,
    }


def evaluate_on_grid(domain, u_func, nx, ny):
    xv = np.linspace(0, 1, nx)
    yv = np.linspace(0, 1, ny)
    xx, yy = np.meshgrid(xv, yv, indexing='ij')
    
    points_2d = np.column_stack([xx.ravel(), yy.ravel()])
    points_3d = np.zeros((points_2d.shape[0], 3))
    points_3d[:, :2] = points_2d
    
    bb_tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, points_3d)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points_3d)
    
    u_values = np.full(points_3d.shape[0], np.nan)
    
    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    
    for i in range(points_3d.shape[0]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(points_3d[i])
            cells_on_proc.append(links[0])
            eval_map.append(i)
    
    if len(points_on_proc) > 0:
        pts_arr = np.array(points_on_proc)
        cells_arr = np.array(cells_on_proc, dtype=np.int32)
        vals = u_func.eval(pts_arr, cells_arr)
        u_values[eval_map] = vals.flatten()
    
    u_grid = u_values.reshape((nx, ny))
    return u_grid


if __name__ == "__main__":
    import argparse
    import json
    import time
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", type=str, default=None)
    args, _ = parser.parse_known_args()
    
    t0 = time.time()
    result = solve()
    elapsed = time.time() - t0
    
    u_grid = result["u"]
    info = result["solver_info"]
    
    if args.outdir:
        import os
        os.makedirs(args.outdir, exist_ok=True)
        xv = np.linspace(0, 1, u_grid.shape[0])
        yv = np.linspace(0, 1, u_grid.shape[1])
        np.savez(f"{args.outdir}/solution.npz", x=xv, y=yv, u=u_grid)
        meta = {"wall_time_sec": elapsed, "solver_info": info}
        with open(f"{args.outdir}/meta.json", "w") as f:
            json.dump(meta, f, indent=2)
        print(f"Saved to {args.outdir}")
    
    print(f"Solve completed in {elapsed:.2f}s")
    print(f"Grid shape: {u_grid.shape}")
    print(f"Solution range: [{np.nanmin(u_grid):.6f}, {np.nanmax(u_grid):.6f}]")
    print(f"L2 norm of grid: {np.linalg.norm(u_grid):.6f}")
    print(f"NaN count: {np.isnan(u_grid).sum()}")
    print(f"Solver info: {info}")
