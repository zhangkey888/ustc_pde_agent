"""
Solver for reaction-diffusion equation with logistic reaction.
Case: reaction_diffusion_no_exact_localized_sink_logistic

PDE: ∂u/∂t - ε∇²u + ρ*u*(1-u) = f  in Ω × (0,T]
     u = 0                           on ∂Ω
     u(x,0) = 0.4 + 0.1*sin(πx)*sin(πy)

Parameters from config:
  ε = 0.06, ρ = 3.0
  f = 4*exp(-200*((x-0.4)² + (y-0.6)²)) - 2*exp(-200*((x-0.65)² + (y-0.35)²))
  t_end = 0.35, dt = 0.01, backward Euler
  Domain: [0,1]²
  Output: 75×75 grid
"""

import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc
import time
import json
import argparse
from pathlib import Path

ScalarType = PETSc.ScalarType


def solve(case_spec: dict = None) -> dict:
    """Solve reaction-diffusion equation with logistic reaction."""
    
    # ---- Parse case_spec or use hardcoded defaults ----
    # Hardcoded defaults from config.json
    epsilon = 0.06
    reaction_rho = 3.0
    t_end = 0.35
    dt_val = 0.01
    nx_out = 75
    ny_out = 75
    bc_value = 0.0
    
    if case_spec is not None:
        # Try to parse from case_spec (could be full config or just pde section)
        oracle_config = case_spec.get("oracle_config", case_spec)
        pde = oracle_config.get("pde", case_spec.get("pde", {}))
        
        # PDE parameters
        pde_params = pde.get("pde_params", {})
        epsilon = float(pde_params.get("epsilon", pde.get("epsilon", epsilon)))
        
        reaction_cfg = pde_params.get("reaction", {})
        if isinstance(reaction_cfg, dict):
            reaction_rho = float(reaction_cfg.get("rho", reaction_rho))
        else:
            reaction_rho = float(pde.get("reaction_rho", pde.get("rho", reaction_rho)))
        
        # Time parameters
        time_params = pde.get("time", {})
        t_end = float(time_params.get("t_end", t_end))
        dt_val = float(time_params.get("dt", dt_val))
        
        # Output grid
        output_cfg = oracle_config.get("output", case_spec.get("output", {}))
        grid_cfg = output_cfg.get("grid", output_cfg)
        nx_out = int(grid_cfg.get("nx", nx_out))
        ny_out = int(grid_cfg.get("ny", ny_out))
        
        # BC
        bc_cfg = oracle_config.get("bc", case_spec.get("bc", {}))
        dirichlet_cfg = bc_cfg.get("dirichlet", {})
        bc_value_str = dirichlet_cfg.get("value", str(bc_value))
        try:
            bc_value = float(bc_value_str)
        except (ValueError, TypeError):
            bc_value = 0.0
    
    # ---- Solve with adaptive mesh refinement ----
    resolutions = [64, 96, 128]
    element_degree = 1
    
    prev_norm = None
    final_result = None
    chosen_N = resolutions[0]
    
    for N in resolutions:
        result = _solve_rd(
            N, element_degree, epsilon, reaction_rho,
            t_end, dt_val, bc_value,
            nx_out, ny_out
        )
        
        if result is None:
            continue
            
        u_grid = result["u"]
        current_norm = np.nanmean(np.abs(u_grid))
        
        if prev_norm is not None:
            rel_change = abs(current_norm - prev_norm) / (abs(current_norm) + 1e-15)
            if rel_change < 0.005:
                chosen_N = N
                final_result = result
                break
        
        prev_norm = current_norm
        final_result = result
        chosen_N = N
    
    if final_result is None:
        raise RuntimeError("Solver failed at all resolutions")
    
    final_result["solver_info"]["mesh_resolution"] = chosen_N
    return final_result


def _solve_rd(N, element_degree, epsilon, reaction_rho,
              t_end, dt_val, bc_value, nx_out, ny_out):
    """Solve the reaction-diffusion PDE at a given mesh resolution."""
    
    comm = MPI.COMM_WORLD
    
    # Create mesh
    domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
    
    # Function space
    V = fem.functionspace(domain, ("Lagrange", element_degree))
    
    # Spatial coordinates
    x = ufl.SpatialCoordinate(domain)
    
    # Source term: f = 4*exp(-200*((x-0.4)^2 + (y-0.6)^2)) - 2*exp(-200*((x-0.65)^2 + (y-0.35)^2))
    f_ufl = (4.0 * ufl.exp(-200.0 * ((x[0] - 0.4)**2 + (x[1] - 0.6)**2))
             - 2.0 * ufl.exp(-200.0 * ((x[0] - 0.65)**2 + (x[1] - 0.35)**2)))
    
    # Initial condition: u0 = 0.4 + 0.1*sin(pi*x)*sin(pi*y)
    u_n = fem.Function(V, name="u_n")
    u_n.interpolate(lambda coords: 0.4 + 0.1 * np.sin(np.pi * coords[0]) * np.sin(np.pi * coords[1]))
    
    # Current solution (initialize from IC)
    u_sol = fem.Function(V, name="u")
    u_sol.x.array[:] = u_n.x.array[:]
    
    # Test function
    v = ufl.TestFunction(V)
    
    # Constants
    dt_c = fem.Constant(domain, ScalarType(dt_val))
    eps_c = fem.Constant(domain, ScalarType(epsilon))
    rho_c = fem.Constant(domain, ScalarType(reaction_rho))
    
    # Logistic reaction: R(u) = rho * u * (1 - u)
    R_u = rho_c * u_sol * (1.0 - u_sol)
    
    # Boundary conditions: u = bc_value on all boundaries
    tdim = domain.topology.dim
    fdim = tdim - 1
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda coords: np.ones(coords.shape[1], dtype=bool))
    dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    
    bc_func = fem.Function(V)
    bc_func.interpolate(lambda coords: np.full_like(coords[0], bc_value))
    bc = fem.dirichletbc(bc_func, dofs)
    bcs = [bc]
    
    # Variational form (backward Euler):
    # (u - u_n)/dt * v + eps * grad(u) . grad(v) + R(u) * v - f * v = 0
    F = ((u_sol - u_n) / dt_c) * v * ufl.dx \
        + eps_c * ufl.inner(ufl.grad(u_sol), ufl.grad(v)) * ufl.dx \
        + R_u * v * ufl.dx \
        - f_ufl * v * ufl.dx
    
    # Time stepping
    n_steps = int(np.ceil(t_end / dt_val))
    actual_dt = t_end / n_steps
    dt_c.value = actual_dt
    
    nonlinear_iterations = []
    total_linear_iters = 0
    
    # Setup nonlinear solver using SNES
    ksp_type = "gmres"
    pc_type = "ilu"
    
    problem = petsc.NonlinearProblem(
        F, u_sol,
        bcs=bcs,
        petsc_options_prefix="rd_",
        petsc_options={
            "snes_type": "newtonls",
            "snes_rtol": 1e-8,
            "snes_atol": 1e-10,
            "snes_max_it": 50,
            "ksp_type": ksp_type,
            "pc_type": pc_type,
            "ksp_rtol": 1e-8,
            "ksp_max_it": 1000,
        }
    )
    
    # Time loop
    t = 0.0
    for step in range(n_steps):
        t += actual_dt
        
        try:
            u_result = problem.solve()
            snes = problem.solver
            newton_its = snes.getIterationNumber()
            nonlinear_iterations.append(newton_its)
            
            ksp = snes.getKSP()
            lin_its = ksp.getIterationNumber()
            total_linear_iters += lin_its * max(newton_its, 1)
        except Exception as e:
            print(f"Warning: Solve failed at step {step}, t={t:.4f}: {e}")
            return None
        
        # Update previous solution
        u_n.x.array[:] = u_sol.x.array[:]
    
    # Evaluate on output grid
    u_grid = _evaluate_on_grid(domain, u_sol, nx_out, ny_out)
    
    # Evaluate initial condition on grid
    u_ic = fem.Function(V)
    u_ic.interpolate(lambda coords: 0.4 + 0.1 * np.sin(np.pi * coords[0]) * np.sin(np.pi * coords[1]))
    u_initial_grid = _evaluate_on_grid(domain, u_ic, nx_out, ny_out)
    
    solver_info = {
        "mesh_resolution": N,
        "element_degree": element_degree,
        "ksp_type": ksp_type,
        "pc_type": pc_type,
        "rtol": 1e-8,
        "iterations": total_linear_iters,
        "dt": actual_dt,
        "n_steps": n_steps,
        "time_scheme": "backward_euler",
        "nonlinear_iterations": nonlinear_iterations,
    }
    
    return {
        "u": u_grid,
        "solver_info": solver_info,
        "u_initial": u_initial_grid,
    }


def _evaluate_on_grid(domain, u_func, nx, ny):
    """Evaluate a FEM function on a uniform grid."""
    xs = np.linspace(0, 1, nx)
    ys = np.linspace(0, 1, ny)
    XX, YY = np.meshgrid(xs, ys, indexing='ij')
    
    points = np.zeros((3, nx * ny))
    points[0, :] = XX.flatten()
    points[1, :] = YY.flatten()
    
    bb_tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, points.T)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points.T)
    
    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    
    for i in range(nx * ny):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(points[:, i])
            cells_on_proc.append(links[0])
            eval_map.append(i)
    
    u_values = np.full(nx * ny, np.nan)
    if len(points_on_proc) > 0:
        pts_arr = np.array(points_on_proc)
        cells_arr = np.array(cells_on_proc, dtype=np.int32)
        vals = u_func.eval(pts_arr, cells_arr)
        u_values[eval_map] = vals.flatten()
    
    return u_values.reshape((nx, ny))


def main():
    """Main entry point for CLI execution."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", type=str, required=True, help="Output directory")
    args = parser.parse_args()
    
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    
    # Load case_spec if available
    case_spec = None
    case_file = outdir / "case_spec.json"
    if case_file.exists():
        with open(case_file) as f:
            case_spec = json.load(f)
    
    # Also check parent directories for config.json
    if case_spec is None:
        for parent in [Path("."), outdir, outdir.parent, outdir.parent.parent]:
            cfg = parent / "config.json"
            if cfg.exists():
                with open(cfg) as f:
                    case_spec = json.load(f)
                break
    
    t0 = time.time()
    result = solve(case_spec)
    t1 = time.time()
    
    u_grid = result["u"]
    solver_info = result["solver_info"]
    
    # Write solution.npz
    nx, ny = u_grid.shape
    x = np.linspace(0, 1, nx)
    y = np.linspace(0, 1, ny)
    np.savez(str(outdir / "solution.npz"), x=x, y=y, u=u_grid)
    
    # Write u.npy
    np.save(str(outdir / "u.npy"), u_grid)
    
    # Write u_initial.npy if available
    if "u_initial" in result and result["u_initial"] is not None:
        np.save(str(outdir / "u_initial.npy"), result["u_initial"])
    
    # Write meta.json
    meta = {
        "wall_time_sec": t1 - t0,
        "solver_info": solver_info,
    }
    with open(str(outdir / "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    
    print(f"Solve completed in {t1-t0:.2f}s")
    print(f"u shape: {u_grid.shape}")
    print(f"u range: [{np.nanmin(u_grid):.6f}, {np.nanmax(u_grid):.6f}]")
    print(f"u mean: {np.nanmean(u_grid):.6f}")
    print(f"NaN count: {np.isnan(u_grid).sum()}")


if __name__ == "__main__":
    main()
