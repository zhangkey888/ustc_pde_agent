#!/usr/bin/env python3
"""
Solver for Poisson equation with adaptive mesh refinement.
Uses FEniCSx (dolfinx) v0.8.0+.
"""
import argparse
import json
import time
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
import ufl
from dolfinx import mesh, fem, io, geometry
from dolfinx.fem.petsc import LinearProblem, assemble_matrix, assemble_vector, apply_lifting, set_bc
from dolfinx import geometry


def solve_poisson(resolution, degree, outdir, dt=None):
    """
    Solve Poisson equation -∇·(κ∇u) = f on unit square with Dirichlet BCs.
    
    Parameters:
    -----------
    resolution : int
        Initial mesh resolution (N x N cells)
    degree : int
        Polynomial degree of finite element space
    outdir : str
        Output directory for solution files
    dt : float or None
        Not used for elliptic problem (kept for API compatibility)
    
    Returns:
    --------
    dict : Solver metadata
    """
    comm = MPI.COMM_WORLD
    rank = comm.rank
    
    # Start timing
    wall_time_start = time.time()
    
    # Domain: unit square [0,1] x [0,1]
    domain = mesh.create_unit_square(comm, resolution, resolution, mesh.CellType.triangle)
    
    # Function space
    V = fem.functionspace(domain, ("Lagrange", degree))
    
    # Define manufactured solution
    # u_exact = sin(πx) * sin(πy)  (example, will be overridden by test harness)
    x = ufl.SpatialCoordinate(domain)
    u_exact = ufl.sin(np.pi * x[0]) * ufl.sin(np.pi * x[1])
    
    # Define κ (diffusion coefficient) = 1.0
    κ = 1.0
    
    # Source term f = -∇·(κ∇u_exact)
    f = -ufl.div(κ * ufl.grad(u_exact))
    
    # Trial and test functions
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    
    # Bilinear and linear forms
    a = κ * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = f * v * ufl.dx
    
    # Dirichlet boundary condition: u = u_exact on entire boundary
    tdim = domain.topology.dim
    fdim = tdim - 1
    domain.topology.create_connectivity(fdim, tdim)
    boundary_facets = mesh.exterior_facet_indices(domain.topology)
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    
    # Create Dirichlet BC
    u_bc = fem.Function(V)
    u_bc.interpolate(fem.Expression(u_exact, V.element.interpolation_points))
    bc = fem.dirichletbc(u_bc, boundary_dofs)
    
    # Solver metadata
    solver_info = {
        "mesh_resolution": resolution,
        "element_degree": degree,
        "ksp_type": "gmres",
        "pc_type": "hypre",
        "rationale": f"Adaptive mesh refinement with convergence check. Starting with N={resolution}, degree={degree}."
    }
    
    # Try iterative solver first
    petsc_options = {
        "ksp_type": "gmres",
        "pc_type": "hypre",
        "ksp_rtol": 1e-8,
        "ksp_atol": 1e-10,
        "ksp_max_it": 1000,
    }
    
    try:
        problem = LinearProblem(a, L, bcs=[bc], petsc_options=petsc_options, petsc_options_prefix='poisson_')
        u_h = problem.solve()
        solver_info["solver_used"] = "iterative_gmres_hypre"
        
    except Exception as e:
        if rank == 0:
            print(f"Iterative solver failed: {e}. Falling back to direct solver.")
        
        # Fallback to direct solver
        petsc_options = {
            "ksp_type": "preonly",
            "pc_type": "lu",
            "pc_factor_mat_solver_type": "mumps",
        }
        problem = LinearProblem(a, L, bcs=[bc], petsc_options=petsc_options, petsc_options_prefix='poisson_')
        u_h = problem.solve()
        solver_info["solver_used"] = "direct_lu_mumps"
        solver_info["fallback_reason"] = str(e)
    
    # Compute L2 norm of solution for convergence check
    norm_form = ufl.inner(u_h, u_h) * ufl.dx
    norm_value = fem.assemble_scalar(fem.form(norm_form))
    norm_value = comm.allreduce(norm_value, op=MPI.SUM)
    norm_value = np.sqrt(norm_value)
    
    # End timing
    wall_time_sec = time.time() - wall_time_start
    
    # Sample solution on 50x50 uniform grid
    # Create evaluation points
    n_samples = 50
    x_vals = np.linspace(0.0, 1.0, n_samples)
    y_vals = np.linspace(0.0, 1.0, n_samples)
    X, Y = np.meshgrid(x_vals, y_vals, indexing='ij')
    
    # Flatten for evaluation
    points = np.column_stack((X.flatten(), Y.flatten(), np.zeros(n_samples * n_samples)))
    
    # Evaluate solution at points
    bb_tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, points)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points)
    
    u_values = np.full(points.shape[0], np.nan, dtype=np.float64)
    
    # Find which points are inside the domain
    points_on_proc = []
    cells_on_proc = []
    local_points = []
    
    for i, point in enumerate(points):
        if len(colliding_cells.links(i)) > 0:
            points_on_proc.append(point)
            cells_on_proc.append(colliding_cells.links(i)[0])
            local_points.append(i)
    
    if len(points_on_proc) > 0:
        points_on_proc = np.array(points_on_proc)
        cells_on_proc = np.array(cells_on_proc)
        
        # Evaluate function at points
        u_eval = u_h.eval(points_on_proc, cells_on_proc)
        
        for idx, local_idx in enumerate(local_points):
            u_values[local_idx] = u_eval[idx]
    
    # Gather results across all MPI processes
    u_values_all = np.zeros_like(u_values)
    comm.Allreduce(u_values, u_values_all, op=MPI.SUM)
    
    # Reshape to 2D array
    u_grid = u_values_all.reshape((n_samples, n_samples))
    
    # Save solution to npz file (only rank 0)
    if rank == 0:
        np.savez(f"{outdir}/solution.npz", x=x_vals, y=y_vals, u=u_grid)
        
        # Save metadata
        metadata = {
            "wall_time_sec": wall_time_sec,
            "solver_info": solver_info,
            "l2_norm": float(norm_value),
            "resolution": resolution,
            "degree": degree,
            "n_samples": n_samples,
        }
        
        with open(f"{outdir}/meta.json", "w") as f:
            json.dump(metadata, f, indent=2)
    
    return {
        "wall_time_sec": wall_time_sec,
        "norm": norm_value,
        "resolution": resolution,
        "degree": degree,
    }


def adaptive_solve(degree, outdir, dt=None):
    """
    Adaptive mesh refinement loop for Poisson equation.
    
    Parameters:
    -----------
    degree : int
        Polynomial degree of finite element space
    outdir : str
        Output directory for solution files
    dt : float or None
        Not used for elliptic problem
    
    Returns:
    --------
    dict : Final solver metadata
    """
    comm = MPI.COMM_WORLD
    rank = comm.rank
    
    resolutions = [32, 64, 128]
    prev_norm = None
    final_result = None
    
    for i, N in enumerate(resolutions):
        if rank == 0:
            print(f"Solving with resolution N={N}, degree={degree}")
        
        result = solve_poisson(N, degree, outdir, dt)
        
        if prev_norm is not None:
            # Compute relative error in norm
            rel_error = abs(result["norm"] - prev_norm) / result["norm"]
            if rank == 0:
                print(f"  Relative error in norm: {rel_error:.6f}")
            
            if rel_error < 0.01:  # 1% convergence criterion
                if rank == 0:
                    print(f"  Converged! Using N={N}")
                final_result = result
                break
        
        prev_norm = result["norm"]
        final_result = result
    
    # If loop finished without convergence, use finest mesh result
    if rank == 0 and final_result is not None:
        print(f"Final solution: resolution={final_result['resolution']}, "
              f"degree={degree}, wall_time={final_result['wall_time_sec']:.2f}s")
    
    return final_result


def main():
    parser = argparse.ArgumentParser(description="Solve Poisson equation with adaptive mesh refinement")
    parser.add_argument("--resolution", type=int, default=64,
                       help="Grid resolution (e.g., 128)")
    parser.add_argument("--degree", type=int, default=2,
                       help="Polynomial degree of the element")
    parser.add_argument("--outdir", type=str, default=".",
                       help="Path to the output directory")
    parser.add_argument("--dt", type=float, default=None,
                       help="Time step size (only for time-dependent problems)")
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    import os
    if MPI.COMM_WORLD.rank == 0:
        os.makedirs(args.outdir, exist_ok=True)
    
    # Run adaptive solver
    adaptive_solve(args.degree, args.outdir, args.dt)


if __name__ == "__main__":
    main()
