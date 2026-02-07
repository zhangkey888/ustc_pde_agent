#!/usr/bin/env python3
"""
Poisson equation solver using FEniCSx (dolfinx).
Solves: -∇·(κ∇u) = f on unit square with Dirichlet BCs.
Manufactured solution: u_exact = sin(πx) sin(πy)
"""

import argparse
import json
import time
import numpy as np
from mpi4py import MPI
import ufl
from dolfinx import fem, mesh, default_scalar_type
from petsc4py import PETSc
from dolfinx.fem.petsc import LinearProblem


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Poisson equation solver")
    parser.add_argument("--resolution", type=int, default=128,
                        help="Grid resolution (number of cells per dimension)")
    parser.add_argument("--degree", type=int, default=1,
                        help="Polynomial degree of finite element")
    parser.add_argument("--outdir", type=str, default=".",
                        help="Output directory for solution files")
    parser.add_argument("--dt", type=float, default=0.01,
                        help="Time step size (not used for Poisson, kept for compatibility)")
    args = parser.parse_args()

    # Start timing
    start_time = time.time()

    # MPI communicator
    comm = MPI.COMM_WORLD

    # Create mesh
    n_cells = args.resolution
    domain = mesh.create_unit_square(comm, n_cells, n_cells, mesh.CellType.triangle)

    # Define function space
    V = fem.functionspace(domain, ("Lagrange", args.degree))

    # Define manufactured solution
    x = ufl.SpatialCoordinate(domain)
    u_exact_ufl = ufl.sin(np.pi * x[0]) * ufl.sin(np.pi * x[1])

    # Define source term f = -∇·(κ∇u_exact) with κ=1
    # Since -∇·(∇u_exact) = Δu_exact, and Δu_exact = -2π² sin(πx) sin(πy)
    f_expr = 2 * np.pi**2 * ufl.sin(np.pi * x[0]) * ufl.sin(np.pi * x[1])

    # Define boundary condition (Dirichlet on entire boundary)
    # Create function for exact solution
    u_exact_func = fem.Function(V)
    def u_exact_eval(x):
        return np.sin(np.pi * x[0]) * np.sin(np.pi * x[1])
    
    u_exact_func.interpolate(u_exact_eval)
    
    # Apply Dirichlet BC on all boundaries using geometrical approach
    def boundary(x):
        # Return True for points on boundary (x=0, x=1, y=0, or y=1)
        return np.logical_or.reduce((
            np.isclose(x[0], 0.0),
            np.isclose(x[0], 1.0),
            np.isclose(x[1], 0.0),
            np.isclose(x[1], 1.0)
        ))
    
    boundary_dofs = fem.locate_dofs_geometrical(V, boundary)
    bc = fem.dirichletbc(u_exact_func, boundary_dofs)

    # Define variational problem
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    κ = default_scalar_type(1.0)  # constant coefficient
    a = κ * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = f_expr * v * ufl.dx

    # Solve using LinearProblem (handles assembly and solving)
    problem = LinearProblem(a, L, bcs=[bc], petsc_options_prefix="poisson_",
                           petsc_options={
                               "ksp_type": "cg",
                               "pc_type": "hypre",
                               "ksp_rtol": 1e-10,
                               "ksp_atol": 1e-12,
                               "ksp_max_it": 1000
                           })
    
    uh = problem.solve()

    # Compute solver statistics
    end_time = time.time()
    wall_time = end_time - start_time

    # Sample solution on uniform 50x50 grid
    # Only rank 0 needs to do this for output
    if comm.rank == 0:
        n_samples = 50
        xs = np.linspace(0.0, 1.0, n_samples)
        ys = np.linspace(0.0, 1.0, n_samples)
        X, Y = np.meshgrid(xs, ys, indexing='ij')
        # Create 3D points with z=0 for 2D domain
        points = np.column_stack((X.flatten(), Y.flatten(), np.zeros(X.size)))
        points = points.astype(np.float64)  # Ensure correct dtype
        
        # Create a function that can evaluate at arbitrary points
        from dolfinx import geometry
        # Create bounding box tree
        bb_tree = geometry.bb_tree(domain, domain.topology.dim)
        
        u_values = np.zeros(points.shape[0], dtype=default_scalar_type)
        
        # Find cells containing each point
        cell_candidates = geometry.compute_collisions_points(bb_tree, points)
        colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points)
        
        for i in range(points.shape[0]):
            if len(colliding_cells.links(i)) > 0:
                cell = colliding_cells.links(i)[0]
                # Pass the full 3D point
                point_3d = points[i:i+1, :]
                u_values[i] = uh.eval(point_3d, [cell])[0]
            else:
                u_values[i] = np.nan
        
        u_grid = u_values.reshape((n_samples, n_samples))

        # Save solution to npz file
        import os
        os.makedirs(args.outdir, exist_ok=True)
        np.savez(f"{args.outdir}/solution.npz", x=xs, y=ys, u=u_grid)

        # Create metadata
        metadata = {
            "wall_time_sec": wall_time,
            "solver_info": {
                "mesh_resolution": args.resolution,
                "element_degree": args.degree,
                "ksp_type": "cg",
                "pc_type": "hypre",
                "rationale": f"Resolution {args.resolution} and degree {args.degree} chosen to balance accuracy and speed. CG with hypre AMG is efficient for Poisson problems."
            }
        }

        # Save metadata to JSON
        with open(f"{args.outdir}/meta.json", "w") as f:
            json.dump(metadata, f, indent=2)

        print(f"Solution saved to {args.outdir}/solution.npz")
        print(f"Metadata saved to {args.outdir}/meta.json")
        print(f"Wall time: {wall_time:.3f} seconds")


if __name__ == "__main__":
    main()
