#!/usr/bin/env python3
"""
Poisson equation solver using FEniCSx (dolfinx).
Solves: -∇²u = f on unit square with homogeneous Dirichlet BCs.
Source term: f = sin(10πx) sin(8πy)
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
    parser.add_argument("--resolution", type=int, default=180,
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

    # Define source term f = sin(10πx) sin(8πy)
    x = ufl.SpatialCoordinate(domain)
    f_expr = ufl.sin(10 * np.pi * x[0]) * ufl.sin(8 * np.pi * x[1])

    # Define boundary condition (homogeneous Dirichlet on entire boundary)
    def boundary(x):
        # Return True for points on boundary (x=0, x=1, y=0, or y=1)
        return np.logical_or.reduce((
            np.isclose(x[0], 0.0),
            np.isclose(x[0], 1.0),
            np.isclose(x[1], 0.0),
            np.isclose(x[1], 1.0)
        ))
    
    boundary_dofs = fem.locate_dofs_geometrical(V, boundary)
    
    # Create zero function for BC
    zero_func = fem.Function(V)
    bc = fem.dirichletbc(zero_func, boundary_dofs)

    # Define variational problem
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    a = ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = f_expr * v * ufl.dx

    # Solve using LinearProblem
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
                u_values[i] = 0.0
        
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
                "rationale": f"Resolution {args.resolution} and degree {args.degree} chosen based on oracle config. CG with hypre AMG is efficient for Poisson problems. High frequency source (10π, 8π) requires moderate resolution to resolve oscillations."
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
