#!/usr/bin/env python3
"""
Poisson equation solver for boundary layer problem.
Manufactured solution: u = exp(6*x)*sin(pi*y)
Domain: unit square [0,1]x[0,1]
"""

import argparse
import json
import time
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
import ufl
from dolfinx import mesh, fem, io, default_scalar_type
from dolfinx.fem.petsc import LinearProblem


def main():
    parser = argparse.ArgumentParser(description="Solve Poisson equation with manufactured solution")
    parser.add_argument("--resolution", type=int, default=100,
                        help="Grid resolution (number of cells per dimension)")
    parser.add_argument("--degree", type=int, default=2,
                        help="Polynomial degree of finite element")
    parser.add_argument("--outdir", type=str, default=".",
                        help="Output directory for solution files")
    parser.add_argument("--dt", type=float, default=0.01,
                        help="Time step size (not used for Poisson, kept for compatibility)")
    args = parser.parse_args()

    # Start timer
    start_time = time.time()

    # Create mesh
    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_square(comm, args.resolution, args.resolution, 
                                     cell_type=mesh.CellType.quadrilateral)
    
    # Define function space
    V = fem.functionspace(domain, ("Lagrange", args.degree))
    
    # Define trial and test functions
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    
    # Define manufactured solution expression
    x = ufl.SpatialCoordinate(domain)
    u_exact_expr = ufl.exp(6 * x[0]) * ufl.sin(np.pi * x[1])
    
    # Define kappa coefficient
    kappa = default_scalar_type(1.0)
    
    # Define source term f = -∇·(κ ∇u_exact) analytically
    # u_exact = exp(6*x)*sin(pi*y), κ=1
    # ∇u = (6*exp(6*x)*sin(pi*y), pi*exp(6*x)*cos(pi*y))
    # ∇·(κ∇u) = 36*exp(6*x)*sin(pi*y) - pi^2*exp(6*x)*sin(pi*y)
    # f = -∇·(κ∇u) = -(36 - pi^2)*exp(6*x)*sin(pi*y)
    f = -(36 - np.pi**2) * ufl.exp(6 * x[0]) * ufl.sin(np.pi * x[1])
    
    # Define variational form
    a = kappa * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = f * v * ufl.dx
    
    # Define boundary condition: u = u_exact on all boundaries
    # First, create a function for the boundary values
    u_bc = fem.Function(V)
    
    # Get dof coordinates
    dof_coords = V.tabulate_dof_coordinates()
    
    # Evaluate exact solution at dof coordinates
    # Note: dof_coords has shape (n_dofs, 3) but domain is 2D, third coord is 0
    x_vals = dof_coords[:, 0]
    y_vals = dof_coords[:, 1]
    u_exact_vals = np.exp(6 * x_vals) * np.sin(np.pi * y_vals)
    
    # Set the function values
    u_bc.x.array[:] = u_exact_vals
    
    # Apply boundary condition to all boundary facets
    facets = mesh.locate_entities_boundary(domain, domain.topology.dim-1, lambda x: np.full(x.shape[1], True))
    dofs = fem.locate_dofs_topological(V, domain.topology.dim-1, facets)
    bc = fem.dirichletbc(u_bc, dofs)
    
    # Solve linear system
    # Use CG with hypre AMG preconditioner for symmetric positive definite problem
    petsc_options = {
        "ksp_type": "cg",
        "pc_type": "hypre",
        "pc_hypre_type": "boomeramg",
        "ksp_rtol": 1e-10,
        "ksp_atol": 1e-12,
        "ksp_max_it": 1000,
    }
    
    problem = LinearProblem(a, L, bcs=[bc], petsc_options_prefix="poisson_",
                           petsc_options=petsc_options)
    uh = problem.solve()
    
    # Compute error for diagnostics (not required but useful)
    error_L2 = fem.assemble_scalar(fem.form((uh - u_exact_expr)**2 * ufl.dx))
    error_L2 = np.sqrt(comm.allreduce(error_L2, op=MPI.SUM))
    
    # End timer
    end_time = time.time()
    wall_time = end_time - start_time
    
    # Sample solution on 50x50 uniform grid
    n_samples = 50
    xs = np.linspace(0, 1, n_samples)
    ys = np.linspace(0, 1, n_samples)
    X, Y = np.meshgrid(xs, ys, indexing='ij')
    
    # Evaluate solution at grid points
    points = np.column_stack((X.ravel(), Y.ravel(), np.zeros(X.size))).astype(default_scalar_type)
    u_values = np.zeros(points.shape[0], dtype=default_scalar_type)
    
    # Create bounding box tree for point location
    from dolfinx import geometry
    bb_tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, points)
    cell_indices = geometry.compute_colliding_cells(domain, cell_candidates, points)
    
    # Evaluate solution at points that are in this process
    points_on_proc = []
    cells_on_proc = []
    for i in range(points.shape[0]):
        if len(cell_indices.links(i)) > 0:
            points_on_proc.append(points[i])
            cells_on_proc.append(cell_indices.links(i)[0])
    
    if len(points_on_proc) > 0:
        points_on_proc = np.array(points_on_proc, dtype=default_scalar_type)
        u_values_local = uh.eval(points_on_proc, cells_on_proc)
        # u_values_local has shape (n_points, value_size) where value_size=1 for scalar
        u_values_local = u_values_local[:, 0]
        
        # Map back to global array
        idx = 0
        for i in range(points.shape[0]):
            if len(cell_indices.links(i)) > 0:
                u_values[i] = u_values_local[idx]
                idx += 1
    
    U = u_values.reshape(n_samples, n_samples)
    
    # Save solution to npz file
    import os
    os.makedirs(args.outdir, exist_ok=True)
    np.savez(os.path.join(args.outdir, "solution.npz"), 
             x=xs, y=ys, u=U)
    
    # Save metadata
    metadata = {
        "wall_time_sec": wall_time,
        "solver_info": {
            "mesh_resolution": args.resolution,
            "element_degree": args.degree,
            "ksp_type": "cg",
            "pc_type": "hypre",
            "pc_hypre_type": "boomeramg",
            "error_L2": float(error_L2),
            "rationale": f"CG with hypre AMG for symmetric positive definite Poisson. Resolution {args.resolution} with degree {args.degree} provides good accuracy for smooth solution."
        }
    }
    
    with open(os.path.join(args.outdir, "meta.json"), "w") as f:
        json.dump(metadata, f, indent=2)
    
    if comm.rank == 0:
        print(f"Solution saved to {args.outdir}/solution.npz")
        print(f"Metadata saved to {args.outdir}/meta.json")
        print(f"Wall time: {wall_time:.3f} seconds")
        print(f"L2 error: {error_L2:.2e}")


if __name__ == "__main__":
    main()
