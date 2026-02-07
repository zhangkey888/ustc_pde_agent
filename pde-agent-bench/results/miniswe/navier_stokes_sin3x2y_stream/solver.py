#!/usr/bin/env python3
"""
Solver for vector Poisson equation with manufactured solution.
"""
import argparse
import json
import time
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
import ufl
from dolfinx import mesh, fem, default_real_type
from dolfinx.fem import (functionspace, Function, dirichletbc,
                         locate_dofs_geometrical, form, Constant, Expression)
from dolfinx.fem.petsc import LinearProblem
import dolfinx.geometry as geometry

def main():
    parser = argparse.ArgumentParser(description="Solve vector Poisson equation.")
    parser.add_argument("--resolution", type=int, default=128,
                        help="Grid resolution (number of cells per dimension).")
    parser.add_argument("--degree", type=int, default=2,
                        help="Polynomial degree of finite element.")
    parser.add_argument("--outdir", type=str, default=".",
                        help="Output directory for solution files.")
    parser.add_argument("--dt", type=float, default=0.01,
                        help="Time step size (not used for Poisson, but kept for compatibility).")
    args = parser.parse_args()

    # Start timer
    start_time = time.perf_counter()

    # MPI communicator
    comm = MPI.COMM_WORLD

    # Create mesh
    nx = ny = args.resolution
    domain = mesh.create_unit_square(comm, nx, ny, dtype=default_real_type)

    # Define function space: vector-valued Lagrange
    V = functionspace(domain, ("Lagrange", args.degree, (2,)))

    # Manufactured solution
    x = ufl.SpatialCoordinate(domain)
    u_exact = ufl.as_vector([
        2 * np.pi * ufl.cos(2 * np.pi * x[1]) * ufl.sin(3 * np.pi * x[0]),
        -3 * np.pi * ufl.cos(3 * np.pi * x[0]) * ufl.sin(2 * np.pi * x[1])
    ])

    # Define κ (identity matrix for simplicity)
    kappa = Constant(domain, 1.0)  # scalar diffusivity

    # Define variational problem
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    a = ufl.inner(kappa * ufl.grad(u), ufl.grad(v)) * ufl.dx
    # Source term f = -∇·(κ ∇u_exact)
    f = -ufl.div(kappa * ufl.grad(u_exact))
    L = ufl.inner(f, v) * ufl.dx

    # Boundary condition: u = u_exact on entire boundary
    def boundary(x):
        return np.full(x.shape[1], True, dtype=bool)

    boundary_dofs = locate_dofs_geometrical(V, boundary)
    u_bc = Function(V)
    u_bc.interpolate(Expression(u_exact, V.element.interpolation_points))
    bc = dirichletbc(u_bc, boundary_dofs)

    # Solve linear problem
    petsc_options = {
        "ksp_type": "gmres",
        "pc_type": "hypre",
        "ksp_rtol": 1e-10,
        "ksp_atol": 1e-12,
        "ksp_max_it": 1000,
    }
    problem = LinearProblem(a, L, bcs=[bc], petsc_options_prefix="poisson_", petsc_options=petsc_options)
    uh = problem.solve()

    # Compute error (optional, for verification)
    error = fem.assemble_scalar(form(ufl.inner(uh - u_exact, uh - u_exact) * ufl.dx))
    error = np.sqrt(comm.allreduce(error, op=MPI.SUM))

    # End timer
    end_time = time.perf_counter()
    wall_time = end_time - start_time

    # Sample solution on a uniform 50x50 grid
    n_sample = 50
    xs = np.linspace(0.0, 1.0, n_sample)
    ys = np.linspace(0.0, 1.0, n_sample)
    xv, yv = np.meshgrid(xs, ys, indexing='ij')
    points = np.vstack([xv.flatten(), yv.flatten()]).T
    # Add zero z-coordinate for 3D points (required by dolfinx)
    points_3d = np.hstack([points, np.zeros((points.shape[0], 1))]).astype(default_real_type)

    # Find cells containing each point
    bb_tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, points_3d)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points_3d)

    # For each point, get first colliding cell
    cells = []
    points_on_proc = []
    for i in range(len(points_3d)):
        if len(colliding_cells.links(i)) > 0:
            cells.append(colliding_cells.links(i)[0])
            points_on_proc.append(points[i])  # store 2D points for eval
    cells = np.array(cells, dtype=np.int32)
    points_on_proc = np.array(points_on_proc, dtype=default_real_type)

    # Evaluate function at points
    if len(points_on_proc) > 0:
        # Evaluate function at points (need 3D coordinates)
        points_on_proc_3d = np.hstack([points_on_proc, np.zeros((points_on_proc.shape[0], 1))]).astype(default_real_type)
        u_values = uh.eval(points_on_proc_3d, cells)
    else:
        u_values = np.zeros((0, 2), dtype=default_real_type)

    # Gather results to rank 0
    u_all = comm.gather(u_values, root=0)
    points_all = comm.gather(points_on_proc, root=0)

    if comm.rank == 0:
        u_all = np.vstack(u_all)
        points_all = np.vstack(points_all)
        # Reorder according to original grid ordering
        # Map points back to indices
        idx = np.lexsort((points_all[:, 1], points_all[:, 0]))
        points_all = points_all[idx]
        u_all = u_all[idx]

        # Reshape to 2D arrays
        x_arr = points_all[:, 0].reshape(n_sample, n_sample)
        y_arr = points_all[:, 1].reshape(n_sample, n_sample)
        # u is vector, save as 3D array (n_sample, n_sample, 2)
        u_combined = u_all.reshape(n_sample, n_sample, 2)
        # Save to solution.npz
        np.savez(f"{args.outdir}/solution.npz", x=x_arr, y=y_arr, u=u_combined)

        # Save metadata
        meta = {
            "wall_time_sec": wall_time,
            "solver_info": {
                "mesh_resolution": args.resolution,
                "element_degree": args.degree,
                "ksp_type": "gmres",
                "pc_type": "hypre",
                "rationale": "Moderate resolution and quadratic elements balance accuracy and speed. GMRES with hypre preconditioner is efficient for Poisson.",
                "error": float(error),
            }
        }
        with open(f"{args.outdir}/meta.json", "w") as f:
            json.dump(meta, f, indent=2)

        print(f"Solved in {wall_time:.3f} seconds, error L2 = {error:.3e}")
    else:
        # Other ranks do nothing
        pass

if __name__ == "__main__":
    main()
