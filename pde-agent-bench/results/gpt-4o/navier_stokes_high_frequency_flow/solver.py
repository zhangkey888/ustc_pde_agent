#!/usr/bin/env python3
"""
Solver for vector Poisson equation with manufactured solution.
"""
import argparse
import json
import time
import os
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
import ufl
from dolfinx import fem, mesh, io
from dolfinx.fem import Function, functionspace, dirichletbc, locate_dofs_topological, Expression, form
from dolfinx.fem.petsc import assemble_matrix, assemble_vector, create_matrix, create_vector
from dolfinx.mesh import create_unit_square, exterior_facet_indices

def main():
    parser = argparse.ArgumentParser(description="Solve vector Poisson equation.")
    parser.add_argument("--resolution", type=int, default=128,
                        help="Grid resolution (number of cells per side)")
    parser.add_argument("--degree", type=int, default=2,
                        help="Polynomial degree of finite element")
    parser.add_argument("--outdir", type=str, default=".",
                        help="Output directory")
    parser.add_argument("--dt", type=float, default=0.01,
                        help="Time step size (not used for Poisson, but kept for compatibility)")
    args = parser.parse_args()

    comm = MPI.COMM_WORLD
    rank = comm.rank

    # Start timing
    wall_time_start = time.perf_counter()

    # Create mesh
    nx = args.resolution
    ny = args.resolution
    domain = create_unit_square(comm, nx, ny, mesh.CellType.triangle)

    # Create connectivity for facets
    tdim = domain.topology.dim
    domain.topology.create_connectivity(tdim-1, tdim)

    # Define function space: vector-valued Lagrange
    V = functionspace(domain, ("Lagrange", args.degree, (2,)))

    # Define manufactured solution
    x = ufl.SpatialCoordinate(domain)
    u_exact = ufl.as_vector([
        2 * np.pi * ufl.cos(2 * np.pi * x[1]) * ufl.sin(2 * np.pi * x[0]),
        -2 * np.pi * ufl.cos(2 * np.pi * x[0]) * ufl.sin(2 * np.pi * x[1])
    ])

    # Define source term f = -∇² u_exact (since κ=1)
    # For vector Poisson: -∇·(∇ u) = f component-wise
    f = -ufl.div(ufl.grad(u_exact))

    # Define trial and test functions
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    # Bilinear form a(u,v) = ∫ ∇u : ∇v dx
    a = ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
    a_form = form(a)
    # Linear form L(v) = ∫ f·v dx
    L = ufl.inner(f, v) * ufl.dx
    L_form = form(L)

    # Apply Dirichlet BC: u = u_exact on entire boundary
    boundary_facets = exterior_facet_indices(domain.topology)
    boundary_dofs = locate_dofs_topological(V, tdim-1, boundary_facets)
    u_bc = Function(V)
    # Interpolate exact expression
    u_expr = Expression(u_exact, V.element.interpolation_points)
    u_bc.interpolate(u_expr)
    bc = dirichletbc(u_bc, boundary_dofs)

    # Assemble system with boundary conditions
    A = assemble_matrix(a_form, bcs=[bc])
    A.assemble()
    b = assemble_vector(L_form)
    fem.apply_lifting(b, [a_form], [[bc]])
    b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    fem.set_bc(b, [bc])

    # Solve linear system
    ksp = PETSc.KSP().create(comm)
    ksp.setOperators(A)
    ksp.setType("cg")
    ksp.getPC().setType("hypre")
    ksp.setTolerances(rtol=1e-10, atol=1e-10, max_it=1000)
    ksp.setFromOptions()

    u_h = Function(V)
    # Get PETSc vector from function
    u_h_petsc = u_h.x.petsc_vec
    ksp.solve(b, u_h_petsc)
    u_h.x.scatter_forward()

    # Stop timing
    wall_time_sec = time.perf_counter() - wall_time_start

    # Sample solution on a uniform 50x50 grid
    # We'll evaluate at mesh vertices and then interpolate to grid via nearest neighbor.
    # Interpolate u_h to linear Lagrange space to get vertex values.
    V_linear = functionspace(domain, ("Lagrange", 1, (2,)))
    u_linear = Function(V_linear)
    u_linear.interpolate(u_h)

    # Get vertex coordinates and values
    vertices = domain.geometry.x[:, :2]
    values = u_linear.x.array.reshape(-1, 2)

    # Create uniform grid
    n_sample = 50
    x_grid = np.linspace(0.0, 1.0, n_sample)
    y_grid = np.linspace(0.0, 1.0, n_sample)
    X, Y = np.meshgrid(x_grid, y_grid, indexing='ij')
    grid_points = np.stack([X.flatten(), Y.flatten()], axis=1)

    # Nearest neighbor interpolation
    u_grid = np.zeros((grid_points.shape[0], 2))
    for i, pt in enumerate(grid_points):
        dists = np.sum((vertices - pt)**2, axis=1)
        idx = np.argmin(dists)
        u_grid[i] = values[idx]
    u_grid = u_grid.reshape(n_sample, n_sample, 2)

    # Save to solution.npz (only rank 0)
    if rank == 0:
        os.makedirs(args.outdir, exist_ok=True)
        np.savez(f"{args.outdir}/solution.npz", x=x_grid, y=y_grid, u=u_grid)

        # Save metadata
        metadata = {
            "wall_time_sec": wall_time_sec,
            "solver_info": {
                "mesh_resolution": args.resolution,
                "element_degree": args.degree,
                "ksp_type": "cg",
                "pc_type": "hypre",
                "rationale": "CG with hypre AMG is efficient for Poisson. Resolution 128 and degree 2 balance accuracy and speed."
            }
        }
        with open(f"{args.outdir}/meta.json", "w") as f:
            json.dump(metadata, f, indent=2)

if __name__ == "__main__":
    main()
