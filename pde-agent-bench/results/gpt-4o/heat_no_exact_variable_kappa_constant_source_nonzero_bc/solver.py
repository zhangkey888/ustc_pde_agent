#!/usr/bin/env python3
"""
Solver for transient heat equation with variable kappa coefficient.
Uses dolfinx (FEniCSx) v0.8.0+ syntax.
"""

import argparse
import json
import time
import numpy as np
from pathlib import Path
from mpi4py import MPI
from petsc4py import PETSc
import dolfinx
import ufl
from dolfinx import fem, mesh, io, default_scalar_type
from dolfinx.fem.petsc import assemble_matrix, assemble_vector, apply_lifting, set_bc
from dolfinx.fem import functionspace, Function, dirichletbc, locate_dofs_geometrical, Constant, form
from dolfinx.mesh import create_unit_square


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Solve transient heat equation")
    parser.add_argument("--resolution", type=int, default=128,
                        help="Grid resolution (number of cells per dimension)")
    parser.add_argument("--degree", type=int, default=1,
                        help="Polynomial degree of finite element")
    parser.add_argument("--outdir", type=str, default="output",
                        help="Output directory path")
    parser.add_argument("--dt", type=float, default=0.02,
                        help="Time step size")
    return parser.parse_args()


def exact_solution(x, t):
    """Manufactured exact solution for testing."""
    x_coord = x[0]
    y_coord = x[1]
    return np.exp(-2 * np.pi**2 * t) * np.sin(np.pi * x_coord) * np.sin(np.pi * y_coord)


def kappa_func(x):
    """Variable conductivity coefficient: κ = 1 + 0.5*sin(2πx)*sin(2πy)."""
    x_coord = x[0]
    y_coord = x[1]
    return 1.0 + 0.5 * np.sin(2 * np.pi * x_coord) * np.sin(2 * np.pi * y_coord)


def source_term(x, t):
    """Source term f derived from manufactured solution and variable kappa."""
    x_coord = x[0]
    y_coord = x[1]
    
    # u and its derivatives
    u = np.exp(-2 * np.pi**2 * t) * np.sin(np.pi * x_coord) * np.sin(np.pi * y_coord)
    u_t = -2 * np.pi**2 * u  # ∂u/∂t
    
    u_x = np.pi * np.exp(-2 * np.pi**2 * t) * np.cos(np.pi * x_coord) * np.sin(np.pi * y_coord)
    u_y = np.pi * np.exp(-2 * np.pi**2 * t) * np.sin(np.pi * x_coord) * np.cos(np.pi * y_coord)
    
    # kappa and its derivatives
    kappa = 1.0 + 0.5 * np.sin(2 * np.pi * x_coord) * np.sin(2 * np.pi * y_coord)
    kappa_x = np.pi * np.cos(2 * np.pi * x_coord) * np.sin(2 * np.pi * y_coord)
    kappa_y = np.pi * np.sin(2 * np.pi * x_coord) * np.cos(2 * np.pi * y_coord)
    
    # ∇·(κ ∇u) = κ∇²u + ∇κ·∇u
    laplace_u = -2 * np.pi**2 * u
    grad_kappa_dot_grad_u = kappa_x * u_x + kappa_y * u_y
    
    div_kappa_grad_u = kappa * laplace_u + grad_kappa_dot_grad_u
    
    # f = ∂u/∂t - ∇·(κ ∇u)
    f = u_t - div_kappa_grad_u
    
    return f


def sample_solution_on_grid(u_func, n_samples=50):
    """Sample FEM solution on a uniform n_samples x n_samples grid."""
    # Get mesh and solution values at vertices
    msh = u_func.function_space.mesh
    V = u_func.function_space
    
    # Get coordinates of all vertices
    coords = msh.geometry.x[:, :2]  # (n_vertices, 2)
    
    # Get solution values at vertices
    u_vals = u_func.x.array
    
    # Create regular grid
    x_vals = np.linspace(0.0, 1.0, n_samples)
    y_vals = np.linspace(0.0, 1.0, n_samples)
    X, Y = np.meshgrid(x_vals, y_vals, indexing='ij')
    
    # Simple nearest neighbor interpolation
    u_grid = np.zeros((n_samples, n_samples))
    
    for i in range(n_samples):
        for j in range(n_samples):
            x = X[i, j]
            y = Y[i, j]
            # Find closest vertex
            distances = np.sqrt((coords[:, 0] - x)**2 + (coords[:, 1] - y)**2)
            closest_idx = np.argmin(distances)
            u_grid[i, j] = u_vals[closest_idx]
    
    return x_vals, y_vals, u_grid


def main():
    """Main solver function."""
    args = parse_args()
    
    # Start timing
    start_time = time.time()
    
    # Create output directory
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    
    # MPI communicator
    comm = MPI.COMM_WORLD
    
    # Create mesh
    mesh_res = args.resolution
    msh = create_unit_square(comm, mesh_res, mesh_res)
    
    # Define function space
    degree = args.degree
    V = functionspace(msh, ("Lagrange", degree))
    
    # Define trial and test functions
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    
    # Define functions for solution
    u_n = Function(V)  # Solution at previous time step
    u_sol = Function(V)  # Current solution
    
    # Time parameters
    dt_val = args.dt
    T = 0.1  # Final time
    num_steps = int(T / dt_val)
    # Adjust last time step if needed
    if num_steps * dt_val < T - 1e-12:
        num_steps += 1
    
    dt = Constant(msh, default_scalar_type(dt_val))
    
    # Initial condition
    def initial_condition(x):
        return exact_solution(x, 0.0)
    
    u_n.interpolate(initial_condition)
    
    # Define kappa as a Function for proper integration
    kappa = Function(V)
    kappa.interpolate(kappa_func)
    
    # Define source term function
    f = Function(V)
    
    # Define boundary condition
    def boundary(x):
        return np.logical_or(
            np.logical_or(np.isclose(x[0], 0.0), np.isclose(x[0], 1.0)),
            np.logical_or(np.isclose(x[1], 0.0), np.isclose(x[1], 1.0))
        )
    
    # Create boundary condition
    boundary_dofs = locate_dofs_geometrical(V, boundary)
    
    # Variational form for backward Euler
    # (u - u_n)/dt * v * dx + kappa * dot(grad(u), grad(v)) * dx = f * v * dx
    a = (u / dt_val) * v * ufl.dx + kappa * ufl.dot(ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = (u_n / dt_val) * v * ufl.dx + f * v * ufl.dx
    
    # Create forms
    a_form = form(a)
    L_form = form(L)
    
    # Assemble matrix (constant in time)
    A = assemble_matrix(a_form)
    A.assemble()
    
    # Create vector for RHS
    b = Function(V)
    
    # Create linear solver
    solver = PETSc.KSP().create(comm)
    solver.setOperators(A)
    solver.setType("cg")
    solver.getPC().setType("hypre")
    solver.setTolerances(rtol=1e-10, atol=1e-10)
    
    # Time stepping
    t = 0.0
    for step in range(num_steps):
        # Adjust last time step to reach exactly T
        current_dt = dt_val
        if step == num_steps - 1 and t + dt_val > T:
            current_dt = T - t
        
        t += current_dt
        
        # Update source term for current time
        def current_source(x):
            return source_term(x, t)
        f.interpolate(current_source)
        
        # Update boundary condition
        def bc_value(x):
            return exact_solution(x, t)
        u_bc = Function(V)
        u_bc.interpolate(bc_value)
        bc = dirichletbc(u_bc, boundary_dofs)
        
        # Assemble right-hand side
        with b.x.petsc_vec.localForm() as b_local:
            b_local.set(0.0)
        assemble_vector(b.x.petsc_vec, L_form)
        
        # Apply boundary conditions
        apply_lifting(b.x.petsc_vec, [a_form], bcs=[[bc]])
        b.x.petsc_vec.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        set_bc(b.x.petsc_vec, [bc])
        
        # Solve linear system
        solver.solve(b.x.petsc_vec, u_sol.x.petsc_vec)
        
        # Update previous solution
        u_n.x.array[:] = u_sol.x.array[:]
    
    # End timing
    end_time = time.time()
    wall_time = end_time - start_time
    
    # Sample solution on 50x50 grid
    x_vals, y_vals, u_grid = sample_solution_on_grid(u_sol, n_samples=50)
    
    # Save solution to npz file
    np.savez(outdir / "solution.npz", 
             x=x_vals, 
             y=y_vals, 
             u=u_grid)
    
    # Create metadata
    metadata = {
        "wall_time_sec": wall_time,
        "solver_info": {
            "mesh_resolution": mesh_res,
            "element_degree": degree,
            "ksp_type": "cg",
            "pc_type": "hypre",
            "rationale": f"Resolution {mesh_res} and degree {degree} provide good balance between accuracy and speed for target error 0.01. CG with hypre preconditioner is efficient for symmetric positive definite heat equation.",
            "pde_specific": {
                "time_stepping": {
                    "dt": dt_val,
                    "n_steps": num_steps,
                    "scheme": "backward_euler"
                }
            }
        }
    }
    
    # Save metadata to JSON
    with open(outdir / "meta.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    if comm.rank == 0:
        print(f"Solver completed in {wall_time:.2f} seconds")
        print(f"Results saved to {outdir}/")
        print(f"Mesh resolution: {mesh_res}, Element degree: {degree}")
        print(f"Time step: {dt_val}, Number of steps: {num_steps}")


if __name__ == "__main__":
    main()
