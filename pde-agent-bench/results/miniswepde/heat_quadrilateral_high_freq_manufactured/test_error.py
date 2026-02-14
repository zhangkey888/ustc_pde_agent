import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem
import ufl
from dolfinx.fem import petsc
from petsc4py import PETSc

def compute_error(N, degree=1):
    """Compute L2 error for manufactured solution at t=0.1"""
    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_square(comm, N, N)
    V = fem.functionspace(domain, ("Lagrange", degree))
    
    # Exact solution at t=0.1
    def exact(x):
        return np.exp(-0.1) * np.sin(4*np.pi*x[0]) * np.sin(4*np.pi*x[1])
    
    # Create function with exact solution
    u_exact = fem.Function(V)
    u_exact.interpolate(exact)
    
    # Numerical solution (simulate with some error)
    # For testing, use exact solution perturbed
    u_numerical = fem.Function(V)
    u_numerical.interpolate(exact)
    # Add small perturbation to simulate numerical error
    rng = np.random.default_rng(42)
    perturbation = 0.001 * rng.random(u_numerical.x.array.shape)
    u_numerical.x.array[:] += perturbation
    
    # Compute L2 error
    error_form = fem.form(ufl.inner(u_numerical - u_exact, u_numerical - u_exact) * ufl.dx)
    error = np.sqrt(fem.assemble_scalar(error_form))
    
    # Also compute L2 norm of exact solution for relative error
    norm_form = fem.form(ufl.inner(u_exact, u_exact) * ufl.dx)
    norm = np.sqrt(fem.assemble_scalar(norm_form))
    
    return error, norm, error/norm

# Test different resolutions
for N in [32, 64, 128]:
    for degree in [1, 2]:
        error, norm, rel_error = compute_error(N, degree)
        print(f"N={N}, P{degree}: L2 error={error:.2e}, rel error={rel_error:.2e}")
