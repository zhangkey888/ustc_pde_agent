import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, io, geometry
import ufl
from petsc4py import PETSc
import time

# Test case specification
case_spec = {
    'pde': {
        'type': 'poisson',
        'coefficients': {
            'kappa': {'type': 'expr', 'expr': '1 + 0.3*sin(8*pi*x)*sin(8*pi*y)'}
        }
    },
    'domain': {'type': 'square', 'bounds': [[0,1], [0,1]]}
}

# Import the solver function (will be defined in solver.py)
from solver import solve

# Run the solver
start = time.time()
result = solve(case_spec)
end = time.time()

print(f"Time taken: {end - start:.3f}s")
print("\nSolver info:")
for key, value in result['solver_info'].items():
    print(f"  {key}: {value}")

print(f"\nu shape: {result['u'].shape}")
print(f"u min/max: {result['u'].min():.6f}, {result['u'].max():.6f}")

# Check if solution looks reasonable (should be sin(2πx)sin(2πy))
x = np.linspace(0, 1, 50)
y = np.linspace(0, 1, 50)
X, Y = np.meshgrid(x, y, indexing='ij')
u_exact = np.sin(2*np.pi*X) * np.sin(2*np.pi*Y)
error = np.abs(result['u'] - u_exact).max()
print(f"Max error: {error:.6e}")
print(f"Accuracy requirement (1.28e-03): {'PASS' if error <= 1.28e-03 else 'FAIL'}")
print(f"Time requirement (2.569s): {'PASS' if end - start <= 2.569 else 'FAIL'}")

# Check required fields
required_fields = ['mesh_resolution', 'element_degree', 'ksp_type', 'pc_type', 'rtol', 'iterations']
missing = [field for field in required_fields if field not in result['solver_info']]
if missing:
    print(f"\nWARNING: Missing required fields: {missing}")
else:
    print("\nAll required fields present in solver_info")
