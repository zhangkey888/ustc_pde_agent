import numpy as np
import json
from mpi4py import MPI
import ufl
from dolfinx import fem, mesh, default_scalar_type
import sys

# Test the manufactured solution implementation
comm = MPI.COMM_WORLD
domain = mesh.create_unit_square(comm, 4, 4, mesh.CellType.triangle)
V = fem.functionspace(domain, ("Lagrange", 1, (2,)))
x = ufl.SpatialCoordinate(domain)
pi = np.pi

# Manufactured solution from problem statement
u_exact_ufl = ufl.as_vector([pi * ufl.cos(pi * x[1]) * ufl.sin(pi * x[0]),
                             -pi * ufl.cos(pi * x[0]) * ufl.sin(pi * x[1])])

# Compute source term f = -∇·(∇u_exact) = -Δu_exact (component-wise)
# Let's compute symbolically using ufl
# Δu = div(grad(u))
# For vector field, grad gives a matrix, div of matrix gives vector
# Actually -∇·(∇u) = -div(grad(u))
f_computed = -ufl.div(ufl.grad(u_exact_ufl))

# Our derived expression
f_derived = ufl.as_vector([2 * pi**3 * ufl.cos(pi * x[1]) * ufl.sin(pi * x[0]),
                           -2 * pi**3 * ufl.cos(pi * x[0]) * ufl.sin(pi * x[1])])

print("Checking if derived f matches computed f...")
# Evaluate at a point
if comm.rank == 0:
    # Create functions to evaluate
    f_comp_func = fem.Function(V)
    f_der_func = fem.Function(V)
    
    def f_comp_eval(x):
        # We can't easily evaluate ufl expression directly, but we can check numerically
        # Use a simple point
        x0, y0 = 0.3, 0.4
        u1 = pi * np.cos(pi * y0) * np.sin(pi * x0)
        u2 = -pi * np.cos(pi * x0) * np.sin(pi * y0)
        # Compute Laplacian numerically using finite differences
        h = 1e-5
        # ∂²u1/∂x²
        u1_xx = (pi * np.cos(pi * y0) * np.sin(pi * (x0+h)) - 2*pi*np.cos(pi*y0)*np.sin(pi*x0) + pi*np.cos(pi*y0)*np.sin(pi*(x0-h))) / h**2
        # ∂²u1/∂y²
        u1_yy = (pi * np.cos(pi * (y0+h)) * np.sin(pi * x0) - 2*pi*np.cos(pi*y0)*np.sin(pi*x0) + pi*np.cos(pi*(y0-h))*np.sin(pi*x0)) / h**2
        Δu1 = u1_xx + u1_yy
        f1 = -Δu1
        
        # ∂²u2/∂x²
        u2_xx = (-pi * np.cos(pi * (x0+h)) * np.sin(pi * y0) + 2*pi*np.cos(pi*x0)*np.sin(pi*y0) - pi*np.cos(pi*(x0-h))*np.sin(pi*y0)) / h**2
        # ∂²u2/∂y²
        u2_yy = (-pi * np.cos(pi * x0) * np.sin(pi * (y0+h)) + 2*pi*np.cos(pi*x0)*np.sin(pi*y0) - pi*np.cos(pi*x0)*np.sin(pi*(y0-h))) / h**2
        Δu2 = u2_xx + u2_yy
        f2 = -Δu2
        
        return np.vstack((f1, f2))
    
    def f_der_eval(x):
        x0, y0 = 0.3, 0.4
        f1 = 2 * pi**3 * np.cos(pi * y0) * np.sin(pi * x0)
        f2 = -2 * pi**3 * np.cos(pi * x0) * np.sin(pi * y0)
        return np.vstack((f1, f2))
    
    f_comp_func.interpolate(f_comp_eval)
    f_der_func.interpolate(f_der_eval)
    
    # Evaluate at a point
    point = np.array([[0.3, 0.4, 0.0]], dtype=np.float64)
    from dolfinx import geometry
    bb_tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, point)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, point)
    cell = colliding_cells.links(0)[0]
    
    val_comp = f_comp_func.eval(point, [cell])[0]
    val_der = f_der_func.eval(point, [cell])[0]
    
    print(f"Computed f at (0.3,0.4): {val_comp}")
    print(f"Derived f at (0.3,0.4): {val_der}")
    print(f"Difference: {np.linalg.norm(val_comp - val_der)}")
    
    if np.linalg.norm(val_comp - val_der) < 1e-5:
        print("✓ Derived source term matches computed Laplacian")
    else:
        print("✗ Derived source term may be incorrect")
        sys.exit(1)

print("Verification complete.")
