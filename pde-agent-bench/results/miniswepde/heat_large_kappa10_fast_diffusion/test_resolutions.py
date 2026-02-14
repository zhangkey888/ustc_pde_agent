import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem
import ufl
from dolfinx.fem import petsc
import time

comm = MPI.COMM_WORLD

# Parameters
kappa = 10.0
t_end = 0.05
dt = 0.005
n_steps = int(t_end / dt)
pi = np.pi

resolutions = [32, 64, 128]
errors = []

for N in resolutions:
    domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", 1))
    x = ufl.SpatialCoordinate(domain)
    
    # Boundary facets
    tdim = domain.topology.dim
    fdim = tdim - 1
    def boundary_marker(x):
        return np.logical_or.reduce([
            np.isclose(x[0], 0.0),
            np.isclose(x[0], 1.0),
            np.isclose(x[1], 0.0),
            np.isclose(x[1], 1.0)
        ])
    boundary_facets = mesh.locate_entities_boundary(domain, fdim, boundary_marker)
    dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    
    # Time-stepping
    u_n = fem.Function(V)
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    
    # Initial condition
    u_exact_0 = ufl.sin(pi*x[0]) * ufl.sin(pi*x[1])
    u_n.interpolate(fem.Expression(u_exact_0, V.element.interpolation_points))
    
    t = 0.0
    for step in range(n_steps):
        t += dt
        
        # BC at current time
        u_exact_bc = ufl.exp(-t) * ufl.sin(pi*x[0]) * ufl.sin(pi*x[1])
        u_bc = fem.Function(V)
        u_bc.interpolate(fem.Expression(u_exact_bc, V.element.interpolation_points))
        bc = fem.dirichletbc(u_bc, dofs)
        
        # Source term at current time
        f_expr = ufl.exp(-t) * ufl.sin(pi*x[0]) * ufl.sin(pi*x[1]) * (-1 + 2*kappa*pi**2)
        f_func = fem.Function(V)
        f_func.interpolate(fem.Expression(f_expr, V.element.interpolation_points))
        
        # Variational forms
        a = ufl.inner(u, v) * ufl.dx + dt * kappa * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
        L = ufl.inner(u_n, v) * ufl.dx + dt * ufl.inner(f_func, v) * ufl.dx
        
        # Solve
        problem = petsc.LinearProblem(a, L, bcs=[bc], 
                                      petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
                                      petsc_options_prefix="transient_")
        u_sol = problem.solve()
        
        # Update for next step
        u_n.x.array[:] = u_sol.x.array
    
    # Compute error at final time
    u_exact_final = fem.Function(V)
    u_exact_final_expr = ufl.exp(-t_end) * ufl.sin(pi*x[0]) * ufl.sin(pi*x[1])
    u_exact_final.interpolate(fem.Expression(u_exact_final_expr, V.element.interpolation_points))
    
    error = fem.Function(V)
    error.x.array[:] = u_sol.x.array - u_exact_final.x.array
    error_form = fem.form(ufl.inner(error, error) * ufl.dx)
    error_norm = np.sqrt(fem.assemble_scalar(error_form))
    errors.append(error_norm)
    print(f"N={N}: L2 error = {error_norm:.2e}")

print("\nRequired error: ≤4.64e-04")
print(f"With N=128, error is {errors[-1]:.2e}, {'PASS' if errors[-1] <= 4.64e-04 else 'FAIL'}")
