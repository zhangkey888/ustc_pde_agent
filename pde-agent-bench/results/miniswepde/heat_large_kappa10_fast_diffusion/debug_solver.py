import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem
import ufl
from dolfinx.fem import petsc

comm = MPI.COMM_WORLD
domain = mesh.create_unit_square(comm, 32, 32, cell_type=mesh.CellType.triangle)
V = fem.functionspace(domain, ("Lagrange", 1))
x = ufl.SpatialCoordinate(domain)
pi = np.pi

kappa = 10.0
t_end = 0.05
dt = 0.005
n_steps = int(t_end / dt)

# Boundary
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

# Initial condition
u_n = fem.Function(V)
u_exact_0 = ufl.sin(pi*x[0]) * ufl.sin(pi*x[1])
u_n.interpolate(fem.Expression(u_exact_0, V.element.interpolation_points))

t = 0.0
for step in range(n_steps):
    t_prev = t
    t += dt
    
    # BC at new time
    u_exact_bc = ufl.exp(-t) * ufl.sin(pi*x[0]) * ufl.sin(pi*x[1])
    u_bc = fem.Function(V)
    u_bc.interpolate(fem.Expression(u_exact_bc, V.element.interpolation_points))
    bc = fem.dirichletbc(u_bc, dofs)
    
    # Source term at new time
    f_expr = ufl.exp(-t) * ufl.sin(pi*x[0]) * ufl.sin(pi*x[1]) * (-1 + 2*kappa*pi**2)
    f_func = fem.Function(V)
    f_func.interpolate(fem.Expression(f_expr, V.element.interpolation_points))
    
    # Variational forms
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    a = ufl.inner(u, v) * ufl.dx + dt * kappa * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = ufl.inner(u_n, v) * ufl.dx + dt * ufl.inner(f_func, v) * ufl.dx
    
    # Solve
    problem = petsc.LinearProblem(a, L, bcs=[bc],
                                  petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
                                  petsc_options_prefix="debug_")
    u_sol = problem.solve()
    
    # Compute exact at new time
    u_exact = fem.Function(V)
    u_exact.interpolate(fem.Expression(u_exact_bc, V.element.interpolation_points))
    
    error = np.max(np.abs(u_sol.x.array - u_exact.x.array))
    print(f"Step {step}: t={t:.4f}, max error={error:.2e}, u_max={np.max(u_sol.x.array):.6f}, exact_max={np.max(u_exact.x.array):.6f}")
    
    # Update
    u_n.x.array[:] = u_sol.x.array

print(f"\nFinal u range: min={np.min(u_sol.x.array):.6f}, max={np.max(u_sol.x.array):.6f}")
print(f"Exact u range at t={t}: min={np.min(u_exact.x.array):.6f}, max={np.max(u_exact.x.array):.6f}")
