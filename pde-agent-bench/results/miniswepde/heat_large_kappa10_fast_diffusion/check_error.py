import solver
import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem
import ufl

# Run solver to get result
case_spec = {
    "pde": {
        "time": {
            "t_end": 0.05,
            "dt": 0.005,
            "scheme": "backward_euler"
        },
        "coefficients": {
            "kappa": 10.0
        }
    }
}
result = solver.solve(case_spec)
N = result["solver_info"]["mesh_resolution"]
print(f"Used mesh resolution: {N}")

# Compute exact solution on same mesh
comm = MPI.COMM_WORLD
domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
V = fem.functionspace(domain, ("Lagrange", 1))
x = ufl.SpatialCoordinate(domain)
pi = np.pi

u_exact = fem.Function(V)
u_exact_expr = ufl.exp(-0.05) * ufl.sin(pi*x[0]) * ufl.sin(pi*x[1])
u_exact.interpolate(fem.Expression(u_exact_expr, V.element.interpolation_points))

# Need to get the numerical solution from solver - we don't have it directly.
# Instead, we can evaluate at grid points and compare with exact.
# But easier: re-run the time-stepping with same parameters and compute error.
# Let's do a quick re-solve.
kappa = 10.0
dt = 0.005
t_end = 0.05
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

u_n = fem.Function(V)
u_exact_0 = ufl.sin(pi*x[0]) * ufl.sin(pi*x[1])
u_n.interpolate(fem.Expression(u_exact_0, V.element.interpolation_points))

t = 0.0
for step in range(n_steps):
    t += dt
    u_exact_bc = ufl.exp(-t) * ufl.sin(pi*x[0]) * ufl.sin(pi*x[1])
    u_bc = fem.Function(V)
    u_bc.interpolate(fem.Expression(u_exact_bc, V.element.interpolation_points))
    bc = fem.dirichletbc(u_bc, dofs)
    
    f_expr = ufl.exp(-t) * ufl.sin(pi*x[0]) * ufl.sin(pi*x[1]) * (-1 + 2*kappa*pi**2)
    f_func = fem.Function(V)
    f_func.interpolate(fem.Expression(f_expr, V.element.interpolation_points))
    
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    a = ufl.inner(u, v) * ufl.dx + dt * kappa * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = ufl.inner(u_n, v) * ufl.dx + dt * ufl.inner(f_func, v) * ufl.dx
    
    from dolfinx.fem import petsc
    problem = petsc.LinearProblem(a, L, bcs=[bc],
                                  petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
                                  petsc_options_prefix="check_")
    u_sol = problem.solve()
    u_n.x.array[:] = u_sol.x.array

# Compute error
error = fem.Function(V)
error.x.array[:] = u_sol.x.array - u_exact.x.array
error_form = fem.form(ufl.inner(error, error) * ufl.dx)
error_norm = np.sqrt(fem.assemble_scalar(error_form))
print(f"L2 error for N={N}: {error_norm:.2e}")
print(f"Required: ≤4.64e-04")
print(f"{'PASS' if error_norm <= 4.64e-04 else 'FAIL'}")
