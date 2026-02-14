import numpy as np
from solver import solve

# Define exact solution function
def exact_solution_grid(x, y):
    return np.sin(2*np.pi*(x + y)) * np.sin(np.pi*(x - y))

# Create test case
case_spec = {
    "pde": {
        "epsilon": 0.05,
        "beta": [3.0, 1.0]
    },
    "domain": {
        "bounds": [[0.0, 0.0], [1.0, 1.0]]
    }
}

# Get solution
result = solve(case_spec)
u_numeric = result['u']

# Create grid for exact solution
nx, ny = 50, 50
x_min, y_min = 0.0, 0.0
x_max, y_max = 1.0, 1.0
x_vals = np.linspace(x_min + 1e-6, x_max - 1e-6, nx)
y_vals = np.linspace(y_min + 1e-6, y_max - 1e-6, ny)

# Compute exact solution on grid
X, Y = np.meshgrid(x_vals, y_vals, indexing='ij')
u_exact = exact_solution_grid(X, Y)

# Compute errors
abs_error = np.abs(u_numeric - u_exact)
max_error = np.max(abs_error)
mean_error = np.mean(abs_error)
l2_error = np.sqrt(np.mean((u_numeric - u_exact)**2))

print(f"Accuracy test for convection-diffusion problem")
print(f"==============================================")
print(f"Max absolute error: {max_error:.6e}")
print(f"Mean absolute error: {mean_error:.6e}")
print(f"L2 error: {l2_error:.6e}")
print(f"Required accuracy: ≤ 6.33e-03")
print(f"Pass max error test: {max_error <= 6.33e-03}")
print(f"Pass L2 error test: {l2_error <= 6.33e-03}")

# Also test with different mesh resolutions to see convergence
print(f"\nTesting different mesh resolutions:")
for N in [32, 64, 128]:
    # Modify solver to use specific resolution (simplified)
    import numpy as np
    from mpi4py import MPI
    from dolfinx import mesh, fem
    import ufl
    from dolfinx.fem import petsc
    from petsc4py import PETSc
    
    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", 1))
    
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    x = ufl.SpatialCoordinate(domain)
    
    epsilon = 0.05
    beta = ufl.as_vector([3.0, 1.0])
    
    u_exact_ufl = ufl.sin(2*ufl.pi*(x[0] + x[1])) * ufl.sin(ufl.pi*(x[0] - x[1]))
    f_expr = -epsilon * ufl.div(ufl.grad(u_exact_ufl)) + ufl.dot(beta, ufl.grad(u_exact_ufl))
    
    # SUPG
    h = ufl.CellDiameter(domain)
    beta_norm_ufl = ufl.sqrt(ufl.dot(beta, beta))
    from ufl import conditional, lt
    tau = conditional(lt(beta_norm_ufl, 1e-12), 
                     h**2 / (4 * epsilon),
                     h / (2 * beta_norm_ufl) * (1 / ufl.tanh(beta_norm_ufl * h / (2 * epsilon)) - 
                                               (2 * epsilon) / (beta_norm_ufl * h)))
    
    a = epsilon * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
    a += ufl.inner(ufl.dot(beta, ufl.grad(u)), v) * ufl.dx
    a += tau * ufl.inner(ufl.dot(beta, ufl.grad(u)), ufl.dot(beta, ufl.grad(v))) * ufl.dx
    
    L = ufl.inner(f_expr, v) * ufl.dx
    L += tau * ufl.inner(f_expr, ufl.dot(beta, ufl.grad(v))) * ufl.dx
    
    # BCs
    def boundary_marker(x):
        return np.logical_or.reduce([
            np.isclose(x[0], 0.0),
            np.isclose(x[0], 1.0),
            np.isclose(x[1], 0.0),
            np.isclose(x[1], 1.0)
        ])
    
    tdim = domain.topology.dim
    fdim = tdim - 1
    boundary_facets = mesh.locate_entities_boundary(domain, fdim, boundary_marker)
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    
    u_bc = fem.Function(V)
    u_bc.interpolate(lambda x: np.sin(2*np.pi*(x[0] + x[1])) * np.sin(np.pi*(x[0] - x[1])))
    bc = fem.dirichletbc(u_bc, boundary_dofs)
    
    problem = petsc.LinearProblem(
        a, L, bcs=[bc],
        petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
        petsc_options_prefix=f"test_{N}_"
    )
    
    u_sol = problem.solve()
    
    # Compute L2 error on mesh
    error_form = fem.form(ufl.inner(u_sol - u_exact_ufl, u_sol - u_exact_ufl) * ufl.dx)
    error = np.sqrt(fem.assemble_scalar(error_form))
    
    print(f"  N={N}: L2 error = {error:.6e}")
