import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem
import ufl
from dolfinx.fem import petsc
from petsc4py import PETSc

comm = MPI.COMM_WORLD
eps = 0.03
beta = np.array([5.0, 2.0])

def exact_solution(x):
    return np.sin(np.pi * x[0]) * np.sin(2 * np.pi * x[1])

def source_term(x):
    x_coord = x[0]
    y_coord = x[1]
    sin_pi_x = np.sin(np.pi * x_coord)
    sin_2pi_y = np.sin(2 * np.pi * y_coord)
    cos_pi_x = np.cos(np.pi * x_coord)
    cos_2pi_y = np.cos(2 * np.pi * y_coord)
    laplacian_u = -5 * (np.pi**2) * sin_pi_x * sin_2pi_y
    grad_u_x = np.pi * cos_pi_x * sin_2pi_y
    grad_u_y = 2 * np.pi * sin_pi_x * cos_2pi_y
    return -eps * laplacian_u + beta[0] * grad_u_x + beta[1] * grad_u_y

for N in [64, 128, 256]:
    domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
    degree = 2
    V = fem.functionspace(domain, ("Lagrange", degree))
    
    # BC
    def boundary_marker(x):
        return np.ones(x.shape[1], dtype=bool)
    tdim = domain.topology.dim
    fdim = tdim - 1
    boundary_facets = mesh.locate_entities_boundary(domain, fdim, boundary_marker)
    dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    u_bc = fem.Function(V)
    u_bc.interpolate(lambda x: exact_solution(x))
    bc = fem.dirichletbc(u_bc, dofs)
    
    # Forms
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    h = ufl.CellDiameter(domain)
    beta_ufl = ufl.as_vector([beta[0], beta[1]])
    beta_norm_ufl = ufl.sqrt(beta[0]**2 + beta[1]**2)
    Pe = beta_norm_ufl * h / (2 * eps)
    tau = (h / (2 * beta_norm_ufl)) * (1 / ufl.tanh(Pe) - 1 / Pe)
    tau_safe = ufl.conditional(Pe > 1e-10, tau, 0.0)
    
    a = (eps * ufl.inner(ufl.grad(u), ufl.grad(v)) 
         + ufl.inner(ufl.dot(beta_ufl, ufl.grad(u)), v)
         + tau_safe * ufl.inner(ufl.dot(beta_ufl, ufl.grad(u)), ufl.dot(beta_ufl, ufl.grad(v)))) * ufl.dx
    
    f = fem.Function(V)
    f.interpolate(lambda x: source_term(x))
    L = (ufl.inner(f, v) + tau_safe * ufl.inner(f, ufl.dot(beta_ufl, ufl.grad(v)))) * ufl.dx
    
    problem = petsc.LinearProblem(
        a, L, bcs=[bc],
        petsc_options={"ksp_type": "gmres", "pc_type": "hypre", "ksp_rtol": 1e-8},
        petsc_options_prefix="test_"
    )
    u_sol = problem.solve()
    
    # Compute L2 error
    error = u_sol - u_bc  # u_bc is exact solution interpolated
    error_form = fem.form(ufl.inner(error, error) * ufl.dx)
    error_l2 = np.sqrt(comm.allreduce(fem.assemble_scalar(error_form), op=MPI.SUM))
    print(f"N={N}, L2 error={error_l2:.2e}")
