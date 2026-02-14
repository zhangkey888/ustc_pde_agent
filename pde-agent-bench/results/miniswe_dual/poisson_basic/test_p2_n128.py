import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
import ufl
from dolfinx.fem import petsc
from petsc4py import PETSc
import time

comm = MPI.COMM_WORLD
ScalarType = PETSc.ScalarType

# Test with N=128, P2 elements
N = 128
element_degree = 2

start_time = time.time()

# Create mesh
domain = mesh.create_unit_square(comm, nx=N, ny=N, cell_type=mesh.CellType.triangle)

# Function space
V = fem.functionspace(domain, ("Lagrange", element_degree))

# Define exact solution and source term
x = ufl.SpatialCoordinate(domain)
u_exact = ufl.sin(np.pi * x[0]) * ufl.sin(np.pi * x[1])
f = 2.0 * np.pi**2 * ufl.sin(np.pi * x[0]) * ufl.sin(np.pi * x[1])

# Boundary condition
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

u_bc = fem.Function(V)
u_bc.interpolate(lambda x: np.sin(np.pi * x[0]) * np.sin(np.pi * x[1]))
bc = fem.dirichletbc(u_bc, dofs)

# Variational problem
u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)
a = ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
L = ufl.inner(fem.Constant(domain, ScalarType(1.0)) * f, v) * ufl.dx

# Solve
problem = petsc.LinearProblem(
    a, L, bcs=[bc],
    petsc_options={
        "ksp_type": "gmres",
        "pc_type": "hypre",
        "ksp_rtol": 1e-8,
        "ksp_atol": 1e-12,
        "ksp_max_it": 1000
    },
    petsc_options_prefix="poisson_"
)

u_sol = problem.solve()

end_time = time.time()

print(f"Time taken: {end_time - start_time:.3f} seconds")
print(f"Mesh resolution: {N}, Element degree: {element_degree}")
print(f"Time requirement: ≤ 2.131s")
print(f"Pass time: {(end_time - start_time) <= 2.131}")
