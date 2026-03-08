import time
import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem
import ufl
from dolfinx.fem import petsc
from petsc4py import PETSc

comm = MPI.COMM_WORLD
rank = comm.rank

# Test with mesh 256, degree 1
N = 256
degree = 1
k = 15.0

if rank == 0:
    print(f"Testing mesh N={N}, degree={degree}")

domain = mesh.create_unit_square(comm, N, N)
V = fem.functionspace(domain, ("Lagrange", degree))

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
bc = fem.dirichletbc(PETSc.ScalarType(0.0), dofs, V)

# Source term
def source_function(x):
    return 10.0 * np.exp(-80.0 * ((x[0] - 0.35)**2 + (x[1] - 0.55)**2))

f = fem.Function(V)
f.interpolate(source_function)

# Variational form
u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)
a = ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx - (k**2) * ufl.inner(u, v) * ufl.dx
L = ufl.inner(f, v) * ufl.dx

# Solve
start = time.time()
problem = petsc.LinearProblem(
    a, L, bcs=[bc],
    petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
    petsc_options_prefix="test_"
)
u_h = problem.solve()
end = time.time()

# Compute norm
norm_form = fem.form(ufl.inner(u_h, u_h) * ufl.dx)
norm = np.sqrt(fem.assemble_scalar(norm_form))

if rank == 0:
    print(f"  Time: {end - start:.3f} s")
    print(f"  L2 norm: {norm:.6e}")
    
    # Compare with N=128, degree=2 norm (from previous run: 2.315588e-02)
    norm_128_2 = 2.315588e-02
    rel_change = abs(norm - norm_128_2) / (norm + 1e-15)
    print(f"  Relative change from N=128,d=2: {rel_change:.6e}")
