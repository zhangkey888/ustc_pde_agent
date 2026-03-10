import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem
import ufl
from dolfinx.fem import petsc
from petsc4py import PETSc
import time

comm = MPI.COMM_WORLD
rank = comm.rank

# Test N=256, degree=3
N = 256
degree = 3

if rank == 0:
    print(f"Testing N={N}, degree={degree}")

start = time.time()
domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
V = fem.functionspace(domain, ("Lagrange", degree))

# Simple problem setup
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

def source_function(x):
    return 10.0 * np.exp(-80.0 * ((x[0] - 0.35)**2 + (x[1] - 0.55)**2))
f = fem.Function(V)
f.interpolate(source_function)

u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)
k = 15.0
a = ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx - (k**2) * ufl.inner(u, v) * ufl.dx
L = ufl.inner(f, v) * ufl.dx

problem = petsc.LinearProblem(
    a, L, bcs=[bc],
    petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
    petsc_options_prefix="test_"
)
u_h = problem.solve()

solve_time = time.time() - start
if rank == 0:
    print(f"Solve time: {solve_time:.2f}s")
    total_dofs = V.dofmap.index_map.size_global * V.dofmap.index_map_bs
    print(f"Total DOFs: {total_dofs}")
