import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem
import ufl
from dolfinx.fem import petsc
from petsc4py import PETSc

comm = MPI.COMM_WORLD
domain = mesh.create_unit_square(comm, 32, 32, cell_type=mesh.CellType.triangle)
V = fem.functionspace(domain, ("Lagrange", 1))

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
u_bc.interpolate(lambda x: np.zeros_like(x[0]))
bc = fem.dirichletbc(u_bc, dofs)

# Test assembly
dt = 0.03
kappa = 1.0
v = ufl.TestFunction(V)
u = ufl.TrialFunction(V)
dt_constant = fem.Constant(domain, PETSc.ScalarType(dt))
kappa_constant = fem.Constant(domain, PETSc.ScalarType(kappa))

a = ufl.inner(u/dt_constant, v) * ufl.dx + ufl.inner(kappa_constant * ufl.grad(u), ufl.grad(v)) * ufl.dx
a_form = fem.form(a)

print("Assembling matrix...")
try:
    A = petsc.assemble_matrix(a_form, bcs=[bc])
    A.assemble()
    print("Success!")
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
