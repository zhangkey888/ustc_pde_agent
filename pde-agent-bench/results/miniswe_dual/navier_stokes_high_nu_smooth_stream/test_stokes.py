import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem
from dolfinx.fem import petsc
import basix.ufl
import ufl
from petsc4py import PETSc

comm = MPI.COMM_WORLD
N = 8
domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
gdim = domain.geometry.dim
cell_name = domain.topology.cell_name()

vel_el = basix.ufl.element("Lagrange", cell_name, 2, shape=(gdim,))
pres_el = basix.ufl.element("Lagrange", cell_name, 1)
mel = basix.ufl.mixed_element([vel_el, pres_el])

W = fem.functionspace(domain, mel)
V = fem.functionspace(domain, basix.ufl.element("Lagrange", cell_name, 2, shape=(gdim,)))
Q = fem.functionspace(domain, basix.ufl.element("Lagrange", cell_name, 1))

# Simple Stokes: -Δu + ∇p = f, ∇·u = 0
u = ufl.TrialFunction(W)
(v, q) = ufl.TestFunctions(W)

# Split trial function
(u_trial, p_trial) = ufl.split(u)

x = ufl.SpatialCoordinate(domain)
f = ufl.as_vector([1.0, 0.0])

a = (ufl.inner(ufl.grad(u_trial), ufl.grad(v)) * ufl.dx
     - p_trial * ufl.div(v) * ufl.dx
     + ufl.div(u_trial) * q * ufl.dx)
L = ufl.inner(f, v) * ufl.dx

# BCs
tdim = domain.topology.dim
fdim = tdim - 1
boundary_facets = mesh.locate_entities_boundary(domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool))

u_bc_func = fem.Function(V)
u_bc_func.x.array[:] = 0.0
dofs_u = fem.locate_dofs_topological((W.sub(0), V), fdim, boundary_facets)
bc_u = fem.dirichletbc(u_bc_func, dofs_u, W.sub(0))

# Pin pressure
p_bc_func = fem.Function(Q)
p_bc_func.x.array[:] = 0.0
corner_dofs = fem.locate_dofs_geometrical((W.sub(1), Q), lambda x: np.isclose(x[0], 0.0) & np.isclose(x[1], 0.0))
bc_p = fem.dirichletbc(p_bc_func, corner_dofs, W.sub(1))

bcs = [bc_u, bc_p]

problem = petsc.LinearProblem(
    a, L, bcs=bcs,
    petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
    petsc_options_prefix="stokes_"
)
wh = problem.solve()
print(f"Solution norm: {np.linalg.norm(wh.x.array)}")
print(f"Has NaN: {np.any(np.isnan(wh.x.array))}")
print(f"Has Inf: {np.any(np.isinf(wh.x.array))}")
print("Stokes test passed!")
