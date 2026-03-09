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

print(f"W ndofs: {W.dofmap.index_map.size_global}")
print(f"V ndofs: {V.dofmap.index_map.size_global}")
print(f"Q ndofs: {Q.dofmap.index_map.size_global}")

# Use TrialFunctions and TestFunctions
(u, p) = ufl.TrialFunctions(W)
(v, q) = ufl.TestFunctions(W)

x = ufl.SpatialCoordinate(domain)
f = ufl.as_vector([1.0, 0.0])

a = (ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
     - p * ufl.div(v) * ufl.dx
     + ufl.div(u) * q * ufl.dx)
L = ufl.inner(f, v) * ufl.dx

# BCs - only velocity
tdim = domain.topology.dim
fdim = tdim - 1
boundary_facets = mesh.locate_entities_boundary(domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool))

u_bc_func = fem.Function(V)
u_bc_func.x.array[:] = 0.0
dofs_u = fem.locate_dofs_topological((W.sub(0), V), fdim, boundary_facets)
bc_u = fem.dirichletbc(u_bc_func, dofs_u, W.sub(0))

bcs = [bc_u]

# Try solving
a_compiled = fem.form(a)
L_compiled = fem.form(L)

A = petsc.assemble_matrix(a_compiled, bcs=bcs)
A.assemble()

# Check matrix properties
print(f"Matrix size: {A.getSize()}")
print(f"Matrix norm: {A.norm()}")

b = petsc.assemble_vector(L_compiled)
petsc.apply_lifting(b, [a_compiled], bcs=[bcs])
b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
petsc.set_bc(b, bcs)

print(f"RHS norm: {b.norm()}")

# Solve
ksp = PETSc.KSP().create(comm)
ksp.setType(PETSc.KSP.Type.PREONLY)
pc = ksp.getPC()
pc.setType(PETSc.PC.Type.LU)
pc.setFactorSolverType("mumps")
ksp.setOperators(A)

wh = fem.Function(W)
ksp.solve(b, wh.x.petsc_vec)
wh.x.scatter_forward()

print(f"Solution norm: {np.linalg.norm(wh.x.array)}")
print(f"Has NaN: {np.any(np.isnan(wh.x.array))}")
print(f"Has Inf: {np.any(np.isinf(wh.x.array))}")
print(f"KSP converged reason: {ksp.getConvergedReason()}")

# Check if solution is reasonable
print(f"Max abs value: {np.max(np.abs(wh.x.array))}")
