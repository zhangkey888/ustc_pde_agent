import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem
from dolfinx.fem import petsc
import ufl
import basix.ufl
from petsc4py import PETSc

comm = MPI.COMM_WORLD
nu_val = 0.1
N = 32
degree_u = 3
degree_p = 2

domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)

e_u = basix.ufl.element("Lagrange", "triangle", degree_u, shape=(domain.geometry.dim,))
e_p = basix.ufl.element("Lagrange", "triangle", degree_p)
mel = basix.ufl.mixed_element([e_u, e_p])
W = fem.functionspace(domain, mel)

w = fem.Function(W)
(v, q) = ufl.TestFunctions(W)
(u, p) = ufl.split(w)

x = ufl.SpatialCoordinate(domain)
pi = ufl.pi

u_exact = ufl.as_vector([
    2*pi*ufl.cos(2*pi*x[1])*ufl.sin(2*pi*x[0]),
    -2*pi*ufl.cos(2*pi*x[0])*ufl.sin(2*pi*x[1])
])
p_exact = ufl.sin(2*pi*x[0])*ufl.cos(2*pi*x[1])

f = (ufl.grad(u_exact) * u_exact 
     - nu_val * ufl.div(ufl.grad(u_exact)) 
     + ufl.grad(p_exact))

F = (
    nu_val * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
    + ufl.inner(ufl.grad(u) * u, v) * ufl.dx
    - p * ufl.div(v) * ufl.dx
    + q * ufl.div(u) * ufl.dx
    - ufl.inner(f, v) * ufl.dx
)

tdim = domain.topology.dim
fdim = tdim - 1
boundary_facets = mesh.locate_entities_boundary(domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool))

V_col, _ = W.sub(0).collapse()
u_bc_func = fem.Function(V_col)
u_bc_func.interpolate(lambda x: np.stack([
    2*np.pi*np.cos(2*np.pi*x[1])*np.sin(2*np.pi*x[0]),
    -2*np.pi*np.cos(2*np.pi*x[0])*np.sin(2*np.pi*x[1])
]))

dofs_u = fem.locate_dofs_topological((W.sub(0), V_col), fdim, boundary_facets)
bc_u = fem.dirichletbc(u_bc_func, dofs_u, W.sub(0))
bcs = [bc_u]

# Initial guess
w.sub(0).interpolate(lambda x: np.stack([
    2*np.pi*np.cos(2*np.pi*x[1])*np.sin(2*np.pi*x[0]),
    -2*np.pi*np.cos(2*np.pi*x[0])*np.sin(2*np.pi*x[1])
]))
w.sub(1).interpolate(lambda x: np.sin(2*np.pi*x[0])*np.cos(2*np.pi*x[1]))

petsc_options = {
    "ksp_type": "preonly",
    "pc_type": "lu",
    "pc_factor_mat_solver_type": "mumps",
    "snes_rtol": 1e-10,
    "snes_atol": 1e-12,
    "snes_max_it": 30,
    "snes_linesearch_type": "basic",
    "snes_monitor": None,
}

problem = petsc.NonlinearProblem(
    F, w, bcs=bcs,
    petsc_options_prefix="ns_",
    petsc_options=petsc_options,
)

problem.solve()
snes = problem.solver
n_newton = snes.getIterationNumber()
reason = snes.getConvergedReason()
print(f"SNES converged reason: {reason}, iterations: {n_newton}")
