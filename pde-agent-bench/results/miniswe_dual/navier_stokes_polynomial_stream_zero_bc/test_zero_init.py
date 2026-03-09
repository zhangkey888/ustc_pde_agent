import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import basix.ufl
import ufl
from petsc4py import PETSc
import time

comm = MPI.COMM_WORLD
nu_val = 0.25
N = 16
degree_u = 3
degree_p = 2

domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
gdim = domain.geometry.dim

vel_el = basix.ufl.element("Lagrange", domain.topology.cell_name(), degree_u, shape=(gdim,))
pres_el = basix.ufl.element("Lagrange", domain.topology.cell_name(), degree_p)
mel = basix.ufl.mixed_element([vel_el, pres_el])

W = fem.functionspace(domain, mel)
V = fem.functionspace(domain, ("Lagrange", degree_u, (gdim,)))
Q = fem.functionspace(domain, ("Lagrange", degree_p))

w = fem.Function(W)
(u, p) = ufl.split(w)
(v, q) = ufl.TestFunctions(W)

x = ufl.SpatialCoordinate(domain)
u_exact = ufl.as_vector([x[0]*(1 - x[0])*(1 - 2*x[1]), -x[1]*(1 - x[1])*(1 - 2*x[0])])
p_exact = x[0] - x[1]
nu = fem.Constant(domain, PETSc.ScalarType(nu_val))
f = ufl.grad(u_exact) * u_exact - nu * ufl.div(ufl.grad(u_exact)) + ufl.grad(p_exact)

F_form = (
    ufl.inner(ufl.grad(u) * u, v) * ufl.dx
    + nu * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
    - p * ufl.div(v) * ufl.dx
    + ufl.div(u) * q * ufl.dx
    - ufl.inner(f, v) * ufl.dx
)

tdim = domain.topology.dim
fdim = tdim - 1
boundary_facets = mesh.locate_entities_boundary(domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool))

u_bc_func = fem.Function(V)
u_bc_func.interpolate(lambda x: np.stack([x[0]*(1 - x[0])*(1 - 2*x[1]), -x[1]*(1 - x[1])*(1 - 2*x[0])]))
dofs_u = fem.locate_dofs_topological((W.sub(0), V), fdim, boundary_facets)
bc_u = fem.dirichletbc(u_bc_func, dofs_u, W.sub(0))

p_bc_func = fem.Function(Q)
p_bc_func.interpolate(lambda x: x[0] - x[1])
origin_vertices = mesh.locate_entities_boundary(domain, 0, lambda x: np.isclose(x[0], 0.0) & np.isclose(x[1], 0.0))
dofs_p = fem.locate_dofs_topological((W.sub(1), Q), 0, origin_vertices)
bc_p = fem.dirichletbc(p_bc_func, dofs_p, W.sub(1))
bcs = [bc_u, bc_p]

# Start from ZERO initial guess
w.x.array[:] = 0.0

t0 = time.time()
problem = petsc.NonlinearProblem(
    F_form, w, bcs=bcs,
    petsc_options_prefix="ns_",
    petsc_options={
        "snes_type": "newtonls",
        "snes_rtol": 1e-10,
        "snes_atol": 1e-12,
        "snes_max_it": 30,
        "ksp_type": "preonly",
        "pc_type": "lu",
        "snes_monitor": None,
    }
)
problem.solve()
elapsed = time.time() - t0

snes = problem.solver
print(f"Newton iterations: {snes.getIterationNumber()}")
print(f"Converged reason: {snes.getConvergedReason()}")
print(f"Time: {elapsed:.3f}s")
