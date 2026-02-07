import numpy as np
from mpi4py import MPI
import ufl
from dolfinx import fem, mesh, default_scalar_type
from dolfinx.fem.petsc import LinearProblem
import sys

comm = MPI.COMM_WORLD
domain = mesh.create_unit_square(comm, 32, 32, mesh.CellType.triangle)
V = fem.functionspace(domain, ("Lagrange", 1, (2,)))
x = ufl.SpatialCoordinate(domain)
pi = np.pi

# Exact solution
u_exact_ufl = ufl.as_vector([pi * ufl.cos(pi * x[1]) * ufl.sin(pi * x[0]),
                             -pi * ufl.cos(pi * x[0]) * ufl.sin(pi * x[1])])

# Source term
f_expr = ufl.as_vector([2 * pi**3 * ufl.cos(pi * x[1]) * ufl.sin(pi * x[0]),
                       -2 * pi**3 * ufl.cos(pi * x[0]) * ufl.sin(pi * x[1])])

# Boundary condition
u_exact_func = fem.Function(V)
def u_exact_eval(x):
    return np.vstack((pi * np.cos(pi * x[1]) * np.sin(pi * x[0]),
                     -pi * np.cos(pi * x[0]) * np.sin(pi * x[1])))

u_exact_func.interpolate(u_exact_eval)

def boundary(x):
    return np.logical_or.reduce((
        np.isclose(x[0], 0.0),
        np.isclose(x[0], 1.0),
        np.isclose(x[1], 0.0),
        np.isclose(x[1], 1.0)
    ))

boundary_dofs = fem.locate_dofs_geometrical(V, boundary)
bc = fem.dirichletbc(u_exact_func, boundary_dofs)

# Variational problem
u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)
κ = default_scalar_type(1.0)
a = κ * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
L = ufl.inner(f_expr, v) * ufl.dx

problem = LinearProblem(a, L, bcs=[bc], petsc_options_prefix="test_",
                       petsc_options={
                           "ksp_type": "cg",
                           "pc_type": "hypre",
                           "ksp_rtol": 1e-10,
                           "ksp_atol": 1e-12,
                       })

uh = problem.solve()

# Compute L2 error
error_form = ufl.inner(uh - u_exact_ufl, uh - u_exact_ufl) * ufl.dx
error_local = fem.assemble_scalar(error_form)
error_global = domain.comm.allreduce(error_local, op=MPI.SUM)
error_l2 = np.sqrt(error_global)

if comm.rank == 0:
    print(f"L2 error: {error_l2}")
    if error_l2 < 0.01:
        print("✓ Error is below target 0.01")
    else:
        print(f"✗ Error {error_l2} is above target 0.01")
        sys.exit(1)
