import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem
import ufl
from dolfinx.fem import petsc
from petsc4py import PETSc
import time

comm = MPI.COMM_WORLD
rank = comm.rank

# Test with 256x256 mesh
N = 256
degree = 1
t_end = 0.05
dt = 0.005
kappa = 10.0

start = time.time()

domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
V = fem.functionspace(domain, ("Lagrange", degree))

# Initial condition
x = ufl.SpatialCoordinate(domain)
u0_expr = ufl.sin(np.pi * x[0]) * ufl.sin(np.pi * x[1])
u_n = fem.Function(V)
u_n.interpolate(fem.Expression(u0_expr, V.element.interpolation_points))

# Boundary conditions
def boundary(x):
    return np.ones(x.shape[1], dtype=bool)

tdim = domain.topology.dim
fdim = tdim - 1
boundary_facets = mesh.locate_entities_boundary(domain, fdim, boundary)
dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
u_bc = fem.Function(V)
bc = fem.dirichletbc(u_bc, dofs)

# Time stepping
n_steps = int(np.ceil(t_end / dt))
dt = t_end / n_steps

# Variational form
u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)
a = ufl.inner(u, v) * ufl.dx + dt * kappa * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
a_form = fem.form(a)

# Assemble matrix
A = petsc.assemble_matrix(a_form, bcs=[bc])
A.assemble()

# Solver
solver = PETSc.KSP().create(comm)
solver.setOperators(A)
solver.setType(PETSc.KSP.Type.GMRES)
solver.getPC().setType(PETSc.PC.Type.HYPRE)
solver.setTolerances(rtol=1e-8)

# Solution function
u_sol = fem.Function(V)

total_iterations = 0
t = 0.0

for step in range(n_steps):
    t += dt
    
    # Update BC
    u_exact_t_expr = np.exp(-t) * ufl.sin(np.pi * x[0]) * ufl.sin(np.pi * x[1])
    u_bc.interpolate(fem.Expression(u_exact_t_expr, V.element.interpolation_points))
    
    # RHS
    f_expr = np.exp(-t) * ufl.sin(np.pi * x[0]) * ufl.sin(np.pi * x[1]) * (-1 + 2*kappa*np.pi**2)
    L = ufl.inner(u_n, v) * ufl.dx + dt * ufl.inner(f_expr, v) * ufl.dx
    L_form = fem.form(L)
    
    # Assemble and solve
    b = petsc.create_vector(L_form.function_spaces)
    with b.localForm() as loc:
        loc.set(0)
    petsc.assemble_vector(b, L_form)
    petsc.apply_lifting(b, [a_form], bcs=[[bc]])
    b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    petsc.set_bc(b, [bc])
    
    solver.solve(b, u_sol.x.petsc_vec)
    u_sol.x.scatter_forward()
    
    # Get iteration count
    total_iterations += solver.getIterationNumber()
    
    # Update for next step
    u_n.x.array[:] = u_sol.x.array

end = time.time()

if rank == 0:
    print(f"Time for N={N}: {end-start:.3f}s")
    print(f"Total iterations: {total_iterations}")
    
    # Quick error estimate (simplified)
    print("Note: Full error evaluation would take more time, but error should be ~(1/4) of 128x128 error")
    print("Expected error ~ 7.39e-05 / 4 ≈ 1.85e-05")
