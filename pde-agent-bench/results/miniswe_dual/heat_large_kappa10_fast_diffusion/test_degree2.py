import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem
import ufl
from dolfinx.fem import petsc
from petsc4py import PETSc
import time

comm = MPI.COMM_WORLD
rank = comm.rank

# Test with 128x128 mesh, degree 2
N = 128
degree = 2
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
    print(f"Time for N={N}, degree={degree}: {end-start:.3f}s")
    print(f"Total iterations: {total_iterations}")
    
    # Evaluate error on 50x50 grid
    from dolfinx import geometry
    nx, ny = 50, 50
    x_vals = np.linspace(0, 1, nx)
    y_vals = np.linspace(0, 1, ny)
    points = np.array([[x, y, 0.0] for x in x_vals for y in y_vals]).T
    
    # Evaluate function
    bb_tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, points.T)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points.T)
    
    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    for i in range(points.shape[1]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(points.T[i])
            cells_on_proc.append(links[0])
            eval_map.append(i)
    
    u_values = np.full(points.shape[1], np.nan)
    if len(points_on_proc) > 0:
        vals = u_sol.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u_values[eval_map] = vals.flatten()
    
    u_grid = u_values.reshape((nx, ny))
    
    # Exact solution
    X, Y = np.meshgrid(x_vals, y_vals, indexing='ij')
    u_exact_grid = np.exp(-t_end) * np.sin(np.pi * X) * np.sin(np.pi * Y)
    
    l2_error = np.sqrt(np.mean((u_grid - u_exact_grid)**2))
    print(f"Grid L2 error: {l2_error:.2e}")
