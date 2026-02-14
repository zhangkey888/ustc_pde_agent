import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem
import ufl
from dolfinx.fem import petsc
from petsc4py import PETSc

comm = MPI.COMM_WORLD
rank = comm.rank

def u_exact(x, t):
    return np.exp(-t) * np.exp(-40 * ((x[0] - 0.5)**2 + (x[1] - 0.5)**2))

def f_source(x, t):
    u = u_exact(x, t)
    du_dt = -u
    r2 = (x[0] - 0.5)**2 + (x[1] - 0.5)**2
    laplacian_u = u * (-80 + 1600 * r2)
    return du_dt - laplacian_u

N = 32
domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
V = fem.functionspace(domain, ("Lagrange", 1))

# Boundary condition dofs
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

t_end = 0.1
u_exact_func = fem.Function(V)
u_exact_func.interpolate(lambda x: u_exact(x, t_end))

f_func = fem.Function(V)
f_func.interpolate(lambda x: f_source(x, t_end))

# Solve -Δu = f + u_exact (since -Δu = f - du/dt, and du/dt = -u_exact)
g_func = fem.Function(V)
g_func.x.array[:] = f_func.x.array + u_exact_func.x.array

# Solve -Δu = g
u = fem.Function(V)
v = ufl.TestFunction(V)
u_trial = ufl.TrialFunction(V)
kappa = fem.Constant(domain, PETSc.ScalarType(1.0))

a = ufl.inner(kappa * ufl.grad(u_trial), ufl.grad(v)) * ufl.dx
L = ufl.inner(g_func, v) * ufl.dx

# Dirichlet BC
u_bc = fem.Function(V)
u_bc.interpolate(lambda x: u_exact(x, t_end))
bc = fem.dirichletbc(u_bc, dofs)

problem = petsc.LinearProblem(a, L, bcs=[bc], petsc_options={"ksp_type": "preonly", "pc_type": "lu"}, petsc_options_prefix="test_")
u_sol = problem.solve()

# Compute error
error_form = fem.form(ufl.inner(u_sol - u_exact_func, u_sol - u_exact_func) * ufl.dx)
error_sq = fem.assemble_scalar(error_form)
error_l2 = np.sqrt(error_sq)

norm_form = fem.form(ufl.inner(u_exact_func, u_exact_func) * ufl.dx)
norm_sq = fem.assemble_scalar(norm_form)
norm_exact = np.sqrt(norm_sq)

if rank == 0:
    print(f"Steady solve error: {error_l2:.6e}")
    print(f"Relative error: {error_l2/norm_exact:.6e}")
    print(f"Max of sol: {np.max(u_sol.x.array):.6f}, max exact: {np.max(u_exact_func.x.array):.6f}")
    # Check center
    points = np.array([[0.5, 0.5, 0.0]])
    from dolfinx import geometry
    bb_tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, points)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points)
    cells = []
    for i in range(points.shape[0]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            cells.append(links[0])
    if cells:
        val = u_sol.eval(points, np.array(cells, dtype=np.int32))
        exact_val = u_exact(np.array([0.5, 0.5]), t_end)
        print(f"Center: sol={val[0]:.6f}, exact={exact_val:.6f}, diff={abs(val[0]-exact_val):.6e}")
