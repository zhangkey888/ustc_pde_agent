import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
import basix.ufl
from petsc4py import PETSc
import time

comm = MPI.COMM_WORLD
nu_val = 0.1
N = 64
degree_u = 3
degree_p = 2

t0 = time.time()

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
    petsc_options_prefix="ns3_",
    petsc_options=petsc_options,
)

problem.solve()
snes = problem.solver
n_newton = snes.getIterationNumber()
reason = snes.getConvergedReason()
print(f"SNES converged reason: {reason}, iterations: {n_newton}")

w.x.scatter_forward()
u_sol = w.sub(0).collapse()

nx_eval, ny_eval = 50, 50
xs = np.linspace(0, 1, nx_eval)
ys = np.linspace(0, 1, ny_eval)
XX, YY = np.meshgrid(xs, ys, indexing='ij')

points_3d = np.zeros((3, nx_eval * ny_eval))
points_3d[0, :] = XX.ravel()
points_3d[1, :] = YY.ravel()

bb_tree = geometry.bb_tree(domain, domain.topology.dim)
cell_candidates = geometry.compute_collisions_points(bb_tree, points_3d.T)
colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points_3d.T)

points_on_proc = []
cells_on_proc = []
eval_map = []
for i in range(points_3d.shape[1]):
    links = colliding_cells.links(i)
    if len(links) > 0:
        points_on_proc.append(points_3d[:, i])
        cells_on_proc.append(links[0])
        eval_map.append(i)

n_pts = points_3d.shape[1]
u_values = np.full((n_pts, domain.geometry.dim), np.nan)

if len(points_on_proc) > 0:
    pts_arr = np.array(points_on_proc)
    cells_arr = np.array(cells_on_proc, dtype=np.int32)
    vals = u_sol.eval(pts_arr, cells_arr)
    u_values[eval_map, :] = vals.reshape(-1, domain.geometry.dim)

vel_mag = np.sqrt(u_values[:, 0]**2 + u_values[:, 1]**2)
vel_mag_grid = vel_mag.reshape(nx_eval, ny_eval)

ux_exact = 2*np.pi*np.cos(2*np.pi*YY)*np.sin(2*np.pi*XX)
uy_exact = -2*np.pi*np.cos(2*np.pi*XX)*np.sin(2*np.pi*YY)
vel_mag_exact = np.sqrt(ux_exact**2 + uy_exact**2)

error = np.sqrt(np.mean((vel_mag_grid - vel_mag_exact)**2))
max_error = np.max(np.abs(vel_mag_grid - vel_mag_exact))
elapsed = time.time() - t0
print(f"Time: {elapsed:.3f}s, RMS error: {error:.2e}, Max error: {max_error:.2e}")
