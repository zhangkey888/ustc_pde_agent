import numpy as np
import time
import sys
sys.path.insert(0, '.')

# Modify solver temporarily to test with finer mesh
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc

ScalarType = PETSc.ScalarType

def solve_test(N, dt_val):
    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", 1))
    
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    u_n = fem.Function(V)
    u_h = fem.Function(V)
    u_n.x.array[:] = 0.0
    
    x = ufl.SpatialCoordinate(domain)
    t = fem.Constant(domain, ScalarType(0.0))
    dt = fem.Constant(domain, ScalarType(dt_val))
    eps_c = fem.Constant(domain, ScalarType(0.02))
    beta = ufl.as_vector([ScalarType(6.0), ScalarType(2.0)])
    f = ufl.exp(-200.0 * ((x[0] - 0.3)**2 + (x[1] - 0.7)**2)) * ufl.exp(-t)
    
    tdim = domain.topology.dim
    fdim = tdim - 1
    boundary_facets = mesh.locate_entities_boundary(domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool))
    dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc = fem.dirichletbc(ScalarType(0.0), dofs, V)
    bcs = [bc]
    
    h = ufl.CellDiameter(domain)
    beta_norm = ufl.sqrt(ufl.dot(beta, beta))
    Pe_local = beta_norm * h / (2.0 * eps_c)
    tau_supg = h / (2.0 * beta_norm + 1e-10) * ufl.min_value(Pe_local / 3.0, 1.0)
    
    a_gal = (u / dt * v + eps_c * ufl.inner(ufl.grad(u), ufl.grad(v)) + ufl.dot(beta, ufl.grad(u)) * v) * ufl.dx
    L_gal = (u_n / dt * v + f * v) * ufl.dx
    
    v_supg = tau_supg * ufl.dot(beta, ufl.grad(v))
    a_supg = (u / dt * v_supg + eps_c * ufl.inner(ufl.grad(u), ufl.grad(v_supg)) + ufl.dot(beta, ufl.grad(u)) * v_supg) * ufl.dx
    L_supg = (u_n / dt * v_supg + f * v_supg) * ufl.dx
    
    a_form = fem.form(a_gal + a_supg)
    L_form = fem.form(L_gal + L_supg)
    
    A = petsc.assemble_matrix(a_form, bcs=bcs)
    A.assemble()
    b = petsc.create_vector(V)
    
    solver = PETSc.KSP().create(domain.comm)
    solver.setOperators(A)
    solver.setType(PETSc.KSP.Type.GMRES)
    solver.getPC().setType(PETSc.PC.Type.ILU)
    solver.setTolerances(rtol=1e-8, max_it=1000)
    solver.setUp()
    
    n_steps = int(np.round(0.1 / dt_val))
    for step in range(n_steps):
        t.value = (step + 1) * dt_val
        with b.localForm() as loc:
            loc.set(0.0)
        petsc.assemble_vector(b, L_form)
        petsc.apply_lifting(b, [a_form], bcs=[bcs])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        petsc.set_bc(b, bcs)
        solver.solve(b, u_h.x.petsc_vec)
        u_h.x.scatter_forward()
        u_n.x.array[:] = u_h.x.array[:]
    
    # Evaluate on 50x50 grid
    nx_out, ny_out = 50, 50
    xs = np.linspace(0.0, 1.0, nx_out)
    ys = np.linspace(0.0, 1.0, ny_out)
    XX, YY = np.meshgrid(xs, ys, indexing='ij')
    points = np.zeros((nx_out * ny_out, 3))
    points[:, 0] = XX.flatten()
    points[:, 1] = YY.flatten()
    
    bb_tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, points)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points)
    
    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    for i in range(len(points)):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(points[i])
            cells_on_proc.append(links[0])
            eval_map.append(i)
    
    u_values = np.zeros(nx_out * ny_out)
    if len(points_on_proc) > 0:
        vals = u_h.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u_values[eval_map] = vals.flatten()
    
    return u_values.reshape((nx_out, ny_out))

# Test convergence
configs = [(64, 0.02), (80, 0.02), (128, 0.02), (80, 0.005)]
results = {}
for N, dt_val in configs:
    t0 = time.time()
    u = solve_test(N, dt_val)
    elapsed = time.time() - t0
    results[(N, dt_val)] = u
    print(f"N={N}, dt={dt_val}: max={u.max():.6e}, L2={np.sqrt(np.sum(u**2)/u.size):.6e}, time={elapsed:.2f}s")

# Compare solutions
ref = results[(128, 0.02)]
for key, u in results.items():
    diff = np.sqrt(np.sum((u - ref)**2) / u.size)
    print(f"  {key} vs ref(128,0.02): RMS diff = {diff:.6e}")

ref2 = results[(80, 0.005)]
for key, u in results.items():
    diff = np.sqrt(np.sum((u - ref2)**2) / u.size)
    print(f"  {key} vs ref(80,0.005): RMS diff = {diff:.6e}")
