import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem
import ufl
from dolfinx.fem import petsc
from petsc4py import PETSc

comm = MPI.COMM_WORLD
rank = comm.rank

def test_dt(N=64, degree=1, dt=0.01):
    # Create mesh
    domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", degree))
    
    # Manufactured solution
    def u_exact(x, t):
        return np.exp(-t) * np.exp(-40 * ((x[0] - 0.5)**2 + (x[1] - 0.5)**2))
    
    def f_source(x, t):
        r2 = (x[0] - 0.5)**2 + (x[1] - 0.5)**2
        u_val = np.exp(-t) * np.exp(-40 * r2)
        return -u_val * (1 + 1.0 * (6400 * r2 - 160))
    
    # Boundary condition
    u_bc = fem.Function(V)
    
    def boundary_marker(x):
        return np.logical_or.reduce([
            np.isclose(x[0], 0.0), np.isclose(x[0], 1.0),
            np.isclose(x[1], 0.0), np.isclose(x[1], 1.0)
        ])
    
    tdim = domain.topology.dim
    fdim = tdim - 1
    boundary_facets = mesh.locate_entities_boundary(domain, fdim, boundary_marker)
    dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    
    # Time-stepping
    t_end = 0.1
    n_steps = int(np.ceil(t_end / dt))
    dt = t_end / n_steps
    
    # Functions
    u_n = fem.Function(V)
    u_n.interpolate(lambda x: u_exact(x, 0.0))
    u_sol = fem.Function(V)
    
    # Variational problem
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    
    a = (u * v + dt * 1.0 * ufl.inner(ufl.grad(u), ufl.grad(v))) * ufl.dx
    L = u_n * v * ufl.dx
    
    # Source term
    f_fe = fem.Function(V)
    
    # Assemble
    a_form = fem.form(a)
    L_form = fem.form(L)
    
    # Direct solver
    A = petsc.assemble_matrix(a_form, bcs=[])
    A.assemble()
    b = petsc.create_vector(L_form.function_spaces)
    
    solver = PETSc.KSP().create(domain.comm)
    solver.setOperators(A)
    solver.setType("preonly")
    solver.getPC().setType("lu")
    solver.setTolerances(rtol=1e-12)
    solver.setFromOptions()
    
    # Time-stepping
    t = 0.0
    
    for step in range(n_steps):
        t_new = t + dt
        
        # Update BC and source
        u_bc.interpolate(lambda x: u_exact(x, t_new))
        bc = fem.dirichletbc(u_bc, dofs)
        f_fe.interpolate(lambda x: f_source(x, t_new))
        
        # Assemble RHS
        with b.localForm() as loc:
            loc.set(0)
        petsc.assemble_vector(b, L_form)
        
        # Add source
        source_form = fem.form(dt * ufl.inner(f_fe, v) * ufl.dx)
        b_source = petsc.create_vector(source_form.function_spaces)
        petsc.assemble_vector(b_source, source_form)
        b.axpy(1.0, b_source)
        
        # Apply BC
        petsc.apply_lifting(b, [a_form], bcs=[[bc]])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        petsc.set_bc(b, [bc])
        
        # Solve
        solver.solve(b, u_sol.x.petsc_vec)
        u_sol.x.scatter_forward()
        
        # Update
        u_n.x.array[:] = u_sol.x.array
        t = t_new
    
    # Compute error
    u_exact_fe = fem.Function(V)
    u_exact_fe.interpolate(lambda x: u_exact(x, t_end))
    
    error_func = fem.Function(V)
    error_func.x.array[:] = u_sol.x.array - u_exact_fe.x.array
    
    error_form = fem.form(ufl.inner(error_func, error_func) * ufl.dx)
    error_l2 = np.sqrt(fem.assemble_scalar(error_form))
    
    return error_l2

if rank == 0:
    for dt in [0.01, 0.001, 0.0001]:
        error = test_dt(N=64, degree=1, dt=dt)
        print(f"dt={dt}: FE L2 error = {error:.6e}")
