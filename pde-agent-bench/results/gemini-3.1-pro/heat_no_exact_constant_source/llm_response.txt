```python
import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc

def solve(case_spec: dict) -> dict:
    # Problem parameters
    t_end = 0.1
    dt = 0.02
    kappa = 1.0
    f_val = 1.0
    u0_val = 0.0
    g_val = 0.0
    
    # Discretization parameters
    nx, ny = 64, 64
    degree = 1
    
    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_square(comm, nx, ny, cell_type=mesh.CellType.triangle)
    
    V = fem.functionspace(domain, ("Lagrange", degree))
    
    # Initial condition
    u_n = fem.Function(V)
    u_n.interpolate(lambda x: np.full_like(x[0], u0_val))
    
    # Boundary condition
    tdim = domain.topology.dim
    fdim = tdim - 1
    domain.topology.create_connectivity(fdim, tdim)
    boundary_facets = mesh.exterior_facet_indices(domain.topology)
    dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc = fem.dirichletbc(PETSc.ScalarType(g_val), dofs, V)
    
    # Variational problem (Backward Euler)
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    f = fem.Constant(domain, PETSc.ScalarType(f_val))
    dt_const = fem.Constant(domain, PETSc.ScalarType(dt))
    kappa_const = fem.Constant(domain, PETSc.ScalarType(kappa))
    
    a = u * v * ufl.dx + dt_const * kappa_const * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = u_n * v * ufl.dx + dt_const * f * v * ufl.dx
    
    a_form = fem.form(a)
    L_form = fem.form(L)
    
    # Assemble matrix (constant in time)
    A = petsc.assemble_matrix(a_form, bcs=[bc])
    A.assemble()
    
    b = petsc.create_vector(L_form.function_spaces)
    
    # Linear solver setup
    solver = PETSc.KSP().create(comm)
    solver.setOperators(A)
    solver.setType(PETSc.KSP.Type.CG)
    solver.getPC().setType(PETSc.PC.Type.JACOBI)
    solver.setTolerances(rtol=1e-8)
    
    u_sol = fem.Function(V)
    u_sol.x.array[:] = u_n.x.array[:]
    
    # Time-stepping loop
    t = 0.0
    n_steps = int(round(t_end / dt))
    total_iterations = 0
    
    for i in range(n_steps):
        t += dt
        
        # Assemble RHS
        with b.localForm() as loc:
            loc.set(0)
        petsc.assemble_vector(b, L_form)
        petsc.apply_lifting(b, [a_form], bcs=[[bc]])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        petsc.set_bc(b, [bc])
        
        # Solve
        solver.solve(b, u_sol.x.petsc_vec)
        u_sol.x.scatter_forward()
        
        total_iterations += solver.getIterationNumber()
        
        # Update previous solution
        u_n.x.array[:] = u_sol.x.array[:]
        
    # Evaluate on 50x50 grid
    x_eval = np.linspace(0, 1, 50)
    y_eval = np.linspace(0, 1, 50)
    X, Y = np.meshgrid(x_eval, y_eval, indexing='ij')
    points = np.vstack((X.flatten(), Y.flatten(), np.zeros_like(X.flatten())))
    
    bb_tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, points.T)
    colliding_cells = geometry.