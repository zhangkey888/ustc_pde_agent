```python
import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc

def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD
    
    # Parameters
    nx, ny = 128, 128
    degree = 1
    t_end = 0.1
    dt = 0.02
    n_steps = int(round(t_end / dt))
    kappa = 1.0
    
    # Mesh and Function Space
    domain = mesh.create_unit_square(comm, nx, ny, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", degree))
    
    # Initial condition
    u_n = fem.Function(V)
    u_n.name = "u_n"
    u_n.x.array[:] = 0.0
    
    # Boundary condition
    tdim = domain.topology.dim
    fdim = tdim - 1
    domain.topology.create_connectivity(fdim, tdim)
    boundary_facets = mesh.exterior_facet_indices(domain.topology)
    dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc = fem.dirichletbc(PETSc.ScalarType(0.0), dofs, V)
    
    # Trial and Test functions
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    
    # Source term
    x = ufl.SpatialCoordinate(domain)
    f = ufl.exp(-220.0 * ((x[0] - 0.25)**2 + (x[1] - 0.25)**2)) + \
        ufl.exp(-220.0 * ((x[0] - 0.75)**2 + (x[1] - 0.7)**2))
    
    # Variational problem (Backward Euler)
    dt_c = fem.Constant(domain, PETSc.ScalarType(dt))
    kappa_c = fem.Constant(domain, PETSc.ScalarType(kappa))
    
    a = u * v * ufl.dx + dt_c * kappa_c * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = (u_n * v + dt_c * f * v) * ufl.dx
    
    a_form = fem.form(a)
    L_form = fem.form(L)
    
    # Assemble matrix
    A = petsc.assemble_matrix(a_form, bcs=[bc])
    A.assemble()
    
    b = petsc.create_vector(L_form.function_spaces)
    
    # Solver setup
    solver = PETSc.KSP().create(comm)
    solver.setOperators(A)
    ksp_type = "cg"
    pc_type = "hypre"
    rtol = 1e-8
    solver.setType(ksp_type)
    solver.getPC().setType(pc_type)
    solver.setTolerances(rtol=rtol)
    
    u_sol = fem.Function(V)
    u_sol.x.array[:] = 0.0
    
    total_iterations = 0
    
    # Time stepping
    t = 0.0
    for i in range(n_steps):
        t += dt
        
        with b.localForm() as loc:
            loc.set(0)
        petsc.assemble_vector(b, L_form)
        petsc.apply_lifting(b, [a_form], bcs=[[bc]])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        petsc.set_bc(b, [bc])
        
        solver.solve(b, u_sol.x.petsc_vec)
        u_sol.x.scatter_forward()
        
        total_iterations += solver.getIterationNumber()
        
        # Update previous solution
        u_n.x.array[:] = u_sol.x.array[:]
        
    # Evaluate on 50x50 grid
    eval_nx, eval_ny = 50, 50
    x_eval = np.linspace(0, 1, eval_nx)
    y_eval = np.linspace(0, 1, eval_ny)
    X, Y = np.meshgrid(x_eval, y_eval, indexing='ij')
    points = np.vstack((X.flatten(), Y.flatten(), np.zeros_like(X.flatten())))
    
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
            
    u_values = np.full((points.shape[1],), np.nan)
    u_init_values = np.full((points.shape[1],), np.nan)
    
    if len(points_on_proc) > 0:
        vals = u_sol.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u_values