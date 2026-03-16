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
    mesh_res = 128
    degree = 2
    ksp_type = "cg"
    pc_type = "hypre"
    rtol = 1e-8
    
    # 1. Mesh
    domain = mesh.create_unit_square(comm, mesh_res, mesh_res, cell_type=mesh.CellType.triangle)
    
    # 2. Function Space
    V = fem.functionspace(domain, ("Lagrange", degree))
    
    # 3. Boundary Conditions
    tdim = domain.topology.dim
    fdim = tdim - 1
    domain.topology.create_connectivity(fdim, tdim)
    boundary_facets = mesh.exterior_facet_indices(domain.topology)
    dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    
    u_bc = fem.Function(V)
    u_bc.x.array[:] = 0.0
    bc = fem.dirichletbc(u_bc, dofs)
    
    # 4. Variational Form
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    
    x = ufl.SpatialCoordinate(domain)
    # kappa = 1 + 1000*exp(-100*(x-0.5)**2)
    kappa = 1.0 + 1000.0 * ufl.exp(-100.0 * (x[0] - 0.5)**2)
    f = fem.Constant(domain, PETSc.ScalarType(1.0))
    
    a = kappa * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = f * v * ufl.dx
    
    # 5. Assembly
    a_form = fem.form(a)
    L_form = fem.form(L)
    
    A = petsc.assemble_matrix(a_form, bcs=[bc])
    A.assemble()
    
    b = petsc.create_vector(L_form.function_spaces)
    with b.localForm() as loc:
        loc.set(0)
    petsc.assemble_vector(b, L_form)
    petsc.apply_lifting(b, [a_form], bcs=[[bc]])
    b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    petsc.set_bc(b, [bc])
    
    # 6. Solver
    solver = PETSc.KSP().create(domain.comm)
    solver.setOperators(A)
    solver.setType(ksp_type)
    solver.getPC().setType(pc_type)
    solver.setTolerances(rtol=rtol)
    
    u_sol = fem.Function(V)
    solver.solve(b, u_sol.x.petsc_vec)
    u_sol.x.scatter_forward()
    
    iterations = solver.getIterationNumber()
    
    # 7. Evaluation on 50x50 grid
    nx, ny = 50, 50
    x_coords = np.linspace(0, 1, nx)
    y_coords = np.linspace(0, 1, ny)
    X, Y = np.meshgrid(x_coords, y_coords, indexing='ij')
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
            
    u_values = np.zeros((points.shape[1