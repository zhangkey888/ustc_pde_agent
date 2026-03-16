import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc

def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD
    
    # Parameters
    nx, ny = 64, 64
    degree = 2
    ksp_type = "cg"
    pc_type = "jacobi"
    rtol = 1e-8
    
    # Create mesh
    domain = mesh.create_unit_square(comm, nx, ny, cell_type=mesh.CellType.triangle)
    
    # Function space
    V = fem.functionspace(domain, ("Lagrange", degree))
    
    # Boundary conditions
    tdim = domain.topology.dim
    fdim = tdim - 1
    domain.topology.create_connectivity(fdim, tdim)
    boundary_facets = mesh.exterior_facet_indices(domain.topology)
    
    dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    
    u_bc = fem.Function(V)
    u_bc.interpolate(lambda x: np.sin(np.pi * x[0] * x[1]))
    bc = fem.dirichletbc(u_bc, dofs)
    
    # Variational problem
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    
    x = ufl.SpatialCoordinate(domain)
    # Source term derived from exact solution u = sin(pi*x*y)
    f = ufl.pi**2 * (x[0]**2 + x[1]**2) * ufl.sin(ufl.pi * x[0] * x[1])
    
    a = ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = ufl.inner(f, v) * ufl.dx
    
    # Assemble matrix and vector
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
    
    # Solve
    solver = PETSc.KSP().create(comm)
    solver.setOperators(A)
    solver.setType(ksp_type)
    solver.getPC().setType(pc_type)
    solver.setTolerances(rtol=rtol)
    
    u_sol = fem.Function(V)
    solver.solve(b, u_sol.x.petsc_vec)
    u_sol.x.scatter_forward()
    
    iterations = solver.getIterationNumber()
    
    # Evaluate on 50x50 grid
    x_eval = np.linspace(0, 1, 50)
    y_eval = np.linspace(0, 1, 50)
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
    if len(points_on_proc) > 0:
        vals = u_sol.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u_values[eval_map] = vals.flatten()
        
    # Handle parallel evaluation
    u_values_global = comm.reduce(np.nan_to_num(u_values, nan=0.0), op=MPI.SUM, root=0)
    if comm.size > 1:
        found = np.zeros_like(u_values)
        found[eval_map] = 1.0
        found_global = comm.reduce(found, op=MPI.SUM, root=0)
        if comm.rank == 0:
            u_grid = (u_values_global / np.maximum(found_global, 1)).reshape((50, 50))
        else:
            u_grid = None
    else:
        u_grid = u_values.reshape((50, 50))
        
    solver_info = {
        "mesh_resolution": nx,
        "element_degree": degree,
        "ksp_type": ksp_type,
        "pc_type": pc_type,
        "rtol": rtol,
        "iterations": iterations
    }
    
    return {
        "u": u_grid,
        "solver_info": solver_info
    }