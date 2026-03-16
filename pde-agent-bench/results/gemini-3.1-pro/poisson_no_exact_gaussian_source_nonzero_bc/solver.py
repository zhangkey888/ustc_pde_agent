import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc

def probe_points(u_func, points_array, domain):
    bb_tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, points_array.T)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points_array.T)

    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    for i in range(points_array.shape[1]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(points_array.T[i])
            cells_on_proc.append(links[0])
            eval_map.append(i)

    u_values = np.full((points_array.shape[1],), np.nan)
    if len(points_on_proc) > 0:
        vals = u_func.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u_values[eval_map] = vals.flatten()
        
    comm = domain.comm
    if comm.size > 1:
        u_values_local = np.where(np.isnan(u_values), 0.0, u_values)
        mask_local = np.where(np.isnan(u_values), 0.0, 1.0)
        
        u_values_global = np.zeros_like(u_values_local)
        mask_global = np.zeros_like(mask_local)
        
        comm.Allreduce(u_values_local, u_values_global, op=MPI.SUM)
        comm.Allreduce(mask_local, mask_global, op=MPI.SUM)
        
        with np.errstate(divide='ignore', invalid='ignore'):
            u_values = np.where(mask_global > 0, u_values_global / mask_global, np.nan)
            
    return u_values

def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD
    
    # Parameters
    mesh_resolution = 64
    element_degree = 2
    ksp_type = "cg"
    pc_type = "jacobi"
    rtol = 1e-8
    
    # 1. Mesh
    domain = mesh.create_unit_square(comm, mesh_resolution, mesh_resolution, cell_type=mesh.CellType.triangle)
    
    # 2. Function Space
    V = fem.functionspace(domain, ("Lagrange", element_degree))
    
    # 3. Boundary Conditions
    fdim = domain.topology.dim - 1
    def boundary_marker(x):
        return np.full(x.shape[1], True)
    
    boundary_facets = mesh.locate_entities_boundary(domain, fdim, boundary_marker)
    dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    
    u_bc = fem.Function(V)
    u_bc.interpolate(lambda x: np.full_like(x[0], 1.0))
    bc = fem.dirichletbc(u_bc, dofs)
    
    # 4. Variational Problem
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    
    x = ufl.SpatialCoordinate(domain)
    f = ufl.exp(-180.0 * ((x[0] - 0.3)**2 + (x[1] - 0.7)**2))
    
    a = ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = ufl.inner(f, v) * ufl.dx
    
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
    solver = PETSc.KSP().create(comm)
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
    
    u_values = probe_points(u_sol, points, domain)
    u_grid = u_values.reshape((nx, ny))
    
    # 8. Return
    solver_info = {
        "mesh_resolution": mesh_resolution,
        "element_degree": element_degree,
        "ksp_type": ksp_type,
        "pc_type": pc_type,
        "rtol": rtol,
        "iterations": iterations
    }
    
    return {
        "u": u_grid,
        "solver_info": solver_info
    }