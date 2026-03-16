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
    nx, ny = 64, 64
    degree = 2
    ksp_type = "cg"
    pc_type = "hypre"
    rtol = 1e-8
    
    # 1. Mesh Generation (Quadrilaterals)
    p0 = np.array([0.0, 0.0])
    p1 = np.array([1.0, 1.0])
    domain = mesh.create_rectangle(
        comm, [p0, p1], [nx, ny], 
        cell_type=mesh.CellType.quadrilateral
    )
    
    # 2. Function Space
    V = fem.functionspace(domain, ("Lagrange", degree))
    
    # 3. Boundary Conditions (u = 0 on ∂Ω)
    tdim = domain.topology.dim
    fdim = tdim - 1
    domain.topology.create_connectivity(fdim, tdim)
    boundary_facets = mesh.exterior_facet_indices(domain.topology)
    dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc = fem.dirichletbc(PETSc.ScalarType(0.0), dofs, V)
    
    # 4. Variational Problem
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    
    f = fem.Constant(domain, PETSc.ScalarType(1.0))
    kappa = fem.Constant(domain, PETSc.ScalarType(1.0))
    
    a = kappa * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = f * v * ufl.dx
    
    # 5. Solver Setup
    problem = petsc.LinearProblem(
        a, L, bcs=[bc],
        petsc_options={
            "ksp_type": ksp_type,
            "pc_type": pc_type,
            "ksp_rtol": rtol,
        },
        petsc_options_prefix="poisson_"
    )
    u_sol = problem.solve()
    iterations = problem.solver.getIterationNumber()
    
    # 6. Evaluate on 50x50 uniform grid
    nx_eval, ny_eval = 50, 50
    x_coords = np.linspace(0, 1, nx_eval)
    y_coords = np.linspace(0, 1, ny_eval)
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
            
    u_values_local = np.full((points.shape[1],), -1e10)
    if len(points_on_proc) > 0:
        vals = u_sol.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u_values_local[eval_map] = vals.flatten()
        
    u_values_global = np.zeros_like(u_values_local)
    comm.Allreduce(u_values_local, u_values_global, op=MPI.MAX)
    
    # Replace any unassigned points (if any) with 0.0
    u_values_global[u_values_global < -1e9] = 0.0
    u_grid = u_values_global.reshape((nx_eval, ny_eval))
    
    solver_info = {
        "mesh_resolution": nx,
        "element_degree": degree,
        "ksp_type": ksp_type,
        "pc_type": pc_type,
        "rtol": rtol,