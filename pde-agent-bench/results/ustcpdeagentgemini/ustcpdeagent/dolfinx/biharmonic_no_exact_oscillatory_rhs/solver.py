import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc

def solve(case_spec: dict) -> dict:
    nx_out = case_spec["output"]["grid"]["nx"]
    ny_out = case_spec["output"]["grid"]["ny"]
    bbox = case_spec["output"]["grid"]["bbox"]
    
    # Mesh and function space
    N = 128
    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
    
    V = fem.functionspace(domain, ("Lagrange", 2))
    
    # Boundary conditions
    domain.topology.create_connectivity(domain.topology.dim-1, domain.topology.dim)
    boundary_facets = mesh.exterior_facet_indices(domain.topology)
    dofs = fem.locate_dofs_topological(V, domain.topology.dim-1, boundary_facets)
    
    bc_u = fem.dirichletbc(PETSc.ScalarType(0.0), dofs, V)
    bc_v = fem.dirichletbc(PETSc.ScalarType(0.0), dofs, V)
    
    # Trial and Test functions
    u = ufl.TrialFunction(V)
    test = ufl.TestFunction(V)
    
    # Source term
    x = ufl.SpatialCoordinate(domain)
    f = ufl.sin(10 * ufl.pi * x[0]) * ufl.sin(8 * ufl.pi * x[1])
    
    # Solve first Poisson: -Delta v = f
    a_v = ufl.inner(ufl.grad(u), ufl.grad(test)) * ufl.dx
    L_v = ufl.inner(f, test) * ufl.dx
    
    problem_v = petsc.LinearProblem(
        a_v, L_v, bcs=[bc_v],
        petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
        petsc_options_prefix="v_"
    )
    v_sol = problem_v.solve()
    
    # Solve second Poisson: -Delta u = v
    a_u = ufl.inner(ufl.grad(u), ufl.grad(test)) * ufl.dx
    L_u = ufl.inner(v_sol, test) * ufl.dx
    
    problem_u = petsc.LinearProblem(
        a_u, L_u, bcs=[bc_u],
        petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
        petsc_options_prefix="u_"
    )
    u_sol = problem_u.solve()
    
    # Interpolate to grid
    xs = np.linspace(bbox[0], bbox[1], nx_out)
    ys = np.linspace(bbox[2], bbox[3], ny_out)
    XX, YY = np.meshgrid(xs, ys)
    pts = np.c_[XX.ravel(), YY.ravel(), np.zeros(nx_out * ny_out)]
    
    bb_tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, pts)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, pts)
    
    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    for i in range(pts.shape[0]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(pts[i])
            cells_on_proc.append(links[0])
            eval_map.append(i)
            
    u_values = np.zeros(pts.shape[0])
    if len(points_on_proc) > 0:
        vals = u_sol.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u_values[eval_map] = vals.flatten()
        
    u_grid = u_values.reshape((ny_out, nx_out))
    
    return {
        "u": u_grid,
        "solver_info": {
            "mesh_resolution": N,
            "element_degree": 2,
            "ksp_type": "preonly",
            "pc_type": "lu",
            "rtol": 1e-8,
            "iterations": 2
        }
    }
