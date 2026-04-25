import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc
from basix.ufl import element as basix_element, mixed_element as basix_mixed_element

def solve(case_spec: dict) -> dict:
    nx = case_spec["output"]["grid"]["nx"]
    ny = case_spec["output"]["grid"]["ny"]
    bbox = case_spec["output"]["grid"]["bbox"]

    comm = MPI.COMM_WORLD
    
    # Mesh resolution
    mesh_res = 128
    degree = 2
    
    domain = mesh.create_unit_square(comm, mesh_res, mesh_res, cell_type=mesh.CellType.triangle)
    
    # Mixed element formulation
    el = basix_element("Lagrange", domain.topology.cell_name(), degree)
    W = fem.functionspace(domain, basix_mixed_element([el, el]))
    
    (u, v) = ufl.TrialFunctions(W)
    (phi, psi) = ufl.TestFunctions(W)
    
    x = ufl.SpatialCoordinate(domain)
    # Source term
    f = 10.0 * ufl.exp(-80.0 * ((x[0] - 0.35)**2 + (x[1] - 0.55)**2))
    
    # Weak form
    # -\Delta u = v => \int \nabla u \cdot \nabla phi dx - \int v phi dx = 0
    # -\Delta v = f => \int \nabla v \cdot \nabla psi dx = \int f psi dx
    
    a = ufl.inner(ufl.grad(u), ufl.grad(phi)) * ufl.dx - ufl.inner(v, phi) * ufl.dx \
      + ufl.inner(ufl.grad(v), ufl.grad(psi)) * ufl.dx
      
    L = ufl.inner(f, psi) * ufl.dx
    
    # Boundary conditions
    V, _ = W.sub(0).collapse()
    Q, _ = W.sub(1).collapse()
    
    fdim = domain.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool))
    
    # u = 0
    dofs_u = fem.locate_dofs_topological((W.sub(0), V), fdim, boundary_facets)
    u_bc = fem.Function(V)
    u_bc.x.array[:] = 0.0
    bc_u = fem.dirichletbc(u_bc, dofs_u, W.sub(0))
    
    # v = 0 (simply supported)
    dofs_v = fem.locate_dofs_topological((W.sub(1), Q), fdim, boundary_facets)
    v_bc = fem.Function(Q)
    v_bc.x.array[:] = 0.0
    bc_v = fem.dirichletbc(v_bc, dofs_v, W.sub(1))
    
    bcs = [bc_u, bc_v]
    
    problem = petsc.LinearProblem(a, L, bcs=bcs, petsc_options={"ksp_type": "preonly", "pc_type": "lu", "pc_factor_mat_solver_type": "mumps"})
    w_sol = problem.solve()
    
    u_sol = w_sol.sub(0).collapse()
    
    # Interpolation onto grid
    xs = np.linspace(bbox[0], bbox[1], nx)
    ys = np.linspace(bbox[2], bbox[3], ny)
    XX, YY = np.meshgrid(xs, ys)
    pts = np.c_[XX.ravel(), YY.ravel(), np.zeros(nx * ny)]
    
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
            
    u_values = np.full((pts.shape[0],), np.nan)
    if len(points_on_proc) > 0:
        vals = u_sol.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u_values[eval_map] = vals.flatten()
        
    # MPI reduce for u_values if running in parallel
    u_values_global = np.zeros_like(u_values)
    comm.Allreduce(np.nan_to_num(u_values), u_values_global, op=MPI.SUM)
    # For nodes that are not mapped, they will be 0. We might need a better reduction if we care about parallel execution.
    # Assuming serial execution for now.
    
    u_grid = u_values.reshape((ny, nx))
    
    return {
        "u": u_grid,
        "solver_info": {
            "mesh_resolution": mesh_res,
            "element_degree": degree,
            "ksp_type": "preonly",
            "pc_type": "lu",
            "rtol": 1e-8,
            "iterations": 1
        }
    }
