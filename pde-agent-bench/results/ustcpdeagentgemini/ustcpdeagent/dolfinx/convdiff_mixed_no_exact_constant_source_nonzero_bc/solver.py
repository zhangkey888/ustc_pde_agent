import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc
import time

def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD
    
    # Extract grid spec
    grid_spec = case_spec["output"]["grid"]
    nx = grid_spec["nx"]
    ny = grid_spec["ny"]
    bbox = grid_spec["bbox"]
    
    # Parameters
    epsilon = 0.005
    beta = [12.0, 0.0]
    f_val = 1.0
    
    # Mesh and function space
    mesh_res = 128
    domain = mesh.create_unit_square(comm, nx=mesh_res, ny=mesh_res, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", 1))
    
    # Boundary Conditions
    tdim = domain.topology.dim
    fdim = tdim - 1
    domain.topology.create_connectivity(fdim, tdim)
    boundary_facets = mesh.exterior_facet_indices(domain.topology)
    dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    
    u_bc = fem.Function(V)
    u_bc.interpolate(lambda x: np.sin(np.pi * x[0]))
    bc = fem.dirichletbc(u_bc, dofs)
    
    # Variational problem
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    f = fem.Constant(domain, PETSc.ScalarType(f_val))
    b = fem.Constant(domain, PETSc.ScalarType(beta))
    eps = fem.Constant(domain, PETSc.ScalarType(epsilon))
    
    # Mesh size for stabilization
    h = ufl.CellDiameter(domain)
    beta_mag = np.linalg.norm(beta)
    # SUPG parameter for advection dominated (Peclet > 1)
    tau = h / (2.0 * beta_mag)
    
    # Standard Galerkin
    a = eps * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx + ufl.inner(ufl.dot(b, ufl.grad(u)), v) * ufl.dx
    L = ufl.inner(f, v) * ufl.dx
    
    # SUPG stabilization
    v_supg = tau * ufl.dot(b, ufl.grad(v))
    a_supg = a + ufl.inner(ufl.dot(b, ufl.grad(u)), v_supg) * ufl.dx
    # Diffusion term in residual vanishes for P1 elements
    L_supg = L + ufl.inner(f, v_supg) * ufl.dx
    
    # Solver
    problem = petsc.LinearProblem(
        a_supg, L_supg, bcs=[bc],
        petsc_options={"ksp_type": "preonly", "pc_type": "lu", "pc_factor_mat_solver_type": "mumps"},
        petsc_options_prefix="cd_"
    )
    u_sol = problem.solve()
    
    # Interpolation to output grid
    xs = np.linspace(bbox[0], bbox[1], nx)
    ys = np.linspace(bbox[2], bbox[3], ny)
    XX, YY = np.meshgrid(xs, ys)
    points = np.c_[XX.ravel(), YY.ravel(), np.zeros(nx*ny)].T
    
    bb_tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, points)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points)
    
    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    for i in range(points.shape[1]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(points[:, i])
            cells_on_proc.append(links[0])
            eval_map.append(i)
            
    u_grid_flat = np.full(nx * ny, np.nan)
    if len(points_on_proc) > 0:
        vals = u_sol.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u_grid_flat[eval_map] = vals.flatten()
        
    u_grid = u_grid_flat.reshape((ny, nx))
    
    solver_info = {
        "mesh_resolution": mesh_res,
        "element_degree": 1,
        "ksp_type": "preonly",
        "pc_type": "lu",
        "rtol": 1e-8,
        "iterations": 1
    }
    
    return {"u": u_grid, "solver_info": solver_info}
