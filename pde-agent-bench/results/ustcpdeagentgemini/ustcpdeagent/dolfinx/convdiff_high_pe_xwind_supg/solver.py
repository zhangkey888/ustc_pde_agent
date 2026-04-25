import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc
import time

def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD
    
    # Read grid specs
    nx_out = case_spec["output"]["grid"]["nx"]
    ny_out = case_spec["output"]["grid"]["ny"]
    bbox = case_spec["output"]["grid"]["bbox"]
    
    # Mesh and space
    N = 100
    domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", 2))
    
    # Parameters
    eps = 0.01
    beta_val = [20.0, 0.0]
    beta = fem.Constant(domain, PETSc.ScalarType(beta_val))
    
    # Exact solution for f and BCs
    x = ufl.SpatialCoordinate(domain)
    u_exact = ufl.sin(ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])
    
    f = eps * 2 * ufl.pi**2 * u_exact + beta_val[0] * ufl.pi * ufl.cos(ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])
    
    # Boundary conditions
    fdim = domain.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(domain, fdim, lambda x: np.full(x.shape[1], True))
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    u_bc = fem.Function(V)
    u_bc.x.array[:] = 0.0
    bc = fem.dirichletbc(u_bc, boundary_dofs)
    
    # Weak form
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    
    # Standard Galerkin
    a = eps * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx + ufl.inner(beta, ufl.grad(u)) * v * ufl.dx
    L = f * v * ufl.dx
    
    # SUPG Stabilization
    # cell size h
    h = ufl.CellDiameter(domain)
    norm_beta = np.linalg.norm(beta_val)
    tau = h / (2.0 * norm_beta)
    
    # Residual
    R = -eps * ufl.div(ufl.grad(u)) + ufl.inner(beta, ufl.grad(u)) - f
    v_supg = tau * ufl.inner(beta, ufl.grad(v))
    
    a_supg = a + (-eps * ufl.div(ufl.grad(u)) + ufl.inner(beta, ufl.grad(u))) * v_supg * ufl.dx
    L_supg = L + f * v_supg * ufl.dx
    
    # Solve
    problem = petsc.LinearProblem(
        a_supg, L_supg, bcs=[bc],
        petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
        petsc_options_prefix="convdiff_"
    )
    u_sol = problem.solve()
    
    # Interpolation onto output grid
    xs = np.linspace(bbox[0], bbox[1], nx_out)
    ys = np.linspace(bbox[2], bbox[3], ny_out)
    XX, YY = np.meshgrid(xs, ys)
    points = np.vstack((XX.flatten(), YY.flatten(), np.zeros_like(XX.flatten())))
    
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
        
    u_grid = u_values.reshape((ny_out, nx_out))
    
    solver_info = {
        "mesh_resolution": N,
        "element_degree": 2,
        "ksp_type": "preonly",
        "pc_type": "lu",
        "rtol": 1e-8,
        "iterations": 1
    }
    
    return {
        "u": u_grid,
        "solver_info": solver_info
    }
