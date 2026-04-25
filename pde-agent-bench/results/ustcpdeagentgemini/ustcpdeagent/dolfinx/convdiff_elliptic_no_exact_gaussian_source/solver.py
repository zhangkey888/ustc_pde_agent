import time
import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
import ufl
from petsc4py import PETSc
from dolfinx.fem import petsc

def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD
    
    # Grid spec
    grid_spec = case_spec["output"]["grid"]
    nx_out = grid_spec["nx"]
    ny_out = grid_spec["ny"]
    bbox = grid_spec["bbox"]
    
    # Discretization parameters
    mesh_resolution = 128
    element_degree = 2
    
    # Create mesh
    domain = mesh.create_unit_square(comm, mesh_resolution, mesh_resolution, mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", element_degree))
    
    # Define boundary condition: u = 0 on all boundaries
    fdim = domain.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool))
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    u_bc = fem.Function(V)
    u_bc.x.array[:] = 0.0
    bc = fem.dirichletbc(u_bc, boundary_dofs)
    
    # Parameters
    epsilon = 0.05
    beta = ufl.as_vector([2.0, 1.0])
    
    # Trial and test functions
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    
    # Source term
    x = ufl.SpatialCoordinate(domain)
    f = ufl.exp(-250 * ((x[0] - 0.35)**2 + (x[1] - 0.65)**2))
    
    # Mesh size for SUPG
    h = ufl.CellDiameter(domain)
    v_norm = ufl.sqrt(ufl.dot(beta, beta))
    
    # SUPG stabilization parameter (tau)
    # Simple choice for steady convection-diffusion
    Pe_h = v_norm * h / (2.0 * epsilon)
    # tau = h / (2 * v_norm) * xi, where xi = coth(Pe_h) - 1/Pe_h
    # For Pe_h > 1, xi ~ 1. Here we use a safe robust form.
    tau = h / (2.0 * v_norm)
    
    # Variational form
    F_galerkin = epsilon * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx \
               + ufl.inner(ufl.dot(beta, ufl.grad(u)), v) * ufl.dx \
               - ufl.inner(f, v) * ufl.dx
               
    # Residual for SUPG
    residual = -epsilon * ufl.div(ufl.grad(u)) + ufl.dot(beta, ufl.grad(u)) - f
    
    # SUPG test function variation
    v_supg = tau * ufl.dot(beta, ufl.grad(v))
    
    F_supg = ufl.inner(residual, v_supg) * ufl.dx
    
    # Total form
    F = F_galerkin + F_supg
    a = ufl.lhs(F)
    L = ufl.rhs(F)
    
    # Solve linear problem
    problem = petsc.LinearProblem(
        a, L, bcs=[bc],
        petsc_options={"ksp_type": "preonly", "pc_type": "lu", "pc_factor_mat_solver_type": "mumps"},
        petsc_options_prefix="convdiff_"
    )
    u_sol = problem.solve()
    
    # Interpolate onto output grid
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
    for i in range(len(pts)):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(pts[i])
            cells_on_proc.append(links[0])
            eval_map.append(i)
            
    u_grid_flat = np.zeros(nx_out * ny_out)
    if len(points_on_proc) > 0:
        vals = u_sol.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u_grid_flat[eval_map] = vals.flatten()
        
    u_grid = u_grid_flat.reshape((ny_out, nx_out))
    
    solver_info = {
        "mesh_resolution": mesh_resolution,
        "element_degree": element_degree,
        "ksp_type": "preonly",
        "pc_type": "lu",
        "rtol": 1e-8,
        "iterations": 1
    }
    
    return {
        "u": u_grid,
        "solver_info": solver_info
    }
