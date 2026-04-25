import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc
import time

def solve(case_spec: dict) -> dict:
    start_time = time.time()
    
    # Grid parameters from case_spec
    grid_spec = case_spec["output"]["grid"]
    nx_out = grid_spec["nx"]
    ny_out = grid_spec["ny"]
    bbox = grid_spec["bbox"]
    
    # Discretization parameters
    mesh_res = 100
    degree = 2
    
    # Create mesh
    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_square(comm, mesh_res, mesh_res, cell_type=mesh.CellType.triangle)
    
    # Function space
    V = fem.functionspace(domain, ("Lagrange", degree))
    
    # Exact solution for boundary conditions and source term
    x = ufl.SpatialCoordinate(domain)
    u_exact = ufl.sin(2 * ufl.pi * (x[0] + x[1])) * ufl.sin(ufl.pi * (x[0] - x[1]))
    
    # Boundary conditions
    fdim = domain.topology.dim - 1
    domain.topology.create_connectivity(fdim, domain.topology.dim)
    boundary_facets = mesh.locate_entities_boundary(domain, fdim, lambda x: np.full(x.shape[1], True, dtype=bool))
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    
    u_bc_func = fem.Function(V)
    u_bc_expr = fem.Expression(u_exact, V.element.interpolation_points)
    u_bc_func.interpolate(u_bc_expr)
    bc = fem.dirichletbc(u_bc_func, boundary_dofs)
    
    # Problem parameters
    epsilon = fem.Constant(domain, PETSc.ScalarType(0.05))
    beta_val = np.array([3.0, 1.0])
    beta = fem.Constant(domain, PETSc.ScalarType(beta_val))
    
    # Compute source term f automatically from exact solution
    f_exact = -epsilon * ufl.div(ufl.grad(u_exact)) + ufl.dot(beta, ufl.grad(u_exact))
    
    # Trial and Test functions
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    
    # Weak form without stabilization
    F = epsilon * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx \
        + ufl.inner(ufl.dot(beta, ufl.grad(u)), v) * ufl.dx \
        - ufl.inner(f_exact, v) * ufl.dx
        
    # SUPG Stabilization
    h = ufl.CellDiameter(domain)
    beta_norm = np.linalg.norm(beta_val)
    # Simple tau formulation for SUPG
    tau = h / (2.0 * beta_norm)
    
    # Residual
    residual = -epsilon * ufl.div(ufl.grad(u)) + ufl.dot(beta, ufl.grad(u)) - f_exact
    # Add stabilization term
    F_supg = F + ufl.inner(residual, tau * ufl.dot(beta, ufl.grad(v))) * ufl.dx
    
    a = ufl.lhs(F_supg)
    L = ufl.rhs(F_supg)
    
    # Solve linear problem
    problem = petsc.LinearProblem(
        a, L, bcs=[bc],
        petsc_options={"ksp_type": "preonly", "pc_type": "lu", "pc_factor_mat_solver_type": "mumps"},
        petsc_options_prefix="cd_"
    )
    u_sol = problem.solve()
    
    # Point evaluation on required grid
    xs = np.linspace(bbox[0], bbox[1], nx_out)
    ys = np.linspace(bbox[2], bbox[3], ny_out)
    XX, YY = np.meshgrid(xs, ys)
    points_array = np.vstack((XX.ravel(), YY.ravel(), np.zeros_like(XX.ravel())))
    
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
            
    u_values = np.zeros(points_array.shape[1])
    if len(points_on_proc) > 0:
        vals = u_sol.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u_values[eval_map] = vals.flatten()
        
    u_grid = u_values.reshape((ny_out, nx_out))
    
    # Return structure
    solver_info = {
        "mesh_resolution": mesh_res,
        "element_degree": degree,
        "ksp_type": "preonly",
        "pc_type": "lu",
        "rtol": 1e-8,
        "iterations": 1
    }
    
    return {
        "u": u_grid,
        "solver_info": solver_info
    }

