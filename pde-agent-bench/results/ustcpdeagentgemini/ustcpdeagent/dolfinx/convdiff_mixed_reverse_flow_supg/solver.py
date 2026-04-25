import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
import ufl
from petsc4py import PETSc
from dolfinx.fem import petsc

def solve(case_spec: dict) -> dict:
    # Extract output grid specs
    nx_out = case_spec["output"]["grid"]["nx"]
    ny_out = case_spec["output"]["grid"]["ny"]
    bbox = case_spec["output"]["grid"]["bbox"]
    
    # Parameters
    epsilon = 0.005
    beta_vec = np.array([-20.0, 5.0])
    
    # Mesh and function space
    mesh_res = 128
    degree = 2
    domain = mesh.create_unit_square(MPI.COMM_WORLD, mesh_res, mesh_res, mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", degree))
    
    x = ufl.SpatialCoordinate(domain)
    
    # Exact solution
    u_exact = ufl.exp(x[0]) * ufl.sin(ufl.pi * x[1])
    
    # Source term f
    # f = -eps * Laplacian(u) + beta \cdot Grad(u)
    f = -epsilon * (1.0 - ufl.pi**2) * u_exact + beta_vec[0] * ufl.exp(x[0]) * ufl.sin(ufl.pi * x[1]) + beta_vec[1] * ufl.pi * ufl.exp(x[0]) * ufl.cos(ufl.pi * x[1])
    
    # Boundary conditions
    domain.topology.create_connectivity(domain.topology.dim - 1, domain.topology.dim)
    boundary_facets = mesh.exterior_facet_indices(domain.topology)
    boundary_dofs = fem.locate_dofs_topological(V, domain.topology.dim - 1, boundary_facets)
    
    u_bc = fem.Function(V)
    u_bc.interpolate(lambda x: np.exp(x[0]) * np.sin(np.pi * x[1]))
    bc = fem.dirichletbc(u_bc, boundary_dofs)
    
    # Trial and Test functions
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    
    beta = fem.Constant(domain, PETSc.ScalarType(beta_vec))
    eps = fem.Constant(domain, PETSc.ScalarType(epsilon))
    
    # Standard Galerkin
    F = eps * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx \
        + ufl.inner(ufl.dot(beta, ufl.grad(u)), v) * ufl.dx \
        - ufl.inner(f, v) * ufl.dx
        
    # SUPG stabilization
    h = ufl.CellDiameter(domain)
    beta_mag = ufl.sqrt(ufl.dot(beta, beta))
    tau = h / (2.0 * beta_mag)
    
    residual = -eps * ufl.div(ufl.grad(u)) + ufl.dot(beta, ufl.grad(u)) - f
    v_supg = tau * ufl.dot(beta, ufl.grad(v))
    F += ufl.inner(residual, v_supg) * ufl.dx
    
    a, L = ufl.lhs(F), ufl.rhs(F)
    
    # Solve
    problem = petsc.LinearProblem(
        a, L, bcs=[bc],
        petsc_options={"ksp_type": "gmres", "pc_type": "ilu", "ksp_rtol": 1e-9},
        petsc_options_prefix="convdiff_"
    )
    u_sol = problem.solve()
    
    # Get iterations
    ksp = problem.solver
    iterations = ksp.getIterationNumber()
    
    # Interpolate to output grid
    xs = np.linspace(bbox[0], bbox[1], nx_out)
    ys = np.linspace(bbox[2], bbox[3], ny_out)
    X, Y = np.meshgrid(xs, ys)
    points = np.vstack([X.ravel(), Y.ravel(), np.zeros_like(X.ravel())])
    
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
    
    # Replace nans with 0.0 or handle if out of bounds (should not happen for exact bbox)
    u_grid = np.nan_to_num(u_grid)
    
    solver_info = {
        "mesh_resolution": mesh_res,
        "element_degree": degree,
        "ksp_type": "gmres",
        "pc_type": "ilu",
        "rtol": 1e-9,
        "iterations": iterations,
    }
    
    return {"u": u_grid, "solver_info": solver_info}

