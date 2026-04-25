import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import mesh, fem, geometry
import ufl
from dolfinx.fem import petsc

def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD
    
    # Mesh and Space
    nx_mesh = 120
    ny_mesh = 120
    domain = mesh.create_unit_square(comm, nx_mesh, ny_mesh, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", 2))
    
    # Exact solution and source term
    x = ufl.SpatialCoordinate(domain)
    u_ex = ufl.sin(ufl.pi * x[0]) * ufl.sin(2 * ufl.pi * x[1])
    beta_val = [5.0, 2.0]
    beta = ufl.as_vector(beta_val)
    eps = 0.03
    
    f = -eps * ufl.div(ufl.grad(u_ex)) + ufl.dot(beta, ufl.grad(u_ex))
    
    # Boundary Conditions
    tdim = domain.topology.dim
    fdim = tdim - 1
    domain.topology.create_connectivity(fdim, tdim)
    boundary_facets = mesh.exterior_facet_indices(domain.topology)
    dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    
    u_bc = fem.Function(V)
    u_bc.interpolate(lambda x: np.sin(np.pi * x[0]) * np.sin(2 * np.pi * x[1]))
    bc = fem.dirichletbc(u_bc, dofs)
    
    # Variational Form with SUPG
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    
    # Galerkin part
    a_G = eps * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx + ufl.dot(beta, ufl.grad(u)) * v * ufl.dx
    L_G = f * v * ufl.dx
    
    # SUPG stabilization
    h = ufl.CellDiameter(domain)
    norm_beta = ufl.sqrt(ufl.dot(beta, beta))
    Pe = norm_beta * h / (2.0 * eps)
    # tau = h / (2.0 * norm_beta) * (ufl.cosh(Pe) / ufl.sinh(Pe) - 1.0 / Pe) # optimal tau
    # Simpler tau for high Pe:
    tau = h / (2.0 * norm_beta)
    
    # Residual
    R_u = -eps * ufl.div(ufl.grad(u)) + ufl.dot(beta, ufl.grad(u)) - f
    # Test function perturbation
    v_supg = tau * ufl.dot(beta, ufl.grad(v))
    
    a_supg = ufl.inner(-eps * ufl.div(ufl.grad(u)) + ufl.dot(beta, ufl.grad(u)), v_supg) * ufl.dx
    L_supg = ufl.inner(f, v_supg) * ufl.dx
    
    a = a_G + a_supg
    L = L_G + L_supg
    
    # Solve
    problem = petsc.LinearProblem(
        a, L, bcs=[bc],
        petsc_options={"ksp_type": "gmres", "pc_type": "ilu", "ksp_rtol": 1e-10},
        petsc_options_prefix="cd_"
    )
    u_sol = problem.solve()
    
    # Get total iterations
    iterations = problem.solver.getIterationNumber()
    
    # Interpolation
    out_grid = case_spec["output"]["grid"]
    nx_out = out_grid["nx"]
    ny_out = out_grid["ny"]
    bbox = out_grid["bbox"]
    
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
            
    u_out = np.full((points.shape[1],), np.nan)
    if len(points_on_proc) > 0:
        vals = u_sol.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u_out[eval_map] = vals.flatten()
        
    u_grid = u_out.reshape((ny_out, nx_out))
    
    return {
        "u": u_grid,
        "solver_info": {
            "mesh_resolution": nx_mesh,
            "element_degree": 2,
            "ksp_type": "gmres",
            "pc_type": "ilu",
            "rtol": 1e-10,
            "iterations": iterations
        }
    }
