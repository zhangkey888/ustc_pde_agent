import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
import dolfinx.fem.petsc as petsc
import ufl
from petsc4py import PETSc

def solve(case_spec: dict) -> dict:
    nx = case_spec["output"]["grid"]["nx"]
    ny = case_spec["output"]["grid"]["ny"]
    bbox = case_spec["output"]["grid"]["bbox"]
    
    # We will use a reasonably fine mesh and P2 elements to hit the accuracy target
    mesh_res = 128
    degree = 2
    
    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_square(comm, mesh_res, mesh_res, cell_type=mesh.CellType.triangle)
    
    V = fem.functionspace(domain, ("Lagrange", degree))
    
    # Parameters
    epsilon = 0.002
    beta_vec = np.array([25.0, 10.0])
    beta = fem.Constant(domain, PETSc.ScalarType(beta_vec))
    eps_const = fem.Constant(domain, PETSc.ScalarType(epsilon))
    
    x = ufl.SpatialCoordinate(domain)
    
    # Exact solution
    u_exact = ufl.sin(ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])
    
    # Right-hand side f = -eps * div(grad(u)) + beta . grad(u)
    # div(grad(u)) = -2 * pi^2 * sin(pi x) * sin(pi y)
    f = eps_const * 2.0 * ufl.pi**2 * u_exact \
        + beta_vec[0] * ufl.pi * ufl.cos(ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1]) \
        + beta_vec[1] * ufl.pi * ufl.sin(ufl.pi * x[0]) * ufl.cos(ufl.pi * x[1])
        
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    
    # Standard weak form
    a = eps_const * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx \
        + ufl.inner(beta, ufl.grad(u)) * v * ufl.dx
    L = f * v * ufl.dx
    
    # SUPG Stabilization
    h = ufl.CellDiameter(domain)
    beta_norm = ufl.sqrt(ufl.inner(beta, beta))
    
    # SUPG parameter tau
    # tau = h / (2 * beta_norm) for high Pe
    tau = h / (2.0 * beta_norm)
    
    # Residual of the strong form
    R = -eps_const * ufl.div(ufl.grad(u)) + ufl.inner(beta, ufl.grad(u)) - f
    
    # SUPG test function modification
    v_supg = tau * ufl.inner(beta, ufl.grad(v))
    
    a_supg = a + R * v_supg * ufl.dx(m=u) # extracting the u-dependent part is tricky in ufl without splitting
    # Actually R = L(u) - f, let's write it explicitly
    a_stab = a + (-eps_const * ufl.div(ufl.grad(u)) + ufl.inner(beta, ufl.grad(u))) * v_supg * ufl.dx
    L_stab = L + f * v_supg * ufl.dx
    
    # Boundary Conditions
    tdim = domain.topology.dim
    fdim = tdim - 1
    domain.topology.create_connectivity(fdim, tdim)
    boundary_facets = mesh.exterior_facet_indices(domain.topology)
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    
    u_bc = fem.Function(V)
    u_bc.interpolate(lambda x_pts: np.sin(np.pi * x_pts[0]) * np.sin(np.pi * x_pts[1]))
    bc = fem.dirichletbc(u_bc, boundary_dofs)
    
    # Solve
    problem = petsc.LinearProblem(a_stab, L_stab, bcs=[bc], 
                                  petsc_options={"ksp_type": "preonly", "pc_type": "lu", "pc_factor_mat_solver_type": "mumps"},
                                  petsc_options_prefix="cd_")
    u_sol = problem.solve()
    
    # Evaluation on grid
    xs = np.linspace(bbox[0], bbox[1], nx)
    ys = np.linspace(bbox[2], bbox[3], ny)
    XX, YY = np.meshgrid(xs, ys)
    points = np.vstack((XX.ravel(), YY.ravel(), np.zeros_like(XX.ravel())))
    
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
            
    u_values = np.zeros(points.shape[1])
    if len(points_on_proc) > 0:
        vals = u_sol.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u_values[eval_map] = vals.flatten()
        
    u_grid = u_values.reshape((ny, nx))
    
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

