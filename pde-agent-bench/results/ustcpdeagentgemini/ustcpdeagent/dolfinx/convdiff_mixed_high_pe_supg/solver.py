import numpy as np
import ufl
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
from petsc4py import PETSc

def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD
    
    # Problem params
    epsilon = 0.005
    beta_vec = [20.0, 10.0]
    
    # Mesh config
    nx_mesh = 120
    ny_mesh = 120
    domain = mesh.create_unit_square(comm, nx_mesh, ny_mesh, cell_type=mesh.CellType.triangle)
    
    # Function Space
    degree = 2
    V = fem.functionspace(domain, ("Lagrange", degree))
    
    # Exact solution
    x = ufl.SpatialCoordinate(domain)
    u_exact = ufl.sin(ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])
    
    lap_u = -2.0 * ufl.pi**2 * ufl.sin(ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])
    grad_u_x = ufl.pi * ufl.cos(ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])
    grad_u_y = ufl.pi * ufl.sin(ufl.pi * x[0]) * ufl.cos(ufl.pi * x[1])
    f_exact = -epsilon * lap_u + beta_vec[0] * grad_u_x + beta_vec[1] * grad_u_y
    
    # BCs
    tdim = domain.topology.dim
    fdim = tdim - 1
    
    def boundary_marker(x):
        return np.ones(x.shape[1], dtype=bool)
        
    boundary_facets = mesh.locate_entities_boundary(domain, fdim, boundary_marker)
    dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    
    u_bc = fem.Function(V)
    u_bc.interpolate(fem.Expression(u_exact, V.element.interpolation_points))
    bc = fem.dirichletbc(u_bc, dofs)
    
    # Variational problem
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    
    beta = fem.Constant(domain, PETSc.ScalarType(beta_vec))
    eps_c = fem.Constant(domain, PETSc.ScalarType(epsilon))
    
    # Galerkin
    F = eps_c * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx \
        + ufl.inner(beta, ufl.grad(u)) * v * ufl.dx \
        - f_exact * v * ufl.dx
        
    # SUPG
    h = ufl.CellDiameter(domain)
    norm_beta = ufl.sqrt(ufl.inner(beta, beta))
    tau = h / (2.0 * norm_beta)
    
    residual = -eps_c * ufl.div(ufl.grad(u)) + ufl.inner(beta, ufl.grad(u)) - f_exact
    F_supg = F + ufl.inner(beta, ufl.grad(v)) * tau * residual * ufl.dx
    
    a = ufl.lhs(F_supg)
    L = ufl.rhs(F_supg)
    
    # Solve
    ksp_type = "preonly"
    pc_type = "lu"
    rtol = 1e-8
    
    problem = petsc.LinearProblem(
        a, L, bcs=[bc],
        petsc_options={
            "ksp_type": ksp_type,
            "pc_type": pc_type,
        },
        petsc_options_prefix="cd_"
    )
    u_sol = problem.solve()
    
    iterations = problem.solver.getIterationNumber()
    
    # Evaluation
    grid_spec = case_spec["output"]["grid"]
    grid_nx = grid_spec["nx"]
    grid_ny = grid_spec["ny"]
    bbox = grid_spec["bbox"]
    
    xs = np.linspace(bbox[0], bbox[1], grid_nx)
    ys = np.linspace(bbox[2], bbox[3], grid_ny)
    XX, YY = np.meshgrid(xs, ys)
    pts = np.c_[XX.ravel(), YY.ravel(), np.zeros_like(XX.ravel())]
    
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
            
    u_values = np.full(pts.shape[0], np.nan)
    if len(points_on_proc) > 0:
        vals = u_sol.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u_values[eval_map] = vals.flatten()
        
    u_grid = u_values.reshape((grid_ny, grid_nx))
    
    return {
        "u": u_grid,
        "solver_info": {
            "mesh_resolution": nx_mesh,
            "element_degree": degree,
            "ksp_type": ksp_type,
            "pc_type": pc_type,
            "rtol": rtol,
            "iterations": iterations
        }
    }
