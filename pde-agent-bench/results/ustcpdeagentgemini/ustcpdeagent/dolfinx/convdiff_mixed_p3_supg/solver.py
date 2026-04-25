import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc
import time

def solve(case_spec: dict) -> dict:
    t0 = time.time()
    
    # Extract parameters
    nx = case_spec["output"]["grid"]["nx"]
    ny = case_spec["output"]["grid"]["ny"]
    bbox = case_spec["output"]["grid"]["bbox"]
    
    # Solver configuration
    mesh_res = 128
    degree = 2
    
    comm = MPI.COMM_WORLD
    domain = mesh.create_rectangle(comm, [[0.0, 0.0], [1.0, 1.0]], [mesh_res, mesh_res], cell_type=mesh.CellType.triangle)
    
    V = fem.functionspace(domain, ("Lagrange", degree))
    
    eps = 0.01
    beta_val = [12.0, 4.0]
    beta = fem.Constant(domain, PETSc.ScalarType(beta_val))
    
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    
    x = ufl.SpatialCoordinate(domain)
    u_exact = ufl.sin(ufl.pi * x[0]) * ufl.sin(2 * ufl.pi * x[1])
    
    # Source term f = -eps * laplacian(u) + beta \cdot grad(u)
    f_exact = -eps * ( -ufl.pi**2 * u_exact - 4*ufl.pi**2 * u_exact ) + \
              beta_val[0] * ufl.pi * ufl.cos(ufl.pi * x[0]) * ufl.sin(2 * ufl.pi * x[1]) + \
              beta_val[1] * 2 * ufl.pi * ufl.sin(ufl.pi * x[0]) * ufl.cos(2 * ufl.pi * x[1])
              
    f = fem.Expression(f_exact, V.element.interpolation_points())
    f_func = fem.Function(V)
    f_func.interpolate(f)
    
    # Boundary conditions
    fdim = domain.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(domain, fdim, lambda x: np.full(x.shape[1], True, dtype=bool))
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    
    u_bc = fem.Function(V)
    u_bc.interpolate(fem.Expression(u_exact, V.element.interpolation_points()))
    bc = fem.dirichletbc(u_bc, boundary_dofs)
    
    # SUPG stabilization
    h = ufl.CellDiameter(domain)
    vnorm = ufl.sqrt(ufl.dot(beta, beta))
    # tau = h / (2 * vnorm)  # simple choice
    tau = ((2.0 * vnorm / h)**2 + 9.0 * (4.0 * eps / h**2)**2)**(-0.5)
    
    # Residual
    F_form = eps * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx \
           + ufl.inner(ufl.dot(beta, ufl.grad(u)), v) * ufl.dx \
           - ufl.inner(f_func, v) * ufl.dx
           
    # SUPG term
    R = -eps * ufl.div(ufl.grad(u)) + ufl.dot(beta, ufl.grad(u)) - f_func
    F_supg = F_form + tau * ufl.inner(R, ufl.dot(beta, ufl.grad(v))) * ufl.dx
    
    a, L = ufl.lhs(F_supg), ufl.rhs(F_supg)
    
    problem = petsc.LinearProblem(
        a, L, bcs=[bc],
        petsc_options={"ksp_type": "preonly", "pc_type": "lu", "pc_factor_mat_solver_type": "mumps"},
        petsc_options_prefix="convdiff_"
    )
    u_sol = problem.solve()
    
    # Interpolation onto grid
    xs = np.linspace(bbox[0], bbox[1], nx)
    ys = np.linspace(bbox[2], bbox[3], ny)
    XX, YY = np.meshgrid(xs, ys)
    pts = np.c_[XX.ravel(), YY.ravel(), np.zeros(nx * ny)]
    
    bb_tree_obj = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(bb_tree_obj, pts)
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
            
    u_out = np.full(nx * ny, np.nan)
    if len(points_on_proc) > 0:
        vals = u_sol.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u_out[eval_map] = vals.flatten()
        
    u_grid = u_out.reshape((ny, nx))
    
    solver_info = {
        "mesh_resolution": mesh_res,
        "element_degree": degree,
        "ksp_type": "preonly",
        "pc_type": "lu",
        "rtol": 1e-8,
        "iterations": 1
    }
    
    return {"u": u_grid, "solver_info": solver_info}
