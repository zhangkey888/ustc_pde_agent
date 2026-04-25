import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
import ufl
from petsc4py import PETSc
from dolfinx.fem.petsc import LinearProblem

def solve(case_spec: dict) -> dict:
    nx_grid = case_spec["output"]["grid"]["nx"]
    ny_grid = case_spec["output"]["grid"]["ny"]
    bbox = case_spec["output"]["grid"]["bbox"]

    mesh_res = 128
    degree = 2
    
    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_square(comm, mesh_res, mesh_res, cell_type=mesh.CellType.triangle)
    
    V = fem.functionspace(domain, ("Lagrange", degree))
    
    x = ufl.SpatialCoordinate(domain)
    
    # Exact solution
    # u = exp(5*(x-1)) * sin(pi*y)
    u_exact = ufl.exp(5 * (x[0] - 1)) * ufl.sin(ufl.pi * x[1])
    
    # v = -Laplacian(u)
    # d^2u/dx^2 = 25 * exp(5*(x-1)) * sin(pi*y)
    # d^2u/dy^2 = -pi^2 * exp(5*(x-1)) * sin(pi*y)
    # v = -(25 - pi^2) * exp(...) * sin(...)
    v_exact = -(25 - ufl.pi**2) * ufl.exp(5 * (x[0] - 1)) * ufl.sin(ufl.pi * x[1])
    
    # f = -Laplacian(v)
    f_exact = (25 - ufl.pi**2)**2 * ufl.exp(5 * (x[0] - 1)) * ufl.sin(ufl.pi * x[1])
    
    f_expr = fem.Expression(f_exact, V.element.interpolation_points)
    f = fem.Function(V)
    f.interpolate(f_expr)
    
    # Boundary conditions
    tdim = domain.topology.dim
    fdim = tdim - 1
    domain.topology.create_connectivity(fdim, tdim)
    boundary_facets = mesh.exterior_facet_indices(domain.topology)
    dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    
    # Solve for v
    v_trial = ufl.TrialFunction(V)
    w_test = ufl.TestFunction(V)
    
    a_v = ufl.inner(ufl.grad(v_trial), ufl.grad(w_test)) * ufl.dx
    L_v = ufl.inner(f, w_test) * ufl.dx
    
    v_bc_expr = fem.Expression(v_exact, V.element.interpolation_points)
    v_bc_func = fem.Function(V)
    v_bc_func.interpolate(v_bc_expr)
    bc_v = fem.dirichletbc(v_bc_func, dofs)
    
    problem_v = LinearProblem(a_v, L_v, bcs=[bc_v], petsc_options={"ksp_type": "preonly", "pc_type": "lu"}, petsc_options_prefix="v_")
    v_sol = problem_v.solve()
    
    # Solve for u
    u_trial = ufl.TrialFunction(V)
    w_test2 = ufl.TestFunction(V)
    
    a_u = ufl.inner(ufl.grad(u_trial), ufl.grad(w_test2)) * ufl.dx
    L_u = ufl.inner(v_sol, w_test2) * ufl.dx
    
    u_bc_expr = fem.Expression(u_exact, V.element.interpolation_points)
    u_bc_func = fem.Function(V)
    u_bc_func.interpolate(u_bc_expr)
    bc_u = fem.dirichletbc(u_bc_func, dofs)
    
    problem_u = LinearProblem(a_u, L_u, bcs=[bc_u], petsc_options={"ksp_type": "preonly", "pc_type": "lu"}, petsc_options_prefix="u_")
    u_sol = problem_u.solve()
    
    # Evaluate on grid
    xs = np.linspace(bbox[0], bbox[1], nx_grid)
    ys = np.linspace(bbox[2], bbox[3], ny_grid)
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
            
    u_values = np.full(points.shape[1], np.nan)
    if len(points_on_proc) > 0:
        vals = u_sol.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u_values[eval_map] = vals.flatten()
        
    u_grid = u_values.reshape((ny_grid, nx_grid))
    
    return {
        "u": u_grid,
        "solver_info": {
            "mesh_resolution": mesh_res,
            "element_degree": degree,
            "ksp_type": "preonly",
            "pc_type": "lu",
            "rtol": 1e-8,
            "iterations": 2
        }
    }
