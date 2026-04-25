import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc
from dolfinx import geometry

def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD
    
    # Grid specs
    nx = case_spec["output"]["grid"]["nx"]
    ny = case_spec["output"]["grid"]["ny"]
    bbox = case_spec["output"]["grid"]["bbox"]
    
    # Solver specs
    mesh_res = 128
    degree = 2
    
    # Create mesh
    domain = mesh.create_unit_square(comm, mesh_res, mesh_res, cell_type=mesh.CellType.triangle)
    
    # Function space
    V = fem.functionspace(domain, ("Lagrange", degree))
    
    # Coordinates and exact expressions
    x = ufl.SpatialCoordinate(domain)
    u_exact = ufl.exp(x[0]) * ufl.sin(ufl.pi * x[1])
    
    # v = -Delta u
    # u_xx = u, u_yy = -pi^2 u => Delta u = (1 - pi^2) u
    v_exact = (ufl.pi**2 - 1.0) * u_exact
    
    # f = Delta^2 u = (1 - pi^2)^2 u
    f_expr = (1.0 - ufl.pi**2)**2 * u_exact
    
    # --- Solve for v ---
    v_trial = ufl.TrialFunction(V)
    w_test = ufl.TestFunction(V)
    
    a_v = ufl.inner(ufl.grad(v_trial), ufl.grad(w_test)) * ufl.dx
    L_v = ufl.inner(f_expr, w_test) * ufl.dx
    
    # BC for v
    v_bc_func = fem.Function(V)
    v_bc_func.interpolate(fem.Expression(v_exact, V.element.interpolation_points))
    domain.topology.create_connectivity(domain.topology.dim - 1, domain.topology.dim)
    boundary_facets = mesh.exterior_facet_indices(domain.topology)
    dofs = fem.locate_dofs_topological(V, domain.topology.dim - 1, boundary_facets)
    bc_v = fem.dirichletbc(v_bc_func, dofs)
    
    problem_v = petsc.LinearProblem(
        a_v, L_v, bcs=[bc_v],
        petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
        petsc_options_prefix="v_"
    )
    v_sol = problem_v.solve()
    
    # --- Solve for u ---
    u_trial = ufl.TrialFunction(V)
    
    a_u = ufl.inner(ufl.grad(u_trial), ufl.grad(w_test)) * ufl.dx
    L_u = ufl.inner(v_sol, w_test) * ufl.dx
    
    # BC for u
    u_bc_func = fem.Function(V)
    u_bc_func.interpolate(fem.Expression(u_exact, V.element.interpolation_points))
    bc_u = fem.dirichletbc(u_bc_func, dofs)
    
    problem_u = petsc.LinearProblem(
        a_u, L_u, bcs=[bc_u],
        petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
        petsc_options_prefix="u_"
    )
    u_sol = problem_u.solve()
    
    # Evaluate on target grid
    xs = np.linspace(bbox[0], bbox[1], nx)
    ys = np.linspace(bbox[2], bbox[3], ny)
    XX, YY = np.meshgrid(xs, ys)
    pts = np.c_[XX.ravel(), YY.ravel(), np.zeros(nx * ny)]
    
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
            
    u_values = np.full(len(pts), np.nan)
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
        "iterations": 2
    }
    
    return {"u": u_grid, "solver_info": solver_info}
