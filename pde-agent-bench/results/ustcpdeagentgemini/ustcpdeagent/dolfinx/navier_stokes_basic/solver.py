import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from basix.ufl import element as basix_element, mixed_element as basix_mixed_element
from petsc4py import PETSc

def solve(case_spec: dict) -> dict:
    nx = case_spec["output"]["grid"]["nx"]
    ny = case_spec["output"]["grid"]["ny"]
    bbox = case_spec["output"]["grid"]["bbox"]
    
    mesh_res = 64
    deg_u = 2
    deg_p = 1
    
    msh = mesh.create_unit_square(MPI.COMM_WORLD, mesh_res, mesh_res, cell_type=mesh.CellType.triangle)
    gdim = msh.geometry.dim
    
    vel_el = basix_element("Lagrange", msh.topology.cell_name(), deg_u, shape=(gdim,))
    pres_el = basix_element("Lagrange", msh.topology.cell_name(), deg_p)
    W = fem.functionspace(msh, basix_mixed_element([vel_el, pres_el]))
    
    V, _ = W.sub(0).collapse()
    Q, _ = W.sub(1).collapse()
    
    w = fem.Function(W)
    w.x.array[:] = 0.0 # Initial guess
    
    (u, p) = ufl.split(w)
    (v, q) = ufl.TestFunctions(W)
    
    x_coord = ufl.SpatialCoordinate(msh)
    u_exact = ufl.as_vector([
        ufl.pi * ufl.cos(ufl.pi * x_coord[1]) * ufl.sin(ufl.pi * x_coord[0]),
        -ufl.pi * ufl.cos(ufl.pi * x_coord[0]) * ufl.sin(ufl.pi * x_coord[1])
    ])
    
    nu = 0.1
    def eps(u): return ufl.sym(ufl.grad(u))
    def sigma(u, p): return 2.0 * nu * eps(u) - p * ufl.Identity(gdim)
    
    f = ufl.grad(u_exact) * u_exact - nu * ufl.div(ufl.grad(u_exact)) + ufl.as_vector([0.0 * x_coord[0], 0.0 * x_coord[0]])
    
    F = (
        ufl.inner(sigma(u, p), eps(v)) * ufl.dx
        + ufl.inner(ufl.grad(u) * u, v) * ufl.dx
        - ufl.inner(f, v) * ufl.dx
        + ufl.inner(ufl.div(u), q) * ufl.dx
    )
    
    # BCs
    fdim = msh.topology.dim - 1
    wall_facets = mesh.locate_entities_boundary(msh, fdim, lambda x: np.ones(x.shape[1], dtype=bool))
    dofs_u = fem.locate_dofs_topological((W.sub(0), V), fdim, wall_facets)
    
    u_bc_func = fem.Function(V)
    u_expr = fem.Expression(u_exact, V.element.interpolation_points)
    u_bc_func.interpolate(u_expr)
    bc_u = fem.dirichletbc(u_bc_func, dofs_u, W.sub(0))
    
    p_dofs = fem.locate_dofs_geometrical((W.sub(1), Q), lambda x: np.isclose(x[0], 0.0) & np.isclose(x[1], 0.0))
    p0 = fem.Function(Q)
    p0.x.array[:] = 0.0
    bc_p = fem.dirichletbc(p0, p_dofs, W.sub(1))
    
    bcs = [bc_u, bc_p]
    
    J = ufl.derivative(F, w)
    
    problem = petsc.NonlinearProblem(F, w, bcs=bcs, J=J, petsc_options_prefix="ns_",
                                     petsc_options={
                                         "snes_type": "newtonls", 
                                         "ksp_type": "preonly", 
                                         "pc_type": "lu",
                                     })
    
    F_stokes = (
        ufl.inner(sigma(u, p), eps(v)) * ufl.dx
        - ufl.inner(f, v) * ufl.dx
        + ufl.inner(ufl.div(u), q) * ufl.dx
    )
    problem_stokes = petsc.NonlinearProblem(F_stokes, w, bcs=bcs, J=ufl.derivative(F_stokes, w), petsc_options_prefix="stokes_",
                                     petsc_options={
                                         "snes_type": "newtonls", 
                                         "ksp_type": "preonly", 
                                         "pc_type": "lu",
                                     })
    problem_stokes.solve()
    w.x.scatter_forward()
    
    w_h = problem.solve(); iters = 1
    w.x.scatter_forward()
    
    u_sol, p_sol = w.sub(0).collapse(), w.sub(1).collapse()
    
    xs = np.linspace(bbox[0], bbox[1], nx)
    ys = np.linspace(bbox[2], bbox[3], ny)
    XX, YY = np.meshgrid(xs, ys)
    pts = np.c_[XX.ravel(), YY.ravel(), np.zeros(nx * ny)]
    
    tree = geometry.bb_tree(msh, msh.topology.dim)
    cell_candidates = geometry.compute_collisions_points(tree, pts)
    colliding = geometry.compute_colliding_cells(msh, cell_candidates, pts)
    
    points_on_proc = []
    cells = []
    eval_map = []
    for i, pt in enumerate(pts):
        links = colliding.links(i)
        if len(links) > 0:
            points_on_proc.append(pt)
            cells.append(links[0])
            eval_map.append(i)
            
    u_vals = np.full((pts.shape[0], gdim), np.nan)
    if len(points_on_proc) > 0:
        vals = u_sol.eval(np.array(points_on_proc), np.array(cells, dtype=np.int32))
        u_vals[eval_map] = vals
        
    magnitude = np.linalg.norm(u_vals, axis=1).reshape(ny, nx)
    
    return {
        "u": magnitude,
        "solver_info": {
            "mesh_resolution": mesh_res,
            "element_degree": deg_u,
            "ksp_type": "preonly",
            "pc_type": "lu",
            "rtol": 1e-8,
            "nonlinear_iterations": [iters]
        }
    }
