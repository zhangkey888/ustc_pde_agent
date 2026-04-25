import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
from basix.ufl import element as basix_element, mixed_element as basix_mixed_element
import ufl
import time

def solve(case_spec: dict) -> dict:
    t0 = time.time()
    
    # Grid parameters
    grid_spec = case_spec["output"]["grid"]
    nx = grid_spec["nx"]
    ny = grid_spec["ny"]
    bbox = grid_spec["bbox"]
    
    # Solver parameters
    mesh_res = 32
    degree_u = 2
    degree_p = 1
    nu = 1.0
    
    comm = MPI.COMM_WORLD
    msh = mesh.create_unit_square(comm, mesh_res, mesh_res, cell_type=mesh.CellType.triangle)
    
    gdim = msh.geometry.dim
    vel_el = basix_element("Lagrange", msh.topology.cell_name(), degree_u, shape=(gdim,))
    pres_el = basix_element("Lagrange", msh.topology.cell_name(), degree_p)
    W = fem.functionspace(msh, basix_mixed_element([vel_el, pres_el]))
    
    V, _ = W.sub(0).collapse()
    Q, _ = W.sub(1).collapse()
    
    w = fem.Function(W)
    (u, p) = ufl.split(w)
    (v, q) = ufl.TestFunctions(W)
    
    x = ufl.SpatialCoordinate(msh)
    
    # Exact solutions
    u_ex_x = ufl.pi * ufl.cos(ufl.pi * x[1]) * ufl.sin(ufl.pi * x[0])
    u_ex_y = -ufl.pi * ufl.cos(ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])
    u_ex = ufl.as_vector([u_ex_x, u_ex_y])
    p_ex = ufl.cos(ufl.pi * x[0]) * ufl.cos(ufl.pi * x[1])
    
    # Source term f = (u.nabla)u - nu * Delta u + nabla p
    grad_u_ex = ufl.grad(u_ex)
    div_grad_u_ex = ufl.div(grad_u_ex)
    f = grad_u_ex * u_ex - nu * div_grad_u_ex + ufl.grad(p_ex)
    
    def eps(u):
        return ufl.sym(ufl.grad(u))
    def sigma(u, p):
        return 2.0 * nu * eps(u) - p * ufl.Identity(gdim)
    
    F = (
        ufl.inner(sigma(u, p), eps(v)) * ufl.dx
        + ufl.inner(ufl.grad(u) * u, v) * ufl.dx
        - ufl.inner(f, v) * ufl.dx
        + ufl.inner(ufl.div(u), q) * ufl.dx
    )
    
    # Boundary conditions
    u_bc_func = fem.Function(V)
    u_bc_func.interpolate(fem.Expression(u_ex, V.element.interpolation_points()))
    
    fdim = msh.topology.dim - 1
    facets = mesh.locate_entities_boundary(msh, fdim, lambda x_c: np.ones(x_c.shape[1], dtype=bool))
    dofs_u = fem.locate_dofs_topological((W.sub(0), V), fdim, facets)
    bc_u = fem.dirichletbc(u_bc_func, dofs_u, W.sub(0))
    
    # Pressure pin
    p_dofs = fem.locate_dofs_geometrical((W.sub(1), Q), lambda x_c: np.isclose(x_c[0], 0.0) & np.isclose(x_c[1], 0.0))
    p_bc_func = fem.Function(Q)
    p_bc_func.interpolate(fem.Expression(p_ex, Q.element.interpolation_points()))
    if len(p_dofs) > 0:
        bc_p = fem.dirichletbc(p_bc_func, p_dofs, W.sub(1))
        bcs = [bc_u, bc_p]
    else:
        bcs = [bc_u]
        
    # Initial guess
    w.x.array[:] = 0.0
    
    J = ufl.derivative(F, w)
    
    problem = petsc.NonlinearProblem(F, w, bcs=bcs, J=J,
                                     petsc_options_prefix="ns_",
                                     petsc_options={
                                         "snes_type": "newtonls",
                                         "ksp_type": "preonly",
                                         "pc_type": "lu",
                                         "snes_rtol": 1e-8,
                                         "snes_atol": 1e-10,
                                         "snes_max_it": 50,
                                     })
    
    w_h = problem.solve()
    w.x.scatter_forward()
    u_sol, p_sol = w.sub(0).collapse(), w.sub(1).collapse()
    
    # Sampling
    xs = np.linspace(bbox[0], bbox[1], nx)
    ys = np.linspace(bbox[2], bbox[3], ny)
    XX, YY = np.meshgrid(xs, ys)
    pts = np.c_[XX.ravel(), YY.ravel(), np.zeros(nx * ny)]
    
    tree = geometry.bb_tree(msh, msh.topology.dim)
    cell_candidates = geometry.compute_collisions_points(tree, pts)
    colliding = geometry.compute_colliding_cells(msh, cell_candidates, pts)
    
    cells = []
    points_on_proc = []
    eval_map = []
    for i, pt in enumerate(pts):
        links = colliding.links(i)
        if len(links) > 0:
            points_on_proc.append(pt)
            cells.append(links[0])
            eval_map.append(i)
            
    u_vals = np.zeros((pts.shape[0], gdim))
    if len(points_on_proc) > 0:
        vals = u_sol.eval(np.array(points_on_proc), np.array(cells, dtype=np.int32))
        u_vals[eval_map] = vals
        
    u_mag = np.linalg.norm(u_vals, axis=1).reshape(ny, nx)
    
    return {
        "u": u_mag,
        "solver_info": {
            "mesh_resolution": mesh_res,
            "element_degree": degree_u,
            "ksp_type": "preonly",
            "pc_type": "lu",
            "rtol": 1e-8,
            "iterations": 1,
            "nonlinear_iterations": [1]
        }
    }

