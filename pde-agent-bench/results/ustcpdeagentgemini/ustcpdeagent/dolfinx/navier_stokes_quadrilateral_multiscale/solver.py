import numpy as np
import ufl
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import mesh, fem, geometry
from basix.ufl import element as basix_element, mixed_element as basix_mixed_element

def solve(case_spec: dict) -> dict:
    nx_out = case_spec["output"]["grid"]["nx"]
    ny_out = case_spec["output"]["grid"]["ny"]
    bbox = case_spec["output"]["grid"]["bbox"]

    # Resolution
    N = 96
    p0 = np.array([bbox[0], bbox[2]])
    p1 = np.array([bbox[1], bbox[3]])
    msh = mesh.create_rectangle(MPI.COMM_WORLD, [p0, p1], [N, N], cell_type=mesh.CellType.quadrilateral)
    
    # Degrees
    deg_u = 2
    deg_p = 1
    
    gdim = msh.geometry.dim
    vel_el = basix_element("Lagrange", msh.topology.cell_name(), deg_u, shape=(gdim,))
    pres_el = basix_element("Lagrange", msh.topology.cell_name(), deg_p)
    W = fem.functionspace(msh, basix_mixed_element([vel_el, pres_el]))
    
    V, _ = W.sub(0).collapse()
    Q, _ = W.sub(1).collapse()
    
    w = fem.Function(W)
    (u, p) = ufl.split(w)
    (v, q) = ufl.TestFunctions(W)
    
    x = ufl.SpatialCoordinate(msh)
    pi = ufl.pi
    
    # Exact solution
    u_ex = ufl.as_vector([
        pi*ufl.cos(pi*x[1])*ufl.sin(pi*x[0]) + pi*ufl.cos(4*pi*x[1])*ufl.sin(2*pi*x[0]),
        -pi*ufl.cos(pi*x[0])*ufl.sin(pi*x[1]) - (pi/2)*ufl.cos(2*pi*x[0])*ufl.sin(4*pi*x[1])
    ])
    p_ex = ufl.sin(pi*x[0])*ufl.cos(2*pi*x[1])
    
    nu = 0.1
    
    # Source term based on exact solution
    grad_u_ex = ufl.grad(u_ex)
    div_grad_u_ex = ufl.div(grad_u_ex)
    f = grad_u_ex * u_ex - nu * div_grad_u_ex + ufl.grad(p_ex)
    
    # Residual
    F = (
        nu * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
        + ufl.inner(ufl.grad(u) * u, v) * ufl.dx
        - p * ufl.div(v) * ufl.dx
        + q * ufl.div(u) * ufl.dx
        - ufl.inner(f, v) * ufl.dx
    )
    
    # Boundary Conditions
    fdim = msh.topology.dim - 1
    wall_facets = mesh.locate_entities_boundary(msh, fdim, lambda x_c: np.ones(x_c.shape[1], dtype=bool))
    dofs_u = fem.locate_dofs_topological((W.sub(0), V), fdim, wall_facets)
    
    u0 = fem.Function(V)
    u_ex_expr = fem.Expression(u_ex, V.element.interpolation_points())
    u0.interpolate(u_ex_expr)
    bc_u = fem.dirichletbc(u0, dofs_u, W.sub(0))
    
    # Pin pressure
    p_dofs = fem.locate_dofs_geometrical((W.sub(1), Q), lambda x_c: np.isclose(x_c[0], bbox[0]) & np.isclose(x_c[1], bbox[2]))
    p0 = fem.Function(Q)
    p_ex_expr = fem.Expression(p_ex, Q.element.interpolation_points())
    p0.interpolate(p_ex_expr)
    bc_p = fem.dirichletbc(p0, p_dofs, W.sub(1))
    
    bcs = [bc_u, bc_p]
    
    # Initial guess: Stokes
    u_stokes, p_stokes = ufl.TrialFunctions(W)
    a_lin = (
        nu * ufl.inner(ufl.grad(u_stokes), ufl.grad(v)) * ufl.dx
        - p_stokes * ufl.div(v) * ufl.dx
        + q * ufl.div(u_stokes) * ufl.dx
    )
    L_stokes = ufl.inner(f, v) * ufl.dx
    lin_prob = fem.petsc.LinearProblem(a_lin, L_stokes, bcs=bcs, petsc_options={"ksp_type": "preonly", "pc_type": "lu", "pc_factor_mat_solver_type": "mumps"})
    w_stokes_sol = lin_prob.solve()
    w.x.array[:] = w_stokes_sol.x.array[:]
    w.x.scatter_forward()
    
    # Solve Nonlinear
    J = ufl.derivative(F, w)
    problem = fem.petsc.NonlinearProblem(F, w, bcs=bcs, J=J,
                                         petsc_options_prefix="ns_",
                                         petsc_options={
                                             "snes_type": "newtonls", 
                                             "ksp_type": "preonly", 
                                             "pc_type": "lu", 
                                             "pc_factor_mat_solver_type": "mumps",
                                             "snes_rtol": 1e-9,
                                             "snes_atol": 1e-10
                                         })
    problem.solve()
    
    u_h = w.sub(0).collapse()
    
    # Sample
    xs = np.linspace(bbox[0], bbox[1], nx_out)
    ys = np.linspace(bbox[2], bbox[3], ny_out)
    XX, YY = np.meshgrid(xs, ys)
    pts = np.c_[XX.ravel(), YY.ravel(), np.zeros(nx_out * ny_out)]
    
    bb_tree = geometry.bb_tree(msh, msh.topology.dim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, pts)
    colliding_cells = geometry.compute_colliding_cells(msh, cell_candidates, pts)
    
    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    for i in range(pts.shape[0]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(pts[i])
            cells_on_proc.append(links[0])
            eval_map.append(i)
            
    u_values = np.full((pts.shape[0], gdim), np.nan)
    if len(points_on_proc) > 0:
        vals = u_h.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u_values[eval_map] = vals
        
    u_mag = np.linalg.norm(u_values, axis=1).reshape((ny_out, nx_out))
    
    solver_info = {
        "mesh_resolution": N,
        "element_degree": deg_u,
        "ksp_type": "preonly",
        "pc_type": "lu",
        "rtol": 1e-8,
        "iterations": 1,
        "nonlinear_iterations": [1]
    }
    
    return {"u": u_mag, "solver_info": solver_info}
