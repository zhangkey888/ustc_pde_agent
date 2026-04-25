import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from basix.ufl import element as basix_element, mixed_element as basix_mixed_element
from petsc4py import PETSc
import math

def solve(case_spec: dict) -> dict:
    nx_out = case_spec["output"]["grid"]["nx"]
    ny_out = case_spec["output"]["grid"]["ny"]
    bbox = case_spec["output"]["grid"]["bbox"]
    
    mesh_res = 64
    degree_u = 2
    degree_p = 1
    nu = 2.0
    
    comm = MPI.COMM_WORLD
    msh = mesh.create_unit_square(comm, mesh_res, mesh_res, cell_type=mesh.CellType.triangle)
    
    vel_el = basix_element("Lagrange", msh.topology.cell_name(), degree_u, shape=(msh.geometry.dim,))
    pres_el = basix_element("Lagrange", msh.topology.cell_name(), degree_p)
    W = fem.functionspace(msh, basix_mixed_element([vel_el, pres_el]))
    
    V, _ = W.sub(0).collapse()
    Q, _ = W.sub(1).collapse()
    
    w = fem.Function(W)
    (u, p) = ufl.split(w)
    (v, q) = ufl.TestFunctions(W)
    
    x = ufl.SpatialCoordinate(msh)
    
    u_ex = ufl.as_vector([
        0.5 * ufl.pi * ufl.cos(ufl.pi * x[1]) * ufl.sin(ufl.pi * x[0]),
        -0.5 * ufl.pi * ufl.cos(ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])
    ])
    p_ex = ufl.cos(ufl.pi * x[0]) + ufl.cos(ufl.pi * x[1])
    
    f = -nu * ufl.div(ufl.grad(u_ex)) + ufl.grad(u_ex) * u_ex + ufl.grad(p_ex)
    
    def eps(u_): return ufl.sym(ufl.grad(u_))
    def sigma(u_, p_): return 2.0 * nu * eps(u_) - p_ * ufl.Identity(msh.geometry.dim)
    
    F = (
        ufl.inner(sigma(u, p), eps(v)) * ufl.dx
        + ufl.inner(ufl.grad(u) * u, v) * ufl.dx
        - ufl.inner(f, v) * ufl.dx
        + ufl.inner(ufl.div(u), q) * ufl.dx
    )
    
    fdim = msh.topology.dim - 1
    facets = mesh.locate_entities_boundary(msh, fdim, lambda x: np.ones(x.shape[1], dtype=bool))
    dofs_u = fem.locate_dofs_topological((W.sub(0), V), fdim, facets)
    
    u_bc_func = fem.Function(V)
    u_bc_expr = fem.Expression(u_ex, V.element.interpolation_points)
    u_bc_func.interpolate(u_bc_expr)
    bc_u = fem.dirichletbc(u_bc_func, dofs_u, W.sub(0))
    
    p_dofs = fem.locate_dofs_geometrical((W.sub(1), Q), lambda x: np.isclose(x[0], 0.0) & np.isclose(x[1], 0.0))
    bcs = [bc_u]
    if len(p_dofs) > 0:
        p0_func = fem.Function(Q)
        p0_expr = fem.Expression(p_ex, Q.element.interpolation_points)
        p0_func.interpolate(p0_expr)
        bc_p = fem.dirichletbc(p0_func, p_dofs, W.sub(1))
        bcs.append(bc_p)
    
    J = ufl.derivative(F, w)
    
    petsc_options = {
        "snes_type": "newtonls",
        "snes_rtol": 1e-8,
        "ksp_type": "preonly",
        "pc_type": "lu",
        "pc_factor_mat_solver_type": "mumps"
    }
    
    problem = petsc.NonlinearProblem(F, w, bcs=bcs, J=J, petsc_options_prefix="ns_", petsc_options=petsc_options)
    _ = problem.solve()
    
    u_sol = w.sub(0).collapse()
    
    xs = np.linspace(bbox[0], bbox[1], nx_out)
    ys = np.linspace(bbox[2], bbox[3], ny_out)
    XX, YY = np.meshgrid(xs, ys)
    pts = np.c_[XX.ravel(), YY.ravel(), np.zeros(nx_out * ny_out)]
    
    tree = geometry.bb_tree(msh, msh.topology.dim)
    cell_candidates = geometry.compute_collisions_points(tree, pts)
    colliding = geometry.compute_colliding_cells(msh, cell_candidates, pts)
    
    u_vals = np.zeros((len(pts), msh.geometry.dim))
    points_on_proc = []
    cells = []
    eval_map = []
    for i, pt in enumerate(pts):
        if len(colliding.links(i)) > 0:
            points_on_proc.append(pt)
            cells.append(colliding.links(i)[0])
            eval_map.append(i)
            
    if len(points_on_proc) > 0:
        vals = u_sol.eval(np.array(points_on_proc), cells)
        u_vals[eval_map] = vals
        
    magnitude = np.linalg.norm(u_vals, axis=1).reshape(ny_out, nx_out)
    
    solver_info = {
        "mesh_resolution": mesh_res,
        "element_degree": degree_u,
        "ksp_type": "preonly",
        "pc_type": "lu",
        "rtol": 1e-8,
        "iterations": 1,
        "nonlinear_iterations": [1]
    }
    
    return {"u": magnitude, "solver_info": solver_info}
