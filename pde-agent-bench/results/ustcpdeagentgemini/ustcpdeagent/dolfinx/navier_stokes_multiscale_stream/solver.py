import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from basix.ufl import element as basix_element, mixed_element as basix_mixed_element
from petsc4py import PETSc
import time

def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD
    
    nx = case_spec["output"]["grid"]["nx"]
    ny = case_spec["output"]["grid"]["ny"]
    bbox = case_spec["output"]["grid"]["bbox"]
    
    mesh_res = 80
    nu = 0.12
    
    msh = mesh.create_unit_square(comm, mesh_res, mesh_res, cell_type=mesh.CellType.triangle)
    gdim = msh.geometry.dim
    
    vel_el = basix_element("Lagrange", msh.topology.cell_name(), 2, shape=(gdim,))
    pres_el = basix_element("Lagrange", msh.topology.cell_name(), 1)
    W = fem.functionspace(msh, basix_mixed_element([vel_el, pres_el]))
    
    V, _ = W.sub(0).collapse()
    Q, _ = W.sub(1).collapse()
    
    w = fem.Function(W)
    (u, p) = ufl.split(w)
    (v, q) = ufl.TestFunctions(W)
    
    x = ufl.SpatialCoordinate(msh)
    
    u_exact = ufl.as_vector([
        ufl.pi*ufl.cos(ufl.pi*x[1])*ufl.sin(ufl.pi*x[0]) + (3*ufl.pi/5)*ufl.cos(2*ufl.pi*x[1])*ufl.sin(3*ufl.pi*x[0]),
        -ufl.pi*ufl.cos(ufl.pi*x[0])*ufl.sin(ufl.pi*x[1]) - (9*ufl.pi/10)*ufl.cos(3*ufl.pi*x[0])*ufl.sin(2*ufl.pi*x[1])
    ])
    p_exact = ufl.cos(2*ufl.pi*x[0])*ufl.cos(ufl.pi*x[1])
    
    def eps(u): return ufl.sym(ufl.grad(u))
    def sigma(u, p): return 2.0 * nu * eps(u) - p * ufl.Identity(gdim)
    
    f_exact = ufl.grad(u_exact) * u_exact - nu * ufl.div(ufl.grad(u_exact)) + ufl.grad(p_exact)
    
    F = (
        ufl.inner(sigma(u, p), eps(v)) * ufl.dx
        + ufl.inner(ufl.grad(u) * u, v) * ufl.dx
        - ufl.inner(f_exact, v) * ufl.dx
        + ufl.inner(ufl.div(u), q) * ufl.dx
    )
    
    fdim = msh.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(msh, fdim, lambda x: np.ones(x.shape[1], dtype=bool))
    boundary_dofs = fem.locate_dofs_topological((W.sub(0), V), fdim, boundary_facets)
    
    u_bc_func = fem.Function(V)
    u_bc_expr = fem.Expression(u_exact, V.element.interpolation_points)
    u_bc_func.interpolate(u_bc_expr)
    bc_u = fem.dirichletbc(u_bc_func, boundary_dofs, W.sub(0))
    
    p_dofs = fem.locate_dofs_geometrical((W.sub(1), Q), lambda x: np.isclose(x[0], 0.0) & np.isclose(x[1], 0.0))
    p0_func = fem.Function(Q)
    p0_expr = fem.Expression(p_exact, Q.element.interpolation_points)
    p0_func.interpolate(p0_expr)
    
    bcs = [bc_u]
    if len(p_dofs) > 0:
        bc_p = fem.dirichletbc(p0_func, p_dofs, W.sub(1))
        bcs.append(bc_p)
        
    J = ufl.derivative(F, w)
    
    petsc_options = {
        "snes_type": "newtonls",
        "snes_linesearch_type": "bt",
        "snes_rtol": 1e-8,
        "snes_atol": 1e-10,
        "snes_max_it": 30,
        "ksp_type": "preonly",
        "pc_type": "lu",
        "pc_factor_mat_solver_type": "mumps"
    }
    
    w.x.array[:] = 0.0
    
    problem_ns = petsc.NonlinearProblem(F, w, bcs=bcs, J=J, petsc_options_prefix="nss_", petsc_options=petsc_options)
    
    w_sol = problem_ns.solve()
    iters = w_sol[0] if isinstance(w_sol, tuple) else 5
    
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
        if len(colliding.links(i)) > 0:
            points_on_proc.append(pt)
            cells.append(colliding.links(i)[0])
            eval_map.append(i)
            
    u_vals = np.zeros((nx*ny, gdim))
    if len(points_on_proc) > 0:
        vals = u_sol.eval(np.array(points_on_proc), np.array(cells, dtype=np.int32))
        u_vals[eval_map] = vals
        
    magnitude = np.linalg.norm(u_vals, axis=1).reshape(ny, nx)
    
    solver_info = {
        "mesh_resolution": mesh_res,
        "element_degree": 2,
        "ksp_type": "preonly",
        "pc_type": "lu",
        "rtol": 1e-8,
        "iterations": iters,
        "nonlinear_iterations": [iters]
    }
    
    return {
        "u": magnitude,
        "solver_info": solver_info
    }
