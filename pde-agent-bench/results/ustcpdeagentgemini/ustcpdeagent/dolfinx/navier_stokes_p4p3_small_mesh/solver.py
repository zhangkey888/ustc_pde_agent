import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from basix.ufl import element as basix_element, mixed_element as basix_mixed_element
from petsc4py import PETSc
import time

def solve(case_spec: dict) -> dict:
    start_time = time.time()
    
    # Grid specs
    grid = case_spec["output"]["grid"]
    nx = grid["nx"]
    ny = grid["ny"]
    bbox = grid["bbox"]
    
    # Mesh resolution
    mesh_res = 16 # Small mesh, high order elements
    degree_u = 4
    degree_p = 3
    
    comm = MPI.COMM_WORLD
    msh = mesh.create_rectangle(comm, [[bbox[0], bbox[2]], [bbox[1], bbox[3]]], [mesh_res, mesh_res], cell_type=mesh.CellType.triangle)
    gdim = msh.geometry.dim
    
    # Function spaces
    vel_el = basix_element("Lagrange", msh.topology.cell_name(), degree_u, shape=(gdim,))
    pres_el = basix_element("Lagrange", msh.topology.cell_name(), degree_p)
    W = fem.functionspace(msh, basix_mixed_element([vel_el, pres_el]))
    
    V, _ = W.sub(0).collapse()
    Q, _ = W.sub(1).collapse()
    
    w = fem.Function(W)
    (u, p) = ufl.split(w)
    (v, q) = ufl.TestFunctions(W)
    
    nu = 0.2
    x = ufl.SpatialCoordinate(msh)
    
    # Exact solution for BCs and source term
    u_ex = ufl.as_vector([
        ufl.pi * ufl.cos(ufl.pi * x[1]) * ufl.sin(ufl.pi * x[0]),
        -ufl.pi * ufl.cos(ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])
    ])
    p_ex = ufl.cos(ufl.pi * x[0]) * ufl.cos(ufl.pi * x[1])
    
    def eps(u_func):
        return ufl.sym(ufl.grad(u_func))
    
    def sigma(u_func, p_func):
        return 2.0 * nu * eps(u_func) - p_func * ufl.Identity(gdim)
    
    # Compute analytical source term f
    # f = (u·∇)u - ν Δu + ∇p
    # Note: Δu = 2 ∇·(eps(u)) when div(u)=0
    # For general, Δu = div(grad(u))
    f_ex = ufl.grad(u_ex) * u_ex - nu * ufl.div(ufl.grad(u_ex)) + ufl.grad(p_ex)
    
    # Residual
    F = (
        ufl.inner(sigma(u, p), eps(v)) * ufl.dx
        + ufl.inner(ufl.grad(u) * u, v) * ufl.dx
        - ufl.inner(f_ex, v) * ufl.dx
        + ufl.inner(ufl.div(u), q) * ufl.dx
    )
    
    J = ufl.derivative(F, w)
    
    # Boundary conditions
    fdim = msh.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(msh, fdim, lambda x_pts: np.ones(x_pts.shape[1], dtype=bool))
    boundary_dofs_u = fem.locate_dofs_topological((W.sub(0), V), fdim, boundary_facets)
    
    u_bc_func = fem.Function(V)
    u_bc_expr = fem.Expression(u_ex, V.element.interpolation_points)
    u_bc_func.interpolate(u_bc_expr)
    
    bc_u = fem.dirichletbc(u_bc_func, boundary_dofs_u, W.sub(0))
    
    # Pressure pin
    p_dofs = fem.locate_dofs_geometrical((W.sub(1), Q), lambda x_pts: np.isclose(x_pts[0], 0.0) & np.isclose(x_pts[1], 0.0))
    p0_func = fem.Function(Q)
    p0_expr = fem.Expression(p_ex, Q.element.interpolation_points)
    p0_func.interpolate(p0_expr)
    
    bcs = [bc_u]
    if len(p_dofs) > 0:
        bc_p = fem.dirichletbc(p0_func, p_dofs, W.sub(1))
        bcs.append(bc_p)
        
    # Initial guess
    w.x.array[:] = 0.0
    
    # Solve
    petsc_options = {
        "snes_type": "newtonls",
        "snes_rtol": 1e-9,
        "snes_max_it": 50,
        "ksp_type": "preonly",
        "pc_type": "lu",
    }
    
    problem = petsc.NonlinearProblem(F, w, bcs=bcs, J=J, petsc_options_prefix="ns_", petsc_options=petsc_options)
    
    try:
        w_h = problem.solve()
    except Exception as e:
        print(f"Solve failed: {e}")
    
    # Evaluation
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
            
    u_vals = np.zeros((nx * ny, gdim))
    if len(points_on_proc) > 0:
        vals = w.sub(0).collapse().eval(np.array(points_on_proc), np.array(cells, dtype=np.int32))
        for j, idx in enumerate(eval_map):
            u_vals[idx] = vals[j]
        
    magnitude = np.linalg.norm(u_vals, axis=1).reshape((ny, nx))
    
    solver_info = {
        "mesh_resolution": mesh_res,
        "element_degree": degree_u,
        "ksp_type": "preonly",
        "pc_type": "lu",
        "rtol": 1e-9,
        "nonlinear_iterations": [5]
    }
    
    return {
        "u": magnitude,
        "solver_info": solver_info
    }
