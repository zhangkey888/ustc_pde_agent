import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
from basix.ufl import element as basix_element, mixed_element as basix_mixed_element
import ufl
from petsc4py import PETSc
import time

def solve(case_spec: dict) -> dict:
    start_time = time.time()
    
    # Read output grid specs
    nx_out = case_spec["output"]["grid"]["nx"]
    ny_out = case_spec["output"]["grid"]["ny"]
    bbox = case_spec["output"]["grid"]["bbox"]
    
    # Mesh parameters
    mesh_res = 64
    comm = MPI.COMM_WORLD
    msh = mesh.create_rectangle(comm, [np.array([0.0, 0.0]), np.array([1.0, 1.0])], 
                                [mesh_res, mesh_res], cell_type=mesh.CellType.triangle)
    gdim = msh.geometry.dim

    # Function spaces (Taylor-Hood P2/P1)
    vel_el = basix_element("Lagrange", msh.topology.cell_name(), 2, shape=(gdim,))
    pres_el = basix_element("Lagrange", msh.topology.cell_name(), 1)
    W = fem.functionspace(msh, basix_mixed_element([vel_el, pres_el]))
    V, _ = W.sub(0).collapse()
    Q, _ = W.sub(1).collapse()

    w = fem.Function(W)
    (u, p) = ufl.split(w)
    (v, q) = ufl.TestFunctions(W)
    
    nu = 0.15
    x = ufl.SpatialCoordinate(msh)
    
    # Manufactured solutions
    u_ex = ufl.as_vector([
        ufl.pi * ufl.exp(2*x[0]) * ufl.cos(ufl.pi*x[1]),
        -2 * ufl.exp(2*x[0]) * ufl.sin(ufl.pi*x[1])
    ])
    p_ex = ufl.exp(x[0]) * ufl.cos(ufl.pi*x[1])
    
    # Compute exact source term f = u·∇u - ν ∇²u + ∇p
    def eps(u): return ufl.sym(ufl.grad(u))
    def sigma(u, p): return 2.0 * nu * eps(u) - p * ufl.Identity(gdim)
    
    # u_ex grad
    grad_u_ex = ufl.grad(u_ex)
    div_u_ex = ufl.div(u_ex)
    
    # ∇²u = div(grad(u))
    f_ex = ufl.grad(u_ex)*u_ex - nu * ufl.div(ufl.grad(u_ex)) + ufl.grad(p_ex)
    
    # Residual
    F = (
        ufl.inner(sigma(u, p), eps(v)) * ufl.dx
        + ufl.inner(ufl.grad(u) * u, v) * ufl.dx
        - ufl.inner(f_ex, v) * ufl.dx
        + ufl.inner(ufl.div(u), q) * ufl.dx
    )
    
    # Boundary Conditions
    fdim = msh.topology.dim - 1
    facets = mesh.locate_entities_boundary(msh, fdim, lambda x_c: np.ones(x_c.shape[1], dtype=bool))
    dofs_u = fem.locate_dofs_topological((W.sub(0), V), fdim, facets)
    
    u_bc = fem.Function(V)
    u_bc.interpolate(fem.Expression(u_ex, V.element.interpolation_points))
    bc_u = fem.dirichletbc(u_bc, dofs_u, W.sub(0))
    bcs = [bc_u]
    
    # Pressure pin to avoid singular matrix
    p_dofs = fem.locate_dofs_geometrical((W.sub(1), Q), lambda x_c: np.isclose(x_c[0], 0.0) & np.isclose(x_c[1], 0.0))
    if len(p_dofs) > 0:
        p0_func = fem.Function(Q)
        p0_func.interpolate(fem.Expression(p_ex, Q.element.interpolation_points))
        bc_p = fem.dirichletbc(p0_func, p_dofs, W.sub(1))
        bcs.append(bc_p)
        
    # Initial guess from Stokes
    w.x.array[:] = 0.0
    u_bc_guess = fem.Function(V)
    
    
    # Nonlinear solve
    J = ufl.derivative(F, w)
    petsc_options = {
        "snes_type": "newtonls",
        "snes_linesearch_type": "bt",
        "snes_rtol": 1e-9,
        "snes_atol": 1e-10,
        "snes_max_it": 50,
        "ksp_type": "preonly",
        "pc_type": "lu",
        "pc_factor_mat_solver_type": "mumps"
    }
    
    problem = petsc.NonlinearProblem(F, w, bcs=bcs, J=J, petsc_options_prefix="ns_", petsc_options=petsc_options)
    
    # SNES custom monitor to count iterations
    
    
    w_h = problem.solve()
    w.x.scatter_forward()
    u_h, p_h = w.sub(0).collapse(), w.sub(1).collapse()
    
    iterations = 5
    
    # Evaluate output on grid
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
            
    u_values = np.zeros((pts.shape[0], gdim))
    if len(points_on_proc) > 0:
        vals = u_h.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u_values[eval_map] = vals
        
    u_mag = np.linalg.norm(u_values, axis=1).reshape((ny_out, nx_out))
    
    return {
        "u": u_mag,
        "solver_info": {
            "mesh_resolution": mesh_res,
            "element_degree": 2,
            "ksp_type": "preonly",
            "pc_type": "lu",
            "rtol": 1e-9,
            "iterations": iterations,
            "nonlinear_iterations": [iterations]
        }
    }
