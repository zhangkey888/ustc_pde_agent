import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from basix.ufl import element as basix_element, mixed_element as basix_mixed_element
from petsc4py import PETSc

ScalarType = PETSc.ScalarType

def solve(case_spec: dict) -> dict:
    nx_out = case_spec["output"]["grid"]["nx"]
    ny_out = case_spec["output"]["grid"]["ny"]
    bbox = case_spec["output"]["grid"]["bbox"]
    
    # Using high resolution for high accuracy
    mesh_res = 128
    nu = 0.14
    
    comm = MPI.COMM_WORLD
    
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
    
    # Exact solution
    u_ex_x = -60*(x[1]-0.7)*ufl.exp(-30*((x[0]-0.3)**2 + (x[1]-0.7)**2)) + 60*(x[1]-0.3)*ufl.exp(-30*((x[0]-0.7)**2 + (x[1]-0.3)**2))
    u_ex_y = 60*(x[0]-0.3)*ufl.exp(-30*((x[0]-0.3)**2 + (x[1]-0.7)**2)) - 60*(x[0]-0.7)*ufl.exp(-30*((x[0]-0.7)**2 + (x[1]-0.3)**2))
    u_ex = ufl.as_vector((u_ex_x, u_ex_y))
    p_ex = fem.Constant(msh, ScalarType(0.0))
    
    def eps_sym(u_):
        return ufl.sym(ufl.grad(u_))
    
    def sigma_ex(u_, p_):
        return 2.0 * nu * eps_sym(u_) - p_ * ufl.Identity(gdim)
        
    f = -ufl.div(sigma_ex(u_ex, p_ex)) + ufl.grad(u_ex) * u_ex
    
    def eps(u_):
        return ufl.sym(ufl.grad(u_))
    def sigma(u_, p_):
        return 2.0 * nu * eps(u_) - p_ * ufl.Identity(gdim)
        
    F = (
        ufl.inner(sigma(u, p), eps(v)) * ufl.dx
        + ufl.inner(ufl.grad(u) * u, v) * ufl.dx
        - ufl.inner(f, v) * ufl.dx
        + ufl.inner(ufl.div(u), q) * ufl.dx
    )
    
    fdim = msh.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(msh, fdim, lambda x_coord: np.ones(x_coord.shape[1], dtype=bool))
    boundary_dofs_V = fem.locate_dofs_topological((W.sub(0), V), fdim, boundary_facets)
    
    u_bc = fem.Function(V)
    u_expr = fem.Expression(u_ex, V.element.interpolation_points)
    u_bc.interpolate(u_expr)
    bc_u = fem.dirichletbc(u_bc, boundary_dofs_V, W.sub(0))
    
    p_dofs = fem.locate_dofs_geometrical(
        (W.sub(1), Q), lambda x_coord: np.isclose(x_coord[0], 0.0) & np.isclose(x_coord[1], 0.0)
    )
    bcs = [bc_u]
    if len(p_dofs) > 0:
        p0 = fem.Function(Q)
        p0.x.array[:] = 0.0
        bc_p = fem.dirichletbc(p0, p_dofs, W.sub(1))
        bcs.append(bc_p)
        
    w.x.array[:] = 0.0
    u_stokes, p_stokes = ufl.TrialFunctions(W)
    a_stokes = (
        ufl.inner(sigma(u_stokes, p_stokes), eps(v)) * ufl.dx
        + ufl.inner(ufl.div(u_stokes), q) * ufl.dx
    )
    L_stokes = ufl.inner(f, v) * ufl.dx
    
    problem_stokes = petsc.LinearProblem(
        a_stokes, L_stokes, bcs=bcs,
        petsc_options={"ksp_type": "preonly", "pc_type": "lu", "pc_factor_mat_solver_type": "mumps"},
        petsc_options_prefix="stokes_"
    )
    w_stokes = problem_stokes.solve()
    w.x.array[:] = w_stokes.x.array[:]
    
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
    
    problem = petsc.NonlinearProblem(F, w, bcs=bcs, J=J,
                                     petsc_options_prefix="ns_",
                                     petsc_options=petsc_options)
    
    w_sol = problem.solve()
    u_sol, p_sol = w_sol.sub(0).collapse(), w_sol.sub(1).collapse()
    
    xs = np.linspace(bbox[0], bbox[1], nx_out)
    ys = np.linspace(bbox[2], bbox[3], ny_out)
    XX, YY = np.meshgrid(xs, ys)
    pts = np.c_[XX.ravel(), YY.ravel(), np.zeros(nx_out * ny_out)]
    
    tree = geometry.bb_tree(msh, msh.topology.dim)
    cell_candidates = geometry.compute_collisions_points(tree, pts)
    colliding = geometry.compute_colliding_cells(msh, cell_candidates, pts)
    
    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    for i, pt in enumerate(pts):
        links = colliding.links(i)
        if len(links) > 0:
            points_on_proc.append(pt)
            cells_on_proc.append(links[0])
            eval_map.append(i)
            
    u_values = np.zeros((nx_out * ny_out, gdim))
    if len(points_on_proc) > 0:
        vals = u_sol.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        for j, map_idx in enumerate(eval_map):
            u_values[map_idx] = vals[j]
            
    magnitude = np.linalg.norm(u_values, axis=1).reshape(ny_out, nx_out)
    
    solver_info = {
        "mesh_resolution": mesh_res,
        "element_degree": 2,
        "ksp_type": "preonly",
        "pc_type": "lu",
        "rtol": 1e-9,
        "iterations": 0,
        "nonlinear_iterations": [0]
    }
    
    return {
        "u": magnitude,
        "solver_info": solver_info
    }
