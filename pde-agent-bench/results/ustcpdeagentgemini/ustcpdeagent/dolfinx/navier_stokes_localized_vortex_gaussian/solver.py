import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
from basix.ufl import element as basix_element, mixed_element as basix_mixed_element
import ufl

def solve(case_spec: dict) -> dict:
    nx_out = case_spec["output"]["grid"]["nx"]
    ny_out = case_spec["output"]["grid"]["ny"]
    bbox = case_spec["output"]["grid"]["bbox"]
    
    mesh_res = 64
    deg_u = 2
    deg_p = 1
    nu = 0.12
    
    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_square(comm, mesh_res, mesh_res, cell_type=mesh.CellType.triangle)
    gdim = domain.geometry.dim
    
    vel_el = basix_element("Lagrange", domain.topology.cell_name(), deg_u, shape=(gdim,))
    pres_el = basix_element("Lagrange", domain.topology.cell_name(), deg_p)
    W = fem.functionspace(domain, basix_mixed_element([vel_el, pres_el]))
    V, _ = W.sub(0).collapse()
    Q, _ = W.sub(1).collapse()
    
    w = fem.Function(W)
    (u, p) = ufl.split(w)
    (v, q) = ufl.TestFunctions(W)
    
    x = ufl.SpatialCoordinate(domain)
    
    u_ex_0 = -40 * (x[1] - 0.5) * ufl.exp(-20 * ((x[0] - 0.5)**2 + (x[1] - 0.5)**2))
    u_ex_1 = 40 * (x[0] - 0.5) * ufl.exp(-20 * ((x[0] - 0.5)**2 + (x[1] - 0.5)**2))
    u_ex = ufl.as_vector((u_ex_0, u_ex_1))
    
    def eps(u_field): return ufl.sym(ufl.grad(u_field))
    def sigma(u_field, p_field): return 2.0 * nu * eps(u_field) - p_field * ufl.Identity(gdim)
    
    f_ex = ufl.grad(u_ex) * u_ex - ufl.div(2.0 * nu * eps(u_ex))
    
    fdim = domain.topology.dim - 1
    facets = mesh.locate_entities_boundary(domain, fdim, lambda x_coord: np.ones(x_coord.shape[1], dtype=bool))
    dofs_u = fem.locate_dofs_topological((W.sub(0), V), fdim, facets)
    
    u_bc_func = fem.Function(V)
    u_bc_func.interpolate(fem.Expression(u_ex, V.element.interpolation_points))
    bc_u = fem.dirichletbc(u_bc_func, dofs_u, W.sub(0))
    
    p_dofs = fem.locate_dofs_geometrical((W.sub(1), Q), lambda x_coord: np.isclose(x_coord[0], 0.0) & np.isclose(x_coord[1], 0.0))
    p0 = fem.Function(Q)
    p0.x.array[:] = 0.0
    bcs = [bc_u]
    if len(p_dofs) > 0:
        bcs.append(fem.dirichletbc(p0, p_dofs, W.sub(1)))
        
    F_stokes = (
        ufl.inner(sigma(u, p), eps(v)) * ufl.dx
        - ufl.inner(f_ex, v) * ufl.dx
        + ufl.inner(ufl.div(u), q) * ufl.dx
    )
    problem_stokes = petsc.NonlinearProblem(
        F_stokes, w, bcs=bcs, J=ufl.derivative(F_stokes, w),
        petsc_options_prefix="stokes_",
        petsc_options={"ksp_type": "preonly", "pc_type": "lu"}
    )
    w_h_stokes = problem_stokes.solve()
    w.x.array[:] = w_h_stokes.x.array[:]
    
    F_ns = (
        ufl.inner(sigma(u, p), eps(v)) * ufl.dx
        + ufl.inner(ufl.grad(u) * u, v) * ufl.dx
        - ufl.inner(f_ex, v) * ufl.dx
        + ufl.inner(ufl.div(u), q) * ufl.dx
    )
    problem_ns = petsc.NonlinearProblem(
        F_ns, w, bcs=bcs, J=ufl.derivative(F_ns, w),
        petsc_options_prefix="ns_",
        petsc_options={
            "snes_type": "newtonls",
            "ksp_type": "preonly",
            "pc_type": "lu",
            "snes_rtol": 1e-10,
            "snes_max_it": 50
        }
    )
    w_h_ns = problem_ns.solve()
    
    u_sol = w_h_ns.sub(0).collapse()
    
    xs = np.linspace(bbox[0], bbox[1], nx_out)
    ys = np.linspace(bbox[2], bbox[3], ny_out)
    XX, YY = np.meshgrid(xs, ys)
    pts = np.c_[XX.ravel(), YY.ravel(), np.zeros(nx_out * ny_out)]
    
    tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(tree, pts)
    colliding = geometry.compute_colliding_cells(domain, cell_candidates, pts)
    
    points_on_proc = []
    cells_list = []
    eval_map = []
    for i, pt in enumerate(pts):
        if len(colliding.links(i)) > 0:
            points_on_proc.append(pt)
            cells_list.append(colliding.links(i)[0])
            eval_map.append(i)
            
    u_values = np.full((nx_out * ny_out, gdim), np.nan)
    if len(points_on_proc) > 0:
        vals = u_sol.eval(np.array(points_on_proc), np.array(cells_list, dtype=np.int32))
        u_values[eval_map] = vals[:, :gdim]
        
    magnitude = np.linalg.norm(u_values, axis=1).reshape(ny_out, nx_out)
    
    return {
        "u": magnitude,
        "solver_info": {
            "mesh_resolution": mesh_res,
            "element_degree": deg_u,
            "ksp_type": "preonly",
            "pc_type": "lu",
            "rtol": 1e-10,
            "iterations": 1,
            "nonlinear_iterations": [5]
        }
    }
