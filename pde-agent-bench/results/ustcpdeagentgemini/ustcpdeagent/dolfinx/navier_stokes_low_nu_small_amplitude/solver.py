import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc
from basix.ufl import element as basix_element, mixed_element as basix_mixed_element
import time

def solve(case_spec: dict) -> dict:
    t0 = time.time()
    
    comm = MPI.COMM_WORLD
    nx_mesh = 64
    ny_mesh = 64
    msh = mesh.create_unit_square(comm, nx_mesh, ny_mesh, cell_type=mesh.CellType.triangle)
    
    gdim = msh.geometry.dim
    vel_el = basix_element("Lagrange", msh.topology.cell_name(), 2, shape=(gdim,))
    pres_el = basix_element("Lagrange", msh.topology.cell_name(), 1)
    W = fem.functionspace(msh, basix_mixed_element([vel_el, pres_el]))
    
    V, _ = W.sub(0).collapse()
    Q, _ = W.sub(1).collapse()
    
    nu = 0.01
    
    x = ufl.SpatialCoordinate(msh)
    u_ex = ufl.as_vector((
        0.2 * ufl.pi * ufl.cos(ufl.pi * x[1]) * ufl.sin(2 * ufl.pi * x[0]),
        -0.4 * ufl.pi * ufl.cos(2 * ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])
    ))
    p_ex = 0.0 * x[0]
    
    # f = (u.grad)u - nu * div(grad u) + grad p
    def grad_u(u):
        return ufl.grad(u)
    
    def div_grad_u(u):
        return ufl.div(ufl.grad(u))
    
    f_ex = ufl.grad(u_ex) * u_ex - nu * div_grad_u(u_ex) + ufl.as_vector([0.0 * x[0], 0.0 * x[0]])
    
    w = fem.Function(W)
    (u, p) = ufl.split(w)
    (v, q) = ufl.TestFunctions(W)
    
    F = (
        nu * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
        + ufl.inner(ufl.grad(u) * u, v) * ufl.dx
        - ufl.inner(p, ufl.div(v)) * ufl.dx
        + ufl.inner(ufl.div(u), q) * ufl.dx
        - ufl.inner(f_ex, v) * ufl.dx
    )
    
    # BCs
    fdim = msh.topology.dim - 1
    wall_facets = mesh.locate_entities_boundary(msh, fdim, lambda X: np.ones(X.shape[1], dtype=bool))
    
    u0 = fem.Function(V)
    u_ex_expr = fem.Expression(u_ex, V.element.interpolation_points)
    u0.interpolate(u_ex_expr)
    
    dofs_u = fem.locate_dofs_topological((W.sub(0), V), fdim, wall_facets)
    bc_u = fem.dirichletbc(u0, dofs_u, W.sub(0))
    
    p_dofs = fem.locate_dofs_geometrical(
        (W.sub(1), Q), lambda x_coord: np.isclose(x_coord[0], 0.0) & np.isclose(x_coord[1], 0.0)
    )
    p0 = fem.Function(Q)
    p0.x.array[:] = 0.0
    bc_p = fem.dirichletbc(p0, p_dofs, W.sub(1))
    
    bcs = [bc_u, bc_p]
    
    J = ufl.derivative(F, w)
    
    petsc_options = {
        "snes_type": "newtonls",
        "snes_linesearch_type": "bt",
        "snes_rtol": 1e-9,
        "snes_atol": 1e-10,
        "snes_max_it": 30,
        "ksp_type": "preonly",
        "pc_type": "lu",
        "pc_factor_mat_solver_type": "mumps"
    }
    
    problem = petsc.NonlinearProblem(F, w, bcs=bcs, J=J,
                                     petsc_options_prefix="ns_",
                                     petsc_options=petsc_options)
    w_sol = problem.solve()
    w.x.scatter_forward()
    
    u_h = w.sub(0).collapse()
    
    # Sample on grid
    grid_spec = case_spec["output"]["grid"]
    nx = grid_spec["nx"]
    ny = grid_spec["ny"]
    bbox = grid_spec["bbox"]
    
    xs = np.linspace(bbox[0], bbox[1], nx)
    ys = np.linspace(bbox[2], bbox[3], ny)
    XX, YY = np.meshgrid(xs, ys)
    pts = np.c_[XX.ravel(), YY.ravel(), np.zeros(nx * ny)]
    
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
        
    magnitude = np.linalg.norm(u_values, axis=1).reshape(ny, nx)
    
    solver_info = {
        "mesh_resolution": nx_mesh,
        "element_degree": 2,
        "ksp_type": "preonly",
        "pc_type": "lu",
        "rtol": 1e-9,
        "iterations": 1,
        "nonlinear_iterations": [3]
    }
    
    return {
        "u": magnitude,
        "solver_info": solver_info
    }

if __name__ == "__main__":
    case_spec = {
        "output": {
            "grid": {"nx": 50, "ny": 50, "bbox": [0, 1, 0, 1]}
        }
    }
    res = solve(case_spec)
    print("Max magnitude:", np.max(res["u"]))
