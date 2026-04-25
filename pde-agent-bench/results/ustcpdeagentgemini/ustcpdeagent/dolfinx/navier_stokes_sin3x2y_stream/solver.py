import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from basix.ufl import element as basix_element, mixed_element as basix_mixed_element
from petsc4py import PETSc
import time

def solve(case_spec: dict) -> dict:
    t0 = time.time()
    
    # Extract output grid
    nx = case_spec["output"]["grid"]["nx"]
    ny = case_spec["output"]["grid"]["ny"]
    bbox = case_spec["output"]["grid"]["bbox"]
    
    # Resolution choice
    mesh_res = 64
    comm = MPI.COMM_WORLD
    
    # 1. Mesh and Spaces
    domain = mesh.create_rectangle(
        comm, [np.array([0.0, 0.0]), np.array([1.0, 1.0])], 
        [mesh_res, mesh_res], cell_type=mesh.CellType.triangle
    )
    gdim = domain.geometry.dim

    vel_el = basix_element("Lagrange", domain.topology.cell_name(), 2, shape=(gdim,))
    pres_el = basix_element("Lagrange", domain.topology.cell_name(), 1)
    W = fem.functionspace(domain, basix_mixed_element([vel_el, pres_el]))
    
    V, V_map = W.sub(0).collapse()
    Q, Q_map = W.sub(1).collapse()

    w = fem.Function(W)
    (u, p) = ufl.split(w)
    (v, q) = ufl.TestFunctions(W)

    # 2. Problem Data
    nu = 0.1
    x = ufl.SpatialCoordinate(domain)
    
    # Exact solution
    u_ex_x = 2 * ufl.pi * ufl.cos(2 * ufl.pi * x[1]) * ufl.sin(3 * ufl.pi * x[0])
    u_ex_y = -3 * ufl.pi * ufl.cos(3 * ufl.pi * x[0]) * ufl.sin(2 * ufl.pi * x[1])
    u_ex = ufl.as_vector((u_ex_x, u_ex_y))
    p_ex = ufl.cos(ufl.pi * x[0]) * ufl.cos(2 * ufl.pi * x[1])
    
    # Body force from Navier-Stokes
    def epsilon(u): return ufl.sym(ufl.grad(u))
    def stress(u, p): return 2.0 * nu * epsilon(u) - p * ufl.Identity(gdim)
    
    f = -ufl.div(stress(u_ex, p_ex)) + ufl.grad(u_ex) * u_ex
    
    # 3. Residual Form
    F = (
        ufl.inner(stress(u, p), epsilon(v)) * ufl.dx
        + ufl.inner(ufl.grad(u) * u, v) * ufl.dx
        - ufl.inner(f, v) * ufl.dx
        + ufl.inner(ufl.div(u), q) * ufl.dx
    )
    J = ufl.derivative(F, w)

    # 4. Boundary Conditions
    # Velocity: exact everywhere on boundary
    fdim = domain.topology.dim - 1
    facets = mesh.locate_entities_boundary(domain, fdim, lambda X: np.ones(X.shape[1], dtype=bool))
    dofs_u = fem.locate_dofs_topological((W.sub(0), V), fdim, facets)
    
    u_bc_func = fem.Function(V)
    u_bc_expr = fem.Expression(u_ex, V.element.interpolation_points)
    u_bc_func.interpolate(u_bc_expr)
    bc_u = fem.dirichletbc(u_bc_func, dofs_u, W.sub(0))
    
    # Pressure Pinning
    p_dofs = fem.locate_dofs_geometrical((W.sub(1), Q), lambda x: np.isclose(x[0], 0.0) & np.isclose(x[1], 0.0))
    bc_p_list = []
    if len(p_dofs) > 0:
        p_pin_func = fem.Function(Q)
        p_pin_expr = fem.Expression(p_ex, Q.element.interpolation_points)
        p_pin_func.interpolate(p_pin_expr)
        bc_p = fem.dirichletbc(p_pin_func, p_dofs, W.sub(1))
        bc_p_list.append(bc_p)

    bcs = [bc_u] + bc_p_list

    # 5. Solver
    petsc_options = {
        "snes_type": "newtonls",
        "snes_rtol": 1e-8,
        "ksp_type": "preonly",
        "pc_type": "lu",
        "pc_factor_mat_solver_type": "mumps",
    }
    
    problem = petsc.NonlinearProblem(F, w, bcs=bcs, J=J, petsc_options_prefix="ns_", petsc_options=petsc_options)
    
    # Solve
    problem.solve()
    w.x.scatter_forward()
    u_sol = w.sub(0).collapse()

    # 6. Evaluation on grid
    xs = np.linspace(bbox[0], bbox[1], nx)
    ys = np.linspace(bbox[2], bbox[3], ny)
    XX, YY = np.meshgrid(xs, ys)
    pts = np.c_[XX.ravel(), YY.ravel(), np.zeros(nx * ny)]

    tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(tree, pts)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, pts)

    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    for i in range(len(pts)):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(pts[i])
            cells_on_proc.append(links[0])
            eval_map.append(i)

    u_out = np.zeros((len(pts), gdim))
    if len(points_on_proc) > 0:
        vals = u_sol.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u_out[eval_map] = vals

    u_mag = np.linalg.norm(u_out, axis=1).reshape((ny, nx))
    
    return {
        "u": u_mag,
        "solver_info": {
            "mesh_resolution": mesh_res,
            "element_degree": 2,
            "ksp_type": petsc_options["ksp_type"],
            "pc_type": petsc_options["pc_type"],
            "rtol": petsc_options["snes_rtol"],
            "nonlinear_iterations": [0]
        }
    }

if __name__ == "__main__":
    case_spec = {
        "output": {
            "grid": {
                "nx": 50,
                "ny": 50,
                "bbox": [0.0, 1.0, 0.0, 1.0]
            }
        }
    }
    res = solve(case_spec)
    print("Shape:", res["u"].shape)
    print("Max magnitude:", np.max(res["u"]))
