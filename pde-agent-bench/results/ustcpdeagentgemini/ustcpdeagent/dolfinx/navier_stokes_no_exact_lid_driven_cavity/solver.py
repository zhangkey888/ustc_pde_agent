import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from basix.ufl import element as basix_element, mixed_element as basix_mixed_element
from petsc4py import PETSc

def solve(case_spec: dict) -> dict:
    nx = case_spec["output"]["grid"]["nx"]
    ny = case_spec["output"]["grid"]["ny"]
    bbox = case_spec["output"]["grid"]["bbox"]

    res = 64
    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_square(comm, res, res, cell_type=mesh.CellType.triangle)

    vel_el = basix_element("Lagrange", domain.topology.cell_name(), 2, shape=(domain.geometry.dim,))
    pres_el = basix_element("Lagrange", domain.topology.cell_name(), 1)
    W = fem.functionspace(domain, basix_mixed_element([vel_el, pres_el]))

    V, _ = W.sub(0).collapse()
    Q, _ = W.sub(1).collapse()

    w = fem.Function(W)
    (u, p) = ufl.split(w)
    (v, q) = ufl.TestFunctions(W)

    nu = 0.08
    f = fem.Constant(domain, PETSc.ScalarType((0.0, 0.0)))

    def eps(u):
        return ufl.sym(ufl.grad(u))

    def sigma(u, p):
        return 2.0 * nu * eps(u) - p * ufl.Identity(domain.geometry.dim)

    F = (
        ufl.inner(sigma(u, p), eps(v)) * ufl.dx
        + ufl.inner(ufl.grad(u) * u, v) * ufl.dx
        - ufl.inner(f, v) * ufl.dx
        + ufl.inner(ufl.div(u), q) * ufl.dx
    )

    fdim = domain.topology.dim - 1
    bcs = []

    # Lid boundary condition (y=1)
    lid_facets = mesh.locate_entities_boundary(domain, fdim, lambda x: np.isclose(x[1], 1.0))
    lid_dofs = fem.locate_dofs_topological((W.sub(0), V), fdim, lid_facets)
    u_lid = fem.Function(V)
    u_lid.interpolate(lambda x: np.vstack((np.ones(x.shape[1]), np.zeros(x.shape[1]))))
    bcs.append(fem.dirichletbc(u_lid, lid_dofs, W.sub(0)))

    # Wall boundary conditions (y=0, x=0, x=1)
    wall_facets = mesh.locate_entities_boundary(domain, fdim, lambda x: np.logical_or(np.isclose(x[1], 0.0), np.logical_or(np.isclose(x[0], 0.0), np.isclose(x[0], 1.0))))
    wall_dofs = fem.locate_dofs_topological((W.sub(0), V), fdim, wall_facets)
    u_wall = fem.Function(V)
    u_wall.x.array[:] = 0.0
    bcs.append(fem.dirichletbc(u_wall, wall_dofs, W.sub(0)))

    # Pressure pin (p=0 at the origin)
    p_dofs = fem.locate_dofs_geometrical((W.sub(1), Q), lambda x: np.logical_and(np.isclose(x[0], 0.0), np.isclose(x[1], 0.0)))
    if len(p_dofs) > 0:
        p_pin = fem.Function(Q)
        p_pin.x.array[:] = 0.0
        bcs.append(fem.dirichletbc(p_pin, p_dofs, W.sub(1)))

    J = ufl.derivative(F, w)

    petsc_options = {
        "snes_type": "newtonls",
        "snes_linesearch_type": "bt",
        "snes_rtol": 1e-8,
        "snes_atol": 1e-10,
        "snes_max_it": 30,
        "ksp_type": "preonly",
        "pc_type": "lu",
        "pc_factor_mat_solver_type": "mumps",
    }

    problem = petsc.NonlinearProblem(F, w, bcs=bcs, J=J, petsc_options_prefix="ns_", petsc_options=petsc_options)

    w_h = problem.solve()
    w.x.scatter_forward()
    u_sol, p_sol = w.sub(0).collapse(), w.sub(1).collapse()

    # Interpolate output
    xs = np.linspace(bbox[0], bbox[1], nx)
    ys = np.linspace(bbox[2], bbox[3], ny)
    XX, YY = np.meshgrid(xs, ys)
    pts = np.c_[XX.ravel(), YY.ravel(), np.zeros(nx * ny)]

    tree = geometry.bb_tree(domain, domain.topology.dim)
    # pts is shape (N, 3) which is expected by compute_collisions_points
    cell_candidates = geometry.compute_collisions_points(tree, pts)
    colliding = geometry.compute_colliding_cells(domain, cell_candidates, pts)

    points_on_proc = []
    cells = []
    eval_map = []
    for i in range(pts.shape[0]):
        links = colliding.links(i)
        if len(links) > 0:
            points_on_proc.append(pts[i])
            cells.append(links[0])
            eval_map.append(i)

    u_vals = np.zeros((pts.shape[0], domain.geometry.dim))
    if len(points_on_proc) > 0:
        v_eval = u_sol.eval(np.array(points_on_proc), np.array(cells, dtype=np.int32))
        u_vals[eval_map] = v_eval

    magnitude = np.linalg.norm(u_vals, axis=1).reshape(ny, nx)

    solver_info = {
        "mesh_resolution": res,
        "element_degree": 2,
        "ksp_type": "preonly",
        "pc_type": "lu",
        "rtol": 1e-8,
        "nonlinear_iterations": [0]
    }

    return {"u": magnitude, "solver_info": solver_info}

if __name__ == "__main__":
    case_spec = {
        "output": {
            "grid": {
                "nx": 100,
                "ny": 100,
                "bbox": [0, 1, 0, 1]
            }
        }
    }
    result = solve(case_spec)
    print(f"Solved. Output shape: {result['u'].shape}")

