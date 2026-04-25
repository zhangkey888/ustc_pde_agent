import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
from basix.ufl import element as basix_element, mixed_element as basix_mixed_element
import ufl
from petsc4py import PETSc


def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD

    grid = case_spec["output"]["grid"]
    nx_out = grid["nx"]
    ny_out = grid["ny"]
    bbox = grid["bbox"]

    nu_val = 0.05
    N = 160
    msh = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
    gdim = msh.geometry.dim

    vel_el = basix_element("Lagrange", msh.topology.cell_name(), 2, shape=(gdim,))
    pres_el = basix_element("Lagrange", msh.topology.cell_name(), 1)
    W = fem.functionspace(msh, basix_mixed_element([vel_el, pres_el]))
    V, _ = W.sub(0).collapse()
    Q, _ = W.sub(1).collapse()

    (u, p) = ufl.TrialFunctions(W)
    (v, q) = ufl.TestFunctions(W)

    nu = fem.Constant(msh, PETSc.ScalarType(nu_val))
    f = fem.Constant(msh, PETSc.ScalarType((0.0, 0.0)))

    a = (nu * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
         - p * ufl.div(v) * ufl.dx
         + ufl.div(u) * q * ufl.dx)
    L = ufl.inner(f, v) * ufl.dx

    fdim = msh.topology.dim - 1

    # Inlet x=0: u = (4y(1-y), 0)
    inlet_facets = mesh.locate_entities_boundary(msh, fdim, lambda x: np.isclose(x[0], 0.0))
    u_inlet = fem.Function(V)
    def inlet_profile(x):
        vals = np.zeros((gdim, x.shape[1]))
        vals[0] = 4.0 * x[1] * (1.0 - x[1])
        return vals
    u_inlet.interpolate(inlet_profile)
    dofs_inlet = fem.locate_dofs_topological((W.sub(0), V), fdim, inlet_facets)
    bc_inlet = fem.dirichletbc(u_inlet, dofs_inlet, W.sub(0))

    # y=0 no-slip
    bot_facets = mesh.locate_entities_boundary(msh, fdim, lambda x: np.isclose(x[1], 0.0))
    u_zero = fem.Function(V)
    u_zero.x.array[:] = 0.0
    dofs_bot = fem.locate_dofs_topological((W.sub(0), V), fdim, bot_facets)
    bc_bot = fem.dirichletbc(u_zero, dofs_bot, W.sub(0))

    # y=1 no-slip
    top_facets = mesh.locate_entities_boundary(msh, fdim, lambda x: np.isclose(x[1], 1.0))
    dofs_top = fem.locate_dofs_topological((W.sub(0), V), fdim, top_facets)
    bc_top = fem.dirichletbc(u_zero, dofs_top, W.sub(0))

    bcs = [bc_inlet, bc_bot, bc_top]

    # Pressure pin at (1, 0.5) to give unique pressure; outflow has natural BC
    # Actually since outflow is free, pressure is determined (natural do-nothing BC at x=1)
    # So no pinning needed. But to be safe, we don't pin.

    problem = petsc.LinearProblem(
        a, L, bcs=bcs,
        petsc_options={"ksp_type": "preonly", "pc_type": "lu",
                       "pc_factor_mat_solver_type": "mumps"},
        petsc_options_prefix="stokes_ch_"
    )
    w_h = problem.solve()

    u_h = w_h.sub(0).collapse()

    # Sample on grid
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
    for i in range(pts.shape[0]):
        links = colliding.links(i)
        if len(links) > 0:
            points_on_proc.append(pts[i])
            cells_on_proc.append(links[0])
            eval_map.append(i)

    u_vals = np.zeros((pts.shape[0], gdim))
    if len(points_on_proc) > 0:
        vals = u_h.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        for k, idx in enumerate(eval_map):
            u_vals[idx] = vals[k]

    magnitude = np.linalg.norm(u_vals, axis=1).reshape(ny_out, nx_out)

    return {
        "u": magnitude,
        "solver_info": {
            "mesh_resolution": N,
            "element_degree": 2,
            "ksp_type": "preonly",
            "pc_type": "lu",
            "rtol": 1e-12,
            "iterations": 1,
        }
    }


if __name__ == "__main__":
    import time
    case = {"output": {"grid": {"nx": 64, "ny": 64, "bbox": [0, 1, 0, 1]}}}
    t0 = time.time()
    res = solve(case)
    print(f"Time: {time.time()-t0:.2f}s, shape: {res['u'].shape}, max: {res['u'].max():.4f}")
