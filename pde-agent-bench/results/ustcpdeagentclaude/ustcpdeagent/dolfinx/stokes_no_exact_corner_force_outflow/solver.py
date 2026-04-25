import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem.petsc import LinearProblem
from basix.ufl import element as basix_element, mixed_element as basix_mixed_element
import ufl
from petsc4py import PETSc


def solve(case_spec: dict) -> dict:
    grid = case_spec["output"]["grid"]
    nx_out = grid["nx"]
    ny_out = grid["ny"]
    bbox = grid["bbox"]

    N = 256
    msh = mesh.create_unit_square(MPI.COMM_WORLD, N, N, cell_type=mesh.CellType.triangle)
    gdim = msh.geometry.dim

    vel_el = basix_element("Lagrange", msh.topology.cell_name(), 2, shape=(gdim,))
    pres_el = basix_element("Lagrange", msh.topology.cell_name(), 1)
    W = fem.functionspace(msh, basix_mixed_element([vel_el, pres_el]))
    V, _ = W.sub(0).collapse()
    Q, _ = W.sub(1).collapse()

    nu = 0.1
    (u, p) = ufl.TrialFunctions(W)
    (v, q) = ufl.TestFunctions(W)

    x = ufl.SpatialCoordinate(msh)
    fx = 3 * ufl.exp(-50 * ((x[0] - 0.15) ** 2 + (x[1] - 0.15) ** 2))
    fy = 3 * ufl.exp(-50 * ((x[0] - 0.15) ** 2 + (x[1] - 0.15) ** 2))
    f = ufl.as_vector([fx, fy])

    a = (nu * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
         - p * ufl.div(v) * ufl.dx
         + ufl.div(u) * q * ufl.dx)
    L = ufl.inner(f, v) * ufl.dx

    fdim = msh.topology.dim - 1

    # BCs: u=0 on x=0, y=0, y=1 (outflow at x=1 left free -> natural BC)
    def on_walls(x):
        return np.isclose(x[0], 0.0) | np.isclose(x[1], 0.0) | np.isclose(x[1], 1.0)

    wall_facets = mesh.locate_entities_boundary(msh, fdim, on_walls)
    u0 = fem.Function(V)
    u0.x.array[:] = 0.0
    dofs_u = fem.locate_dofs_topological((W.sub(0), V), fdim, wall_facets)
    bc_u = fem.dirichletbc(u0, dofs_u, W.sub(0))

    bcs = [bc_u]
    # Outflow x=1 provides natural BC, so pressure is determined - no pinning needed.

    problem = LinearProblem(
        a, L, bcs=bcs,
        petsc_options={"ksp_type": "preonly", "pc_type": "lu",
                       "pc_factor_mat_solver_type": "mumps"},
        petsc_options_prefix="stokes_corner_"
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

    u_vals_flat = np.zeros((nx_out * ny_out, gdim))
    if points_on_proc:
        vals = u_h.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        for k, idx in enumerate(eval_map):
            u_vals_flat[idx] = vals[k]

    mag = np.linalg.norm(u_vals_flat, axis=1).reshape(ny_out, nx_out)

    try:
        iters = problem.solver.getIterationNumber()
    except Exception:
        iters = 1

    return {
        "u": mag,
        "solver_info": {
            "mesh_resolution": N,
            "element_degree": 2,
            "ksp_type": "preonly",
            "pc_type": "lu",
            "rtol": 1e-10,
            "iterations": int(iters),
        },
    }


if __name__ == "__main__":
    case_spec = {
        "output": {"grid": {"nx": 64, "ny": 64, "bbox": [0, 1, 0, 1]}}
    }
    import time
    t0 = time.time()
    res = solve(case_spec)
    print("time:", time.time() - t0)
    print("shape:", res["u"].shape, "max:", res["u"].max(), "min:", res["u"].min())
