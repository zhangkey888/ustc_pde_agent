import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
from basix.ufl import element as basix_element, mixed_element as basix_mixed_element
import ufl
from petsc4py import PETSc


def solve(case_spec: dict) -> dict:
    grid = case_spec["output"]["grid"]
    nx_out = grid["nx"]
    ny_out = grid["ny"]
    bbox = grid["bbox"]

    N = 64
    msh = mesh.create_unit_square(MPI.COMM_WORLD, N, N, cell_type=mesh.CellType.triangle)
    gdim = msh.geometry.dim

    vel_el = basix_element("Lagrange", msh.topology.cell_name(), 2, shape=(gdim,))
    pres_el = basix_element("Lagrange", msh.topology.cell_name(), 1)
    W = fem.functionspace(msh, basix_mixed_element([vel_el, pres_el]))
    V, _ = W.sub(0).collapse()
    Q, _ = W.sub(1).collapse()

    nu = 1.0
    (u, p) = ufl.TrialFunctions(W)
    (v, q) = ufl.TestFunctions(W)
    f = fem.Constant(msh, np.zeros(gdim))

    a = (nu * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
         - p * ufl.div(v) * ufl.dx
         + ufl.div(u) * q * ufl.dx)
    L = ufl.inner(f, v) * ufl.dx

    fdim = msh.topology.dim - 1

    # Inflow x=0: parabolic
    def inflow(x):
        return np.isclose(x[0], 0.0)

    def walls(x):
        return np.isclose(x[1], 0.0) | np.isclose(x[1], 1.0)

    inflow_facets = mesh.locate_entities_boundary(msh, fdim, inflow)
    wall_facets = mesh.locate_entities_boundary(msh, fdim, walls)

    u_in = fem.Function(V)
    u_in.interpolate(lambda x: np.vstack([4.0 * x[1] * (1.0 - x[1]), np.zeros(x.shape[1])]))
    bc_in = fem.dirichletbc(u_in,
                             fem.locate_dofs_topological((W.sub(0), V), fdim, inflow_facets),
                             W.sub(0))

    u_wall = fem.Function(V)
    u_wall.x.array[:] = 0.0
    bc_wall = fem.dirichletbc(u_wall,
                               fem.locate_dofs_topological((W.sub(0), V), fdim, wall_facets),
                               W.sub(0))

    # Outflow x=1: natural BC (do nothing) - but need to pin pressure since with do-nothing
    # it should be OK. But for Stokes with natural BC at outflow, pressure is determined.
    # Actually with do-nothing BC (sigma·n = 0), pressure nullspace is removed.
    # However, we used grad(u):grad(v) form, not sym-gradient, so natural BC is different.
    # Let's just pin pressure at outflow corner to be safe. Actually, using the non-symmetric
    # form with do-nothing: pressure IS determined up to a constant unless we have full
    # Dirichlet on velocity. Since x=1 is free, pressure should be fine. Let's not pin.

    bcs = [bc_in, bc_wall]

    problem = petsc.LinearProblem(
        a, L, bcs=bcs,
        petsc_options={"ksp_type": "preonly", "pc_type": "lu",
                       "pc_factor_mat_solver_type": "mumps"},
        petsc_options_prefix="stokes_"
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

    mag = np.full(nx_out * ny_out, np.nan)
    points_on_proc = []
    cells = []
    idx_map = []
    for i in range(pts.shape[0]):
        links = colliding.links(i)
        if len(links) > 0:
            points_on_proc.append(pts[i])
            cells.append(links[0])
            idx_map.append(i)

    if len(points_on_proc) > 0:
        vals = u_h.eval(np.array(points_on_proc), np.array(cells, dtype=np.int32))
        m = np.linalg.norm(vals, axis=1)
        for k, idx in enumerate(idx_map):
            mag[idx] = m[k]

    u_grid = mag.reshape(ny_out, nx_out)

    return {
        "u": u_grid,
        "solver_info": {
            "mesh_resolution": N,
            "element_degree": 2,
            "ksp_type": "preonly",
            "pc_type": "lu",
            "rtol": 1e-12,
        }
    }


if __name__ == "__main__":
    spec = {"output": {"grid": {"nx": 64, "ny": 64, "bbox": [0, 1, 0, 1]}}}
    import time
    t0 = time.time()
    result = solve(spec)
    print("Time:", time.time() - t0)
    print("Shape:", result["u"].shape)
    print("Max u mag:", np.nanmax(result["u"]))
    print("Min u mag:", np.nanmin(result["u"]))
    # Expected: inflow max is 1.0 at y=0.5, fully developed channel flow
    # Exact solution for Poiseuille: u = 4y(1-y), v = 0, p = -8*nu*x + const
    # Check at an interior point
    y_mid = 0.5
    # u should be ~1.0 everywhere at y=0.5
    print("At y=0.5:", result["u"][result["u"].shape[0]//2, :])
