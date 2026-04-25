import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem.petsc import LinearProblem
from basix.ufl import element as basix_element, mixed_element as basix_mixed_element
import ufl
from petsc4py import PETSc


def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD
    grid = case_spec["output"]["grid"]
    nx_out = grid["nx"]
    ny_out = grid["ny"]
    bbox = grid["bbox"]

    N = 128
    msh = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
    gdim = msh.geometry.dim

    deg_u = 2
    deg_p = 1
    vel_el = basix_element("Lagrange", msh.topology.cell_name(), deg_u, shape=(gdim,))
    pres_el = basix_element("Lagrange", msh.topology.cell_name(), deg_p)
    W = fem.functionspace(msh, basix_mixed_element([vel_el, pres_el]))

    V, _ = W.sub(0).collapse()
    Q, _ = W.sub(1).collapse()

    nu = 0.8

    (u, p) = ufl.TrialFunctions(W)
    (v, q) = ufl.TestFunctions(W)

    f = fem.Constant(msh, PETSc.ScalarType((0.0, 0.0)))

    a = (nu * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
         - p * ufl.div(v) * ufl.dx
         - ufl.div(u) * q * ufl.dx)
    L = ufl.inner(f, v) * ufl.dx

    fdim = msh.topology.dim - 1

    # Inflow x=0: u = (2y(1-y), 2y(1-y))
    x0_facets = mesh.locate_entities_boundary(msh, fdim, lambda x: np.isclose(x[0], 0.0))
    u_in = fem.Function(V)
    def inflow(x):
        val = 2.0 * x[1] * (1.0 - x[1])
        return np.vstack([val, val])
    u_in.interpolate(inflow)
    dofs_in = fem.locate_dofs_topological((W.sub(0), V), fdim, x0_facets)
    bc_in = fem.dirichletbc(u_in, dofs_in, W.sub(0))

    # y=0 no-slip
    y0_facets = mesh.locate_entities_boundary(msh, fdim, lambda x: np.isclose(x[1], 0.0))
    u_y0 = fem.Function(V)
    u_y0.x.array[:] = 0.0
    dofs_y0 = fem.locate_dofs_topological((W.sub(0), V), fdim, y0_facets)
    bc_y0 = fem.dirichletbc(u_y0, dofs_y0, W.sub(0))

    # y=1 no-slip
    y1_facets = mesh.locate_entities_boundary(msh, fdim, lambda x: np.isclose(x[1], 1.0))
    u_y1 = fem.Function(V)
    u_y1.x.array[:] = 0.0
    dofs_y1 = fem.locate_dofs_topological((W.sub(0), V), fdim, y1_facets)
    bc_y1 = fem.dirichletbc(u_y1, dofs_y1, W.sub(0))

    bcs = [bc_in, bc_y0, bc_y1]
    # x=1 is natural outflow (do-nothing): pressure nullspace is fixed by do-nothing bc

    problem = LinearProblem(
        a, L, bcs=bcs,
        petsc_options={"ksp_type": "preonly", "pc_type": "lu",
                       "pc_factor_mat_solver_type": "mumps"},
        petsc_options_prefix="stokes_diag_"
    )
    w_h = problem.solve()
    try:
        its = problem.solver.getIterationNumber()
    except Exception:
        its = 1

    u_h = w_h.sub(0).collapse()

    # Sample onto uniform grid
    xs = np.linspace(bbox[0], bbox[1], nx_out)
    ys = np.linspace(bbox[2], bbox[3], ny_out)
    XX, YY = np.meshgrid(xs, ys)
    pts = np.c_[XX.ravel(), YY.ravel(), np.zeros(nx_out * ny_out)]

    tree = geometry.bb_tree(msh, msh.topology.dim)
    cell_candidates = geometry.compute_collisions_points(tree, pts)
    colliding = geometry.compute_colliding_cells(msh, cell_candidates, pts)

    cells = []
    pts_on = []
    idx_map = []
    for i in range(pts.shape[0]):
        links = colliding.links(i)
        if len(links) > 0:
            cells.append(links[0])
            pts_on.append(pts[i])
            idx_map.append(i)

    u_vals = np.zeros((pts.shape[0], 2))
    if len(pts_on) > 0:
        vals = u_h.eval(np.array(pts_on), np.array(cells, dtype=np.int32))
        u_vals[idx_map] = vals

    mag = np.linalg.norm(u_vals, axis=1).reshape(ny_out, nx_out)

    return {
        "u": mag,
        "solver_info": {
            "mesh_resolution": N,
            "element_degree": deg_u,
            "ksp_type": "preonly",
            "pc_type": "lu",
            "rtol": 1e-10,
            "iterations": int(its),
        }
    }


if __name__ == "__main__":
    import time
    spec = {"output": {"grid": {"nx": 64, "ny": 64, "bbox": [0.0, 1.0, 0.0, 1.0]}}}
    t0 = time.time()
    res = solve(spec)
    print("time:", time.time() - t0)
    print("shape:", res["u"].shape)
    print("min/max:", res["u"].min(), res["u"].max())
    print("mean:", res["u"].mean())
