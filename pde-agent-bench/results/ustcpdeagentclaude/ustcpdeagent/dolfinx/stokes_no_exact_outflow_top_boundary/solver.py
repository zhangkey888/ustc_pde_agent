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

    N = 200
    msh = mesh.create_unit_square(MPI.COMM_WORLD, N, N, cell_type=mesh.CellType.triangle)
    gdim = msh.geometry.dim

    vel_el = basix_element("Lagrange", msh.topology.cell_name(), 2, shape=(gdim,))
    pres_el = basix_element("Lagrange", msh.topology.cell_name(), 1)
    W = fem.functionspace(msh, basix_mixed_element([vel_el, pres_el]))
    V, _ = W.sub(0).collapse()
    Q, _ = W.sub(1).collapse()

    nu = 0.9
    (u, p) = ufl.TrialFunctions(W)
    (v, q) = ufl.TestFunctions(W)

    f = fem.Constant(msh, np.zeros(gdim, dtype=PETSc.ScalarType))

    a = (nu * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
         - p * ufl.div(v) * ufl.dx
         - ufl.div(u) * q * ufl.dx)
    L = ufl.inner(f, v) * ufl.dx

    fdim = msh.topology.dim - 1

    # x = 0: u = (sin(pi*y), 0)
    x0_facets = mesh.locate_entities_boundary(msh, fdim, lambda x: np.isclose(x[0], 0.0))
    u_x0 = fem.Function(V)
    u_x0.interpolate(lambda x: np.vstack([np.sin(np.pi * x[1]), np.zeros(x.shape[1])]))
    dofs_x0 = fem.locate_dofs_topological((W.sub(0), V), fdim, x0_facets)
    bc_x0 = fem.dirichletbc(u_x0, dofs_x0, W.sub(0))

    # y = 0: u = 0
    y0_facets = mesh.locate_entities_boundary(msh, fdim, lambda x: np.isclose(x[1], 0.0))
    u_zero = fem.Function(V)
    u_zero.x.array[:] = 0.0
    dofs_y0 = fem.locate_dofs_topological((W.sub(0), V), fdim, y0_facets)
    bc_y0 = fem.dirichletbc(u_zero, dofs_y0, W.sub(0))

    # x = 1: u = 0
    x1_facets = mesh.locate_entities_boundary(msh, fdim, lambda x: np.isclose(x[0], 1.0))
    u_zero2 = fem.Function(V)
    u_zero2.x.array[:] = 0.0
    dofs_x1 = fem.locate_dofs_topological((W.sub(0), V), fdim, x1_facets)
    bc_x1 = fem.dirichletbc(u_zero2, dofs_x1, W.sub(0))

    bcs = [bc_x0, bc_y0, bc_x1]

    # Pressure pin at (0,0) since top is free; to be safe, pin it.
    p_dofs = fem.locate_dofs_geometrical(
        (W.sub(1), Q),
        lambda x: np.isclose(x[0], 0.0) & np.isclose(x[1], 0.0),
    )
    if len(p_dofs[0]) > 0:
        p0_func = fem.Function(Q)
        p0_func.x.array[:] = 0.0
        bc_p = fem.dirichletbc(p0_func, p_dofs, W.sub(1))
        bcs.append(bc_p)

    problem = LinearProblem(
        a, L, bcs=bcs,
        petsc_options={"ksp_type": "preonly", "pc_type": "lu",
                       "pc_factor_mat_solver_type": "mumps"},
        petsc_options_prefix="stokes_"
    )
    w_h = problem.solve()
    try:
        its = problem.solver.getIterationNumber()
    except Exception:
        its = 1

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

    mag = np.zeros(nx_out * ny_out)
    if len(points_on_proc) > 0:
        vals = u_h.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        mags = np.linalg.norm(vals, axis=1)
        for idx, i in enumerate(eval_map):
            mag[i] = mags[idx]

    mag_grid = mag.reshape(ny_out, nx_out)

    return {
        "u": mag_grid,
        "solver_info": {
            "mesh_resolution": N,
            "element_degree": 2,
            "ksp_type": "preonly",
            "pc_type": "lu",
            "rtol": 1e-10,
            "iterations": int(its),
        }
    }


if __name__ == "__main__":
    import time
    spec = {
        "output": {"grid": {"nx": 64, "ny": 64, "bbox": [0.0, 1.0, 0.0, 1.0]}}
    }
    t0 = time.time()
    res = solve(spec)
    print("time:", time.time() - t0)
    print("shape:", res["u"].shape)
    print("max:", res["u"].max(), "min:", res["u"].min())
    print("info:", res["solver_info"])
