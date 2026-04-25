import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
from basix.ufl import element as basix_element, mixed_element as basix_mixed_element
import ufl
from petsc4py import PETSc


def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD
    nu_val = 0.4
    N = 64

    msh = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
    gdim = msh.geometry.dim

    deg_u, deg_p = 2, 1
    vel_el = basix_element("Lagrange", msh.topology.cell_name(), deg_u, shape=(gdim,))
    pres_el = basix_element("Lagrange", msh.topology.cell_name(), deg_p)
    W = fem.functionspace(msh, basix_mixed_element([vel_el, pres_el]))
    V, _ = W.sub(0).collapse()

    (u, p) = ufl.TrialFunctions(W)
    (v, q) = ufl.TestFunctions(W)

    nu = fem.Constant(msh, PETSc.ScalarType(nu_val))
    f = fem.Constant(msh, PETSc.ScalarType((1.0, 0.0)))

    a = (nu * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
         - p * ufl.div(v) * ufl.dx
         + ufl.div(u) * q * ufl.dx)
    L = ufl.inner(f, v) * ufl.dx

    fdim = msh.topology.dim - 1
    # No-slip BCs on x=0, y=0, y=1; x=1 is outflow (natural)
    def walls(x):
        return np.isclose(x[0], 0.0) | np.isclose(x[1], 0.0) | np.isclose(x[1], 1.0)

    wall_facets = mesh.locate_entities_boundary(msh, fdim, walls)
    u0 = fem.Function(V)
    u0.x.array[:] = 0.0
    dofs_u = fem.locate_dofs_topological((W.sub(0), V), fdim, wall_facets)
    bc_u = fem.dirichletbc(u0, dofs_u, W.sub(0))

    bcs = [bc_u]

    problem = petsc.LinearProblem(
        a, L, bcs=bcs,
        petsc_options={"ksp_type": "preonly", "pc_type": "lu",
                       "pc_factor_mat_solver_type": "mumps"},
        petsc_options_prefix="stokes_"
    )
    w_h = problem.solve()

    ksp = problem.solver
    iters = ksp.getIterationNumber()

    u_sub = w_h.sub(0).collapse()

    # Sample on grid
    grid = case_spec["output"]["grid"]
    nx, ny = grid["nx"], grid["ny"]
    bbox = grid["bbox"]
    xs = np.linspace(bbox[0], bbox[1], nx)
    ys = np.linspace(bbox[2], bbox[3], ny)
    XX, YY = np.meshgrid(xs, ys)
    pts = np.c_[XX.ravel(), YY.ravel(), np.zeros(nx*ny)]

    tree = geometry.bb_tree(msh, msh.topology.dim)
    cand = geometry.compute_collisions_points(tree, pts)
    colliding = geometry.compute_colliding_cells(msh, cand, pts)

    points_on_proc = []
    cells_on_proc = []
    idx_map = []
    for i in range(pts.shape[0]):
        links = colliding.links(i)
        if len(links) > 0:
            points_on_proc.append(pts[i])
            cells_on_proc.append(links[0])
            idx_map.append(i)

    mag_flat = np.full(nx*ny, np.nan)
    if points_on_proc:
        vals = u_sub.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        mag = np.linalg.norm(vals, axis=1)
        for k, i in enumerate(idx_map):
            mag_flat[i] = mag[k]

    u_grid = mag_flat.reshape(ny, nx)

    return {
        "u": u_grid,
        "solver_info": {
            "mesh_resolution": N,
            "element_degree": deg_u,
            "ksp_type": "preonly",
            "pc_type": "lu",
            "rtol": 1e-10,
            "iterations": int(iters),
        }
    }


if __name__ == "__main__":
    spec = {"output": {"grid": {"nx": 64, "ny": 64, "bbox": [0,1,0,1]}}}
    import time
    t0 = time.time()
    r = solve(spec)
    print("time:", time.time()-t0)
    print("shape:", r["u"].shape, "max:", np.nanmax(r["u"]), "min:", np.nanmin(r["u"]))
    print("info:", r["solver_info"])
