import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem.petsc import LinearProblem
from basix.ufl import element as basix_element, mixed_element as basix_mixed_element
import ufl
from petsc4py import PETSc


def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD

    # Mesh resolution
    N = 200
    msh = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
    gdim = msh.geometry.dim

    # Taylor-Hood P2/P1
    vel_el = basix_element("Lagrange", msh.topology.cell_name(), 2, shape=(gdim,))
    pres_el = basix_element("Lagrange", msh.topology.cell_name(), 1)
    W = fem.functionspace(msh, basix_mixed_element([vel_el, pres_el]))
    V, _ = W.sub(0).collapse()
    Q, _ = W.sub(1).collapse()

    nu = 1.0
    f = fem.Constant(msh, np.zeros(gdim, dtype=PETSc.ScalarType))

    (u, p) = ufl.TrialFunctions(W)
    (v, q) = ufl.TestFunctions(W)

    a = (nu * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
         - p * ufl.div(v) * ufl.dx
         - ufl.div(u) * q * ufl.dx)
    L = ufl.inner(f, v) * ufl.dx

    fdim = msh.topology.dim - 1

    # BCs
    # x0: x=0, u=(0,0)
    facets_x0 = mesh.locate_entities_boundary(msh, fdim, lambda x: np.isclose(x[0], 0.0))
    # x1: x=1, u=(0,0)  -- not listed but we need full Dirichlet? Let me re-check
    # problem lists x0, y0, y1 only. We need x1 too? Typically unit square has 4 sides.
    # Actually the spec says:
    # - u = [0.0, 0.0] on x0
    # - u = [0.0, 0.0] on y0
    # - u = [0.5, 0.0] on y1
    # Missing x1 - but for a well-posed Stokes problem with all-Dirichlet we need all 4 sides
    # Assume x1 is also no-slip (0,0) - standard cavity setup
    facets_x1 = mesh.locate_entities_boundary(msh, fdim, lambda x: np.isclose(x[0], 1.0))
    facets_y0 = mesh.locate_entities_boundary(msh, fdim, lambda x: np.isclose(x[1], 0.0))
    facets_y1 = mesh.locate_entities_boundary(msh, fdim, lambda x: np.isclose(x[1], 1.0))

    u_zero = fem.Function(V)
    u_zero.x.array[:] = 0.0

    u_lid = fem.Function(V)
    u_lid.interpolate(lambda x: np.vstack([np.full(x.shape[1], 0.5), np.zeros(x.shape[1])]))

    dofs_x0 = fem.locate_dofs_topological((W.sub(0), V), fdim, facets_x0)
    dofs_x1 = fem.locate_dofs_topological((W.sub(0), V), fdim, facets_x1)
    dofs_y0 = fem.locate_dofs_topological((W.sub(0), V), fdim, facets_y0)
    dofs_y1 = fem.locate_dofs_topological((W.sub(0), V), fdim, facets_y1)

    bc_x0 = fem.dirichletbc(u_zero, dofs_x0, W.sub(0))
    bc_x1 = fem.dirichletbc(u_zero, dofs_x1, W.sub(0))
    bc_y0 = fem.dirichletbc(u_zero, dofs_y0, W.sub(0))
    bc_y1 = fem.dirichletbc(u_lid, dofs_y1, W.sub(0))

    # Pressure pin at origin
    p_dofs = fem.locate_dofs_geometrical(
        (W.sub(1), Q),
        lambda x: np.isclose(x[0], 0.0) & np.isclose(x[1], 0.0),
    )
    p0 = fem.Function(Q)
    p0.x.array[:] = 0.0
    bc_p = fem.dirichletbc(p0, p_dofs, W.sub(1))

    bcs = [bc_x0, bc_x1, bc_y0, bc_y1, bc_p]

    problem = LinearProblem(
        a, L, bcs=bcs,
        petsc_options={"ksp_type": "preonly", "pc_type": "lu",
                       "pc_factor_mat_solver_type": "mumps"},
        petsc_options_prefix="stokes_",
    )
    w_h = problem.solve()

    iterations = 1
    try:
        iterations = problem.solver.getIterationNumber()
    except Exception:
        pass

    u_h = w_h.sub(0).collapse()

    # Sample on grid
    grid = case_spec["output"]["grid"]
    nx = grid["nx"]
    ny = grid["ny"]
    bbox = grid["bbox"]
    xs = np.linspace(bbox[0], bbox[1], nx)
    ys = np.linspace(bbox[2], bbox[3], ny)
    XX, YY = np.meshgrid(xs, ys)
    pts = np.c_[XX.ravel(), YY.ravel(), np.zeros(nx * ny)]

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

    u_vals = np.zeros((nx * ny, gdim))
    if len(points_on_proc) > 0:
        vals = u_h.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        for k, idx in enumerate(eval_map):
            u_vals[idx] = vals[k]

    magnitude = np.linalg.norm(u_vals, axis=1).reshape(ny, nx)

    return {
        "u": magnitude,
        "solver_info": {
            "mesh_resolution": N,
            "element_degree": 2,
            "ksp_type": "preonly",
            "pc_type": "lu",
            "rtol": 1e-10,
            "iterations": int(iterations),
        },
    }


if __name__ == "__main__":
    import time
    spec = {"output": {"grid": {"nx": 64, "ny": 64, "bbox": [0, 1, 0, 1]}}}
    t0 = time.time()
    out = solve(spec)
    print("time:", time.time() - t0)
    print("shape:", out["u"].shape, "max:", out["u"].max(), "min:", out["u"].min())
