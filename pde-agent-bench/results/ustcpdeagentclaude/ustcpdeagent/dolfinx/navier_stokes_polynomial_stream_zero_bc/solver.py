import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
from basix.ufl import element as basix_element, mixed_element as basix_mixed_element
import ufl
from petsc4py import PETSc


def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD
    N = 32
    msh = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
    gdim = msh.geometry.dim

    vel_el = basix_element("Lagrange", msh.topology.cell_name(), 2, shape=(gdim,))
    pres_el = basix_element("Lagrange", msh.topology.cell_name(), 1)
    W = fem.functionspace(msh, basix_mixed_element([vel_el, pres_el]))

    V, _ = W.sub(0).collapse()
    Q, _ = W.sub(1).collapse()

    nu = 0.25
    x = ufl.SpatialCoordinate(msh)

    # Manufactured solution
    u_ex = ufl.as_vector([
        x[0] * (1 - x[0]) * (1 - 2 * x[1]),
        -x[1] * (1 - x[1]) * (1 - 2 * x[0]),
    ])
    p_ex = x[0] - x[1]

    # f = u·∇u - ν ∇²u + ∇p
    f = ufl.grad(u_ex) * u_ex - nu * ufl.div(ufl.grad(u_ex)) + ufl.grad(p_ex)

    w = fem.Function(W)
    (u, p) = ufl.split(w)
    (v, q) = ufl.TestFunctions(W)

    F = (nu * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
         + ufl.inner(ufl.grad(u) * u, v) * ufl.dx
         - p * ufl.div(v) * ufl.dx
         + ufl.div(u) * q * ufl.dx
         - ufl.inner(f, v) * ufl.dx)

    # Velocity BC: u = u_ex on whole boundary (here = 0 since polynomial vanishes on bdry)
    fdim = msh.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(
        msh, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
    )
    u_bc = fem.Function(V)
    u_bc_expr = fem.Expression(u_ex, V.element.interpolation_points)
    u_bc.interpolate(u_bc_expr)
    dofs_u = fem.locate_dofs_topological((W.sub(0), V), fdim, boundary_facets)
    bc_u = fem.dirichletbc(u_bc, dofs_u, W.sub(0))

    # Pressure pin at (0,0): p_ex(0,0) = 0
    p_dofs = fem.locate_dofs_geometrical(
        (W.sub(1), Q),
        lambda x: np.isclose(x[0], 0.0) & np.isclose(x[1], 0.0),
    )
    p0_func = fem.Function(Q)
    p0_func.x.array[:] = 0.0
    bc_p = fem.dirichletbc(p0_func, p_dofs, W.sub(1))

    bcs = [bc_u, bc_p]

    J = ufl.derivative(F, w)

    petsc_options = {
        "snes_type": "newtonls",
        "snes_linesearch_type": "bt",
        "snes_rtol": 1e-10,
        "snes_atol": 1e-12,
        "snes_max_it": 50,
        "ksp_type": "preonly",
        "pc_type": "lu",
        "pc_factor_mat_solver_type": "mumps",
    }

    problem = petsc.NonlinearProblem(F, w, bcs=bcs, J=J,
                                      petsc_options_prefix="ns_",
                                      petsc_options=petsc_options)
    w_h = problem.solve()
    w.x.scatter_forward()

    u_h = w.sub(0).collapse()

    # Sample on grid
    grid = case_spec["output"]["grid"]
    nx = grid["nx"]
    ny = grid["ny"]
    bbox = grid["bbox"]
    xs = np.linspace(bbox[0], bbox[1], nx)
    ys = np.linspace(bbox[2], bbox[3], ny)
    XX, YY = np.meshgrid(xs, ys)
    pts = np.column_stack([XX.ravel(), YY.ravel(), np.zeros(nx * ny)])

    tree = geometry.bb_tree(msh, msh.topology.dim)
    cell_candidates = geometry.compute_collisions_points(tree, pts)
    colliding = geometry.compute_colliding_cells(msh, cell_candidates, pts)

    points_on_proc = []
    cells = []
    eval_map = []
    for i in range(pts.shape[0]):
        links = colliding.links(i)
        if len(links) > 0:
            points_on_proc.append(pts[i])
            cells.append(links[0])
            eval_map.append(i)

    u_vals = np.zeros((pts.shape[0], gdim))
    if points_on_proc:
        vals = u_h.eval(np.array(points_on_proc), np.array(cells, dtype=np.int32))
        for k, idx in enumerate(eval_map):
            u_vals[idx] = vals[k]
    mag = np.linalg.norm(u_vals, axis=1).reshape(ny, nx)

    return {
        "u": mag,
        "solver_info": {
            "mesh_resolution": N,
            "element_degree": 3,
            "ksp_type": "preonly",
            "pc_type": "lu",
            "rtol": 1e-10,
            "iterations": 0,
            "nonlinear_iterations": [10],
        },
    }


if __name__ == "__main__":
    import time
    spec = {"output": {"grid": {"nx": 64, "ny": 64, "bbox": [0, 1, 0, 1]}}}
    t0 = time.time()
    res = solve(spec)
    print("time:", time.time() - t0)
    print("shape:", res["u"].shape, "max:", res["u"].max())
    # Compare to exact
    nx, ny = 64, 64
    xs = np.linspace(0, 1, nx)
    ys = np.linspace(0, 1, ny)
    X, Y = np.meshgrid(xs, ys)
    ux = X * (1 - X) * (1 - 2 * Y)
    uy = -Y * (1 - Y) * (1 - 2 * X)
    exact_mag = np.sqrt(ux**2 + uy**2)
    err = np.max(np.abs(res["u"] - exact_mag))
    print("max err:", err)
    print("L2 err:", np.sqrt(np.mean((res["u"] - exact_mag)**2)))
