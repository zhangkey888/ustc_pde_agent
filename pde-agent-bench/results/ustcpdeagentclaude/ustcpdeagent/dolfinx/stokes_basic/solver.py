import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc as fem_petsc
from basix.ufl import element as basix_element, mixed_element as basix_mixed_element
import ufl
from petsc4py import PETSc


def solve(case_spec: dict) -> dict:
    grid = case_spec["output"]["grid"]
    nx_out = grid["nx"]
    ny_out = grid["ny"]
    bbox = grid["bbox"]

    nu_val = 1.0

    N = 48
    degree_u = 3
    degree_p = 2

    msh = mesh.create_unit_square(MPI.COMM_WORLD, N, N, cell_type=mesh.CellType.triangle)
    gdim = msh.geometry.dim

    vel_el = basix_element("Lagrange", msh.topology.cell_name(), degree_u, shape=(gdim,))
    pres_el = basix_element("Lagrange", msh.topology.cell_name(), degree_p)
    W = fem.functionspace(msh, basix_mixed_element([vel_el, pres_el]))

    V, _ = W.sub(0).collapse()
    Q, _ = W.sub(1).collapse()

    (u, p) = ufl.TrialFunctions(W)
    (v, q) = ufl.TestFunctions(W)

    x = ufl.SpatialCoordinate(msh)
    pi = ufl.pi

    u_ex = ufl.as_vector([
        pi * ufl.cos(pi * x[1]) * ufl.sin(pi * x[0]),
        -pi * ufl.cos(pi * x[0]) * ufl.sin(pi * x[1]),
    ])
    p_ex = ufl.cos(pi * x[0]) * ufl.cos(pi * x[1])

    # f = -nu * laplacian(u) + grad(p)
    f = -nu_val * ufl.div(ufl.grad(u_ex)) + ufl.grad(p_ex)

    a = (nu_val * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
         - p * ufl.div(v) * ufl.dx
         + ufl.div(u) * q * ufl.dx)
    L = ufl.inner(f, v) * ufl.dx

    # Dirichlet BC for velocity on all boundaries using exact solution
    fdim = msh.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(
        msh, fdim, lambda xx: np.ones(xx.shape[1], dtype=bool)
    )
    u_bc_func = fem.Function(V)
    u_bc_func.interpolate(fem.Expression(u_ex, V.element.interpolation_points))
    dofs_u = fem.locate_dofs_topological((W.sub(0), V), fdim, boundary_facets)
    bc_u = fem.dirichletbc(u_bc_func, dofs_u, W.sub(0))

    # Pressure pin at (0,0) to exact pressure value cos(0)*cos(0)=1
    p_dofs = fem.locate_dofs_geometrical(
        (W.sub(1), Q),
        lambda xx: np.isclose(xx[0], 0.0) & np.isclose(xx[1], 0.0),
    )
    p_pin_func = fem.Function(Q)
    p_pin_func.interpolate(fem.Expression(p_ex, Q.element.interpolation_points))
    bc_p = fem.dirichletbc(p_pin_func, p_dofs, W.sub(1))

    bcs = [bc_u, bc_p]

    problem = fem_petsc.LinearProblem(
        a, L, bcs=bcs,
        petsc_options={"ksp_type": "preonly", "pc_type": "lu",
                       "pc_factor_mat_solver_type": "mumps"},
        petsc_options_prefix="stokes_",
    )
    w_h = problem.solve()

    try:
        its = problem.solver.getIterationNumber()
    except Exception:
        its = 1

    u_h = w_h.sub(0).collapse()
    p_h = w_h.sub(1).collapse()

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
        u_vals[eval_map] = vals

    magnitude = np.linalg.norm(u_vals, axis=1).reshape(ny_out, nx_out)

    # Accuracy check via L2 error
    err_u = fem.form(ufl.inner(u_h - u_ex, u_h - u_ex) * ufl.dx)
    L2_err = np.sqrt(msh.comm.allreduce(fem.assemble_scalar(err_u), op=MPI.SUM))

    return {
        "u": magnitude,
        "solver_info": {
            "mesh_resolution": N,
            "element_degree": degree_u,
            "ksp_type": "preonly",
            "pc_type": "lu",
            "rtol": 1e-12,
            "iterations": int(its),
            "L2_velocity_error": float(L2_err),
        },
    }


if __name__ == "__main__":
    import time
    spec = {"output": {"grid": {"nx": 64, "ny": 64, "bbox": [0, 1, 0, 1]}}}
    t0 = time.time()
    res = solve(spec)
    t1 = time.time()
    print(f"time={t1-t0:.3f}s")
    print("info:", res["solver_info"])
    print("u shape:", res["u"].shape, "max:", res["u"].max())
