import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
from basix.ufl import element as basix_element, mixed_element as basix_mixed_element
import ufl
from petsc4py import PETSc


def _build_forms(msh, nu, degree_u, degree_p, use_quad=False):
    gdim = msh.geometry.dim
    cell_name = msh.topology.cell_name()
    vel_el = basix_element("Lagrange", cell_name, degree_u, shape=(gdim,))
    pres_el = basix_element("Lagrange", cell_name, degree_p)
    W = fem.functionspace(msh, basix_mixed_element([vel_el, pres_el]))
    V, _ = W.sub(0).collapse()
    Q, _ = W.sub(1).collapse()

    x = ufl.SpatialCoordinate(msh)
    pi = ufl.pi
    sin = ufl.sin
    cos = ufl.cos

    # Manufactured solution
    u_ex = ufl.as_vector([
        pi*cos(pi*x[1])*sin(pi*x[0]) + pi*cos(4*pi*x[1])*sin(2*pi*x[0]),
        -pi*cos(pi*x[0])*sin(pi*x[1]) - (pi/2)*cos(2*pi*x[0])*sin(4*pi*x[1])
    ])
    p_ex = sin(pi*x[0])*cos(2*pi*x[1])

    # f = u·∇u - ν Δu + ∇p
    grad_u = ufl.grad(u_ex)
    conv = grad_u * u_ex  # (u·∇)u
    lap_u = ufl.div(ufl.grad(u_ex))
    grad_p = ufl.grad(p_ex)
    f = conv - nu*lap_u + grad_p

    return W, V, Q, u_ex, p_ex, f


def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD
    nu = 0.1

    # Mesh resolution - quadrilateral domain
    N = 80
    degree_u = 3
    degree_p = 2

    msh = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.quadrilateral)

    W, V, Q, u_ex, p_ex, f = _build_forms(msh, nu, degree_u, degree_p)

    w = fem.Function(W)
    (u, p) = ufl.split(w)
    (v, q) = ufl.TestFunctions(W)

    F = (
        nu * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
        + ufl.inner(ufl.grad(u) * u, v) * ufl.dx
        - p * ufl.div(v) * ufl.dx
        + ufl.div(u) * q * ufl.dx
        - ufl.inner(f, v) * ufl.dx
    )
    J = ufl.derivative(F, w)

    # Velocity Dirichlet BC from exact solution
    fdim = msh.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(
        msh, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
    )
    u_bc = fem.Function(V)
    u_bc_expr = fem.Expression(u_ex, V.element.interpolation_points)
    u_bc.interpolate(u_bc_expr)

    dofs_u = fem.locate_dofs_topological((W.sub(0), V), fdim, boundary_facets)
    bc_u = fem.dirichletbc(u_bc, dofs_u, W.sub(0))

    # Pressure pin: fix p at (0,0) to exact pressure value (which is 0)
    p_dofs = fem.locate_dofs_geometrical(
        (W.sub(1), Q),
        lambda x: np.isclose(x[0], 0.0) & np.isclose(x[1], 0.0),
    )
    p_pin = fem.Function(Q)
    p_pin.x.array[:] = 0.0
    bc_p = fem.dirichletbc(p_pin, p_dofs, W.sub(1))

    bcs = [bc_u, bc_p]

    # Initialize with exact velocity to help convergence
    w.x.array[:] = 0.0

    petsc_options = {
        "snes_type": "newtonls",
        "snes_linesearch_type": "bt",
        "snes_rtol": 1e-12,
        "snes_atol": 1e-12,
        "snes_max_it": 50,
        "ksp_type": "preonly",
        "pc_type": "lu",
        "pc_factor_mat_solver_type": "mumps",
    }

    problem = petsc.NonlinearProblem(
        F, w, bcs=bcs, J=J,
        petsc_options_prefix="ns_",
        petsc_options=petsc_options,
    )
    w_h = problem.solve()
    w.x.scatter_forward()

    # Extract velocity
    u_h = w.sub(0).collapse()
    p_h = w.sub(1).collapse()

    # Compute L2 errors for diagnostics
    err_u_form = fem.form(ufl.inner(u_h - u_ex, u_h - u_ex) * ufl.dx)
    err_u = np.sqrt(comm.allreduce(fem.assemble_scalar(err_u_form), op=MPI.SUM))
    print(f"[solver] L2 velocity error = {err_u:.3e}", flush=True)

    # Sample velocity magnitude on output grid
    grid = case_spec["output"]["grid"]
    nx = grid["nx"]
    ny = grid["ny"]
    bbox = grid["bbox"]
    xs = np.linspace(bbox[0], bbox[1], nx)
    ys = np.linspace(bbox[2], bbox[3], ny)
    XX, YY = np.meshgrid(xs, ys)
    pts = np.column_stack([XX.ravel(), YY.ravel(), np.zeros(nx*ny)])

    tree = geometry.bb_tree(msh, msh.topology.dim)
    cand = geometry.compute_collisions_points(tree, pts)
    coll = geometry.compute_colliding_cells(msh, cand, pts)

    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    for i in range(pts.shape[0]):
        links = coll.links(i)
        if len(links) > 0:
            points_on_proc.append(pts[i])
            cells_on_proc.append(links[0])
            eval_map.append(i)

    vals = np.full((pts.shape[0], 2), np.nan)
    if len(points_on_proc) > 0:
        out = u_h.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        vals[eval_map] = out

    mag = np.sqrt(vals[:, 0]**2 + vals[:, 1]**2).reshape(ny, nx)

    return {
        "u": mag,
        "solver_info": {
            "mesh_resolution": N,
            "element_degree": degree_u,
            "ksp_type": "preonly",
            "pc_type": "lu",
            "rtol": 1e-12,
            "iterations": 0,
            "nonlinear_iterations": [int(problem.solver.getIterationNumber())],
        },
    }


if __name__ == "__main__":
    import time
    spec = {"output": {"grid": {"nx": 128, "ny": 128, "bbox": [0.0, 1.0, 0.0, 1.0]}}}
    t0 = time.time()
    res = solve(spec)
    print(f"Time: {time.time()-t0:.2f}s, shape={res['u'].shape}")

    # Accuracy check vs exact magnitude
    nx, ny = 128, 128
    xs = np.linspace(0, 1, nx)
    ys = np.linspace(0, 1, ny)
    X, Y = np.meshgrid(xs, ys)
    ux = np.pi*np.cos(np.pi*Y)*np.sin(np.pi*X) + np.pi*np.cos(4*np.pi*Y)*np.sin(2*np.pi*X)
    uy = -np.pi*np.cos(np.pi*X)*np.sin(np.pi*Y) - (np.pi/2)*np.cos(2*np.pi*X)*np.sin(4*np.pi*Y)
    mag_ex = np.sqrt(ux**2 + uy**2)
    err = np.sqrt(np.mean((res["u"] - mag_ex)**2))
    maxerr = np.max(np.abs(res["u"] - mag_ex))
    print(f"RMSE = {err:.3e}, max err = {maxerr:.3e}")
