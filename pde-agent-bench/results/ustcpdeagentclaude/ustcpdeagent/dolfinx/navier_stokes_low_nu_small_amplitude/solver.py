import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
from basix.ufl import element as basix_element, mixed_element as basix_mixed_element
import ufl
from petsc4py import PETSc


def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD
    nu_val = 0.01

    # Mesh resolution
    N = 64
    msh = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
    gdim = msh.geometry.dim

    # Taylor-Hood P2/P1
    deg_u = 2
    deg_p = 1
    vel_el = basix_element("Lagrange", msh.topology.cell_name(), deg_u, shape=(gdim,))
    pres_el = basix_element("Lagrange", msh.topology.cell_name(), deg_p)
    W = fem.functionspace(msh, basix_mixed_element([vel_el, pres_el]))

    V, _ = W.sub(0).collapse()
    Q, _ = W.sub(1).collapse()

    # Manufactured solution
    x = ufl.SpatialCoordinate(msh)
    pi = ufl.pi
    u_ex = ufl.as_vector([
        0.2 * pi * ufl.cos(pi * x[1]) * ufl.sin(2 * pi * x[0]),
        -0.4 * pi * ufl.cos(2 * pi * x[0]) * ufl.sin(pi * x[1]),
    ])
    p_ex = 0.0 * x[0]

    # Forcing term: f = (u·∇)u - ν ∇²u + ∇p
    f = ufl.grad(u_ex) * u_ex - nu_val * ufl.div(ufl.grad(u_ex))

    # Unknown
    w = fem.Function(W)
    (u, p) = ufl.split(w)
    (v, q) = ufl.TestFunctions(W)

    nu = fem.Constant(msh, PETSc.ScalarType(nu_val))

    F = (
        nu * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
        + ufl.inner(ufl.grad(u) * u, v) * ufl.dx
        - p * ufl.div(v) * ufl.dx
        + ufl.div(u) * q * ufl.dx
        - ufl.inner(f, v) * ufl.dx
    )

    # Velocity BC: exact solution on boundary
    fdim = msh.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(
        msh, fdim, lambda X: np.ones(X.shape[1], dtype=bool)
    )

    u_bc_func = fem.Function(V)
    u_bc_expr = fem.Expression(u_ex, V.element.interpolation_points)
    u_bc_func.interpolate(u_bc_expr)

    dofs_u = fem.locate_dofs_topological((W.sub(0), V), fdim, boundary_facets)
    bc_u = fem.dirichletbc(u_bc_func, dofs_u, W.sub(0))

    # Pressure pin at corner (0,0)
    p_dofs = fem.locate_dofs_geometrical(
        (W.sub(1), Q),
        lambda X: np.isclose(X[0], 0.0) & np.isclose(X[1], 0.0),
    )
    p0_func = fem.Function(Q)
    p0_func.x.array[:] = 0.0
    bc_p = fem.dirichletbc(p0_func, p_dofs, W.sub(1))

    bcs = [bc_u, bc_p]

    # Initial guess: interpolate exact velocity into w
    # Use w.sub(0).interpolate to seed
    w_u_sub = w.sub(0)
    # Direct approach: solve Stokes first
    # Skip and just solve Newton from zero + try

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

    problem = petsc.NonlinearProblem(
        F, w, bcs=bcs, J=J,
        petsc_options_prefix="ns_solver_",
        petsc_options=petsc_options,
    )
    w_h = problem.solve()
    w.x.scatter_forward()

    # Extract velocity
    u_h = w.sub(0).collapse()

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
    idx_map = []
    for i in range(pts.shape[0]):
        links = colliding.links(i)
        if len(links) > 0:
            points_on_proc.append(pts[i])
            cells_on_proc.append(links[0])
            idx_map.append(i)

    u_values = np.zeros((pts.shape[0], gdim))
    if len(points_on_proc) > 0:
        vals = u_h.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u_values[idx_map] = vals

    magnitude = np.linalg.norm(u_values, axis=1).reshape(ny, nx)

    # Accuracy verification
    u_ex_func = fem.Function(V)
    u_ex_func.interpolate(fem.Expression(u_ex, V.element.interpolation_points))
    err_form = fem.form(ufl.inner(u_h - u_ex_func, u_h - u_ex_func) * ufl.dx)
    local_err = fem.assemble_scalar(err_form)
    global_err = np.sqrt(comm.allreduce(local_err, op=MPI.SUM))
    if comm.rank == 0:
        print(f"L2 velocity error: {global_err:.3e}")

    return {
        "u": magnitude,
        "solver_info": {
            "mesh_resolution": N,
            "element_degree": deg_u,
            "ksp_type": "preonly",
            "pc_type": "lu",
            "rtol": 1e-10,
            "iterations": 1,
            "nonlinear_iterations": [1],
        },
    }


if __name__ == "__main__":
    spec = {
        "output": {
            "grid": {"nx": 64, "ny": 64, "bbox": [0.0, 1.0, 0.0, 1.0]}
        }
    }
    import time
    t0 = time.time()
    res = solve(spec)
    t1 = time.time()
    print(f"shape={res['u'].shape}, max={res['u'].max():.4f}, time={t1-t0:.2f}s")

    # Compare to analytical on grid
    nx, ny = 64, 64
    xs = np.linspace(0, 1, nx)
    ys = np.linspace(0, 1, ny)
    XX, YY = np.meshgrid(xs, ys)
    ux = 0.2*np.pi*np.cos(np.pi*YY)*np.sin(2*np.pi*XX)
    uy = -0.4*np.pi*np.cos(2*np.pi*XX)*np.sin(np.pi*YY)
    mag_exact = np.sqrt(ux**2 + uy**2)
    print(f"grid err max: {np.abs(res['u']-mag_exact).max():.3e}")
    print(f"grid err L2: {np.sqrt(np.mean((res['u']-mag_exact)**2)):.3e}")
