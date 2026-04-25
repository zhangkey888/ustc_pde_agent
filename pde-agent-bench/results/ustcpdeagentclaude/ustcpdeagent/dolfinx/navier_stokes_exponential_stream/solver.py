import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
from basix.ufl import element as basix_element, mixed_element as basix_mixed_element
import ufl
from petsc4py import PETSc


def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD
    nu_val = 0.15

    # Mesh
    N = 64
    msh = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
    gdim = msh.geometry.dim

    # Taylor-Hood P2/P1
    deg_u, deg_p = 2, 1
    vel_el = basix_element("Lagrange", msh.topology.cell_name(), deg_u, shape=(gdim,))
    pres_el = basix_element("Lagrange", msh.topology.cell_name(), deg_p)
    W = fem.functionspace(msh, basix_mixed_element([vel_el, pres_el]))
    V, _ = W.sub(0).collapse()
    Q, _ = W.sub(1).collapse()

    # Manufactured solution (UFL)
    x = ufl.SpatialCoordinate(msh)
    pi = ufl.pi
    u_ex = ufl.as_vector([
        pi * ufl.exp(2 * x[0]) * ufl.cos(pi * x[1]),
        -2.0 * ufl.exp(2 * x[0]) * ufl.sin(pi * x[1]),
    ])
    p_ex = ufl.exp(x[0]) * ufl.cos(pi * x[1])

    nu = fem.Constant(msh, PETSc.ScalarType(nu_val))

    # Compute forcing f = u·∇u - ν Δu + ∇p
    f_expr = ufl.grad(u_ex) * u_ex - nu * ufl.div(ufl.grad(u_ex)) + ufl.grad(p_ex)

    # Unknown
    w = fem.Function(W)
    (u, p) = ufl.split(w)
    (v, q) = ufl.TestFunctions(W)

    # Residual (standard form)
    F = (
        nu * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
        + ufl.inner(ufl.grad(u) * u, v) * ufl.dx
        - p * ufl.div(v) * ufl.dx
        + ufl.div(u) * q * ufl.dx
        - ufl.inner(f_expr, v) * ufl.dx
    )
    J = ufl.derivative(F, w)

    # BCs: Dirichlet u = u_ex on entire boundary
    fdim = msh.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(
        msh, fdim, lambda X: np.ones(X.shape[1], dtype=bool)
    )
    u_bc = fem.Function(V)
    u_bc.interpolate(fem.Expression(u_ex, V.element.interpolation_points))
    dofs_u = fem.locate_dofs_topological((W.sub(0), V), fdim, boundary_facets)
    bc_u = fem.dirichletbc(u_bc, dofs_u, W.sub(0))

    # Pressure pin at (0,0): use exact pressure value there = exp(0)*cos(0) = 1.0
    p_bc = fem.Function(Q)
    p_bc.interpolate(fem.Expression(p_ex, Q.element.interpolation_points))
    p_dofs = fem.locate_dofs_geometrical(
        (W.sub(1), Q),
        lambda X: np.isclose(X[0], 0.0) & np.isclose(X[1], 0.0),
    )
    bcs = [bc_u]
    if len(p_dofs[0]) > 0:
        bc_p = fem.dirichletbc(p_bc, p_dofs, W.sub(1))
        bcs.append(bc_p)

    # Initial guess: interpolate exact u to help Newton
    w_u, w_u_map = W.sub(0).collapse()
    # Set initial guess = 0 (simple)
    w.x.array[:] = 0.0

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
        petsc_options_prefix="ns_",
        petsc_options=petsc_options,
    )
    w_h = problem.solve()
    w.x.scatter_forward()

    # Newton iterations
    try:
        snes = problem.solver
        nl_iters = int(snes.getIterationNumber())
    except Exception:
        nl_iters = -1

    # Extract velocity
    u_sol = w.sub(0).collapse()

    # Sample on output grid
    grid = case_spec["output"]["grid"]
    nx = grid["nx"]; ny = grid["ny"]
    bbox = grid["bbox"]
    xs = np.linspace(bbox[0], bbox[1], nx)
    ys = np.linspace(bbox[2], bbox[3], ny)
    XX, YY = np.meshgrid(xs, ys)
    pts = np.column_stack([XX.ravel(), YY.ravel(), np.zeros(nx * ny)])

    tree = geometry.bb_tree(msh, msh.topology.dim)
    cand = geometry.compute_collisions_points(tree, pts)
    col = geometry.compute_colliding_cells(msh, cand, pts)

    points_on_proc = []
    cells_on_proc = []
    idx_map = []
    for i in range(pts.shape[0]):
        links = col.links(i)
        if len(links) > 0:
            points_on_proc.append(pts[i])
            cells_on_proc.append(links[0])
            idx_map.append(i)

    vals = np.zeros((pts.shape[0], gdim))
    if points_on_proc:
        v_eval = u_sol.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        vals[idx_map] = v_eval

    mag = np.linalg.norm(vals, axis=1).reshape(ny, nx)

    return {
        "u": mag,
        "solver_info": {
            "mesh_resolution": N,
            "element_degree": deg_u,
            "ksp_type": "preonly",
            "pc_type": "lu",
            "rtol": 1e-10,
            "iterations": 0,
            "nonlinear_iterations": [nl_iters],
        },
    }


if __name__ == "__main__":
    import time
    spec = {"output": {"grid": {"nx": 64, "ny": 64, "bbox": [0.0, 1.0, 0.0, 1.0]}}}
    t0 = time.time()
    out = solve(spec)
    t1 = time.time()
    print(f"Wall time: {t1-t0:.2f}s")
    print(f"u shape: {out['u'].shape}")
    print(f"u min/max: {out['u'].min():.4f} / {out['u'].max():.4f}")
    print(f"solver_info: {out['solver_info']}")

    # Accuracy check against exact
    nx, ny = 64, 64
    xs = np.linspace(0, 1, nx); ys = np.linspace(0, 1, ny)
    XX, YY = np.meshgrid(xs, ys)
    u1 = np.pi * np.exp(2*XX) * np.cos(np.pi*YY)
    u2 = -2.0 * np.exp(2*XX) * np.sin(np.pi*YY)
    mag_ex = np.sqrt(u1**2 + u2**2)
    err = np.sqrt(np.mean((out['u'] - mag_ex)**2))
    print(f"RMS error vs exact magnitude: {err:.6e}")
    print(f"Max error: {np.max(np.abs(out['u'] - mag_ex)):.6e}")
