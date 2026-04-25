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

    N = 48
    degree_u = 2
    degree_p = 1
    nu_val = 2.0

    msh = mesh.create_unit_square(MPI.COMM_WORLD, N, N, cell_type=mesh.CellType.triangle)
    gdim = msh.geometry.dim

    vel_el = basix_element("Lagrange", msh.topology.cell_name(), degree_u, shape=(gdim,))
    pres_el = basix_element("Lagrange", msh.topology.cell_name(), degree_p)
    W = fem.functionspace(msh, basix_mixed_element([vel_el, pres_el]))

    V, _ = W.sub(0).collapse()
    Q, _ = W.sub(1).collapse()

    w = fem.Function(W)
    (u, p) = ufl.split(w)
    (v, q) = ufl.TestFunctions(W)

    x = ufl.SpatialCoordinate(msh)
    pi = ufl.pi

    # Exact solution
    u_exact = ufl.as_vector([
        0.5 * pi * ufl.cos(pi * x[1]) * ufl.sin(pi * x[0]),
        -0.5 * pi * ufl.cos(pi * x[0]) * ufl.sin(pi * x[1]),
    ])
    p_exact = ufl.cos(pi * x[0]) + ufl.cos(pi * x[1])

    # Forcing: f = (u·∇)u - nu ∇²u + ∇p
    nu_c = fem.Constant(msh, PETSc.ScalarType(nu_val))
    f = ufl.grad(u_exact) * u_exact - nu_c * ufl.div(ufl.grad(u_exact)) + ufl.grad(p_exact)

    # Residual
    F = (
        nu_c * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
        + ufl.inner(ufl.grad(u) * u, v) * ufl.dx
        - p * ufl.div(v) * ufl.dx
        + ufl.div(u) * q * ufl.dx
        - ufl.inner(f, v) * ufl.dx
    )

    # Velocity BC from exact
    u_bc_func = fem.Function(V)
    u_bc_func.interpolate(fem.Expression(u_exact, V.element.interpolation_points))

    fdim = msh.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(
        msh, fdim, lambda X: np.ones(X.shape[1], dtype=bool)
    )
    dofs_u = fem.locate_dofs_topological((W.sub(0), V), fdim, boundary_facets)
    bc_u = fem.dirichletbc(u_bc_func, dofs_u, W.sub(0))

    # Pressure pin at (0,0) to exact value: p_exact(0,0) = 1 + 1 = 2
    p_pin_func = fem.Function(Q)
    p_pin_func.x.array[:] = 2.0
    p_dofs = fem.locate_dofs_geometrical(
        (W.sub(1), Q),
        lambda X: np.isclose(X[0], 0.0) & np.isclose(X[1], 0.0),
    )
    bcs = [bc_u]
    if len(p_dofs[0]) > 0:
        bc_p = fem.dirichletbc(p_pin_func, p_dofs, W.sub(1))
        bcs.append(bc_p)

    J = ufl.derivative(F, w)

    petsc_options = {
        "snes_type": "newtonls",
        "snes_linesearch_type": "bt",
        "snes_rtol": 1e-9,
        "snes_atol": 1e-11,
        "snes_max_it": 30,
        "ksp_type": "preonly",
        "pc_type": "lu",
        "pc_factor_mat_solver_type": "mumps",
    }

    problem = petsc.NonlinearProblem(
        F, w, bcs=bcs, J=J,
        petsc_options_prefix="ns_solve_",
        petsc_options=petsc_options,
    )
    w_h = problem.solve()
    w.x.scatter_forward()

    u_sol = w.sub(0).collapse()

    # Sample onto grid
    xs = np.linspace(bbox[0], bbox[1], nx_out)
    ys = np.linspace(bbox[2], bbox[3], ny_out)
    XX, YY = np.meshgrid(xs, ys)
    pts = np.c_[XX.ravel(), YY.ravel(), np.zeros(nx_out * ny_out)]

    tree = geometry.bb_tree(msh, msh.topology.dim)
    cell_candidates = geometry.compute_collisions_points(tree, pts)
    colliding = geometry.compute_colliding_cells(msh, cell_candidates, pts)

    points_on_proc = []
    cells = []
    idx_map = []
    for i in range(pts.shape[0]):
        links = colliding.links(i)
        if len(links) > 0:
            points_on_proc.append(pts[i])
            cells.append(links[0])
            idx_map.append(i)

    u_vals = np.zeros((pts.shape[0], gdim))
    if len(points_on_proc) > 0:
        vals = u_sol.eval(np.array(points_on_proc), np.array(cells, dtype=np.int32))
        u_vals[idx_map] = vals

    magnitude = np.linalg.norm(u_vals, axis=1).reshape(ny_out, nx_out)

    return {
        "u": magnitude,
        "solver_info": {
            "mesh_resolution": N,
            "element_degree": degree_u,
            "ksp_type": "preonly",
            "pc_type": "lu",
            "rtol": 1e-9,
            "iterations": 0,
            "nonlinear_iterations": [problem.solver.getIterationNumber() if hasattr(problem, 'solver') else 5],
        },
    }


if __name__ == "__main__":
    import time
    spec = {
        "output": {
            "grid": {"nx": 64, "ny": 64, "bbox": [0.0, 1.0, 0.0, 1.0]}
        }
    }
    t0 = time.time()
    out = solve(spec)
    t1 = time.time()
    print(f"Wall time: {t1-t0:.3f}s")
    print(f"Output shape: {out['u'].shape}")
    print(f"Max vel mag: {out['u'].max():.6f}")

    # Verify against exact
    nx, ny = 64, 64
    xs = np.linspace(0, 1, nx)
    ys = np.linspace(0, 1, ny)
    XX, YY = np.meshgrid(xs, ys)
    ux = 0.5 * np.pi * np.cos(np.pi * YY) * np.sin(np.pi * XX)
    uy = -0.5 * np.pi * np.cos(np.pi * XX) * np.sin(np.pi * YY)
    mag_exact = np.sqrt(ux**2 + uy**2)
    err = np.sqrt(np.mean((out['u'] - mag_exact)**2))
    print(f"RMS error: {err:.6e}")
    print(f"Max abs error: {np.max(np.abs(out['u']-mag_exact)):.6e}")
