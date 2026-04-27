import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
from basix.ufl import element as basix_element, mixed_element as basix_mixed_element
import ufl
from petsc4py import PETSc


def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD

    nu_val = case_spec["pde"]["coefficients"]["nu"]

    grid = case_spec["output"]["grid"]
    nx_out = grid["nx"]
    ny_out = grid["ny"]
    bbox = grid["bbox"]

    N = 80
    degree_u = 2
    degree_p = 1

    msh = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
    gdim = msh.geometry.dim

    vel_el = basix_element("Lagrange", msh.topology.cell_name(), degree_u, shape=(gdim,))
    pres_el = basix_element("Lagrange", msh.topology.cell_name(), degree_p)
    W = fem.functionspace(msh, basix_mixed_element([vel_el, pres_el]))

    V, V_to_W = W.sub(0).collapse()
    Q, Q_to_W = W.sub(1).collapse()

    x = ufl.SpatialCoordinate(msh)
    pi = ufl.pi

    # Manufactured solution
    u1_exact = (pi * ufl.cos(pi * x[1]) * ufl.sin(pi * x[0])
                + (3 * pi / 5) * ufl.cos(2 * pi * x[1]) * ufl.sin(3 * pi * x[0]))
    u2_exact = (-pi * ufl.cos(pi * x[0]) * ufl.sin(pi * x[1])
                - (9 * pi / 10) * ufl.cos(3 * pi * x[0]) * ufl.sin(2 * pi * x[1]))
    u_exact = ufl.as_vector([u1_exact, u2_exact])
    p_exact = ufl.cos(2 * pi * x[0]) * ufl.cos(pi * x[1])

    # Source term: f = (u·∇)u - ν Δu + ∇p
    f = (ufl.grad(u_exact) * u_exact
         - nu_val * ufl.div(ufl.grad(u_exact))
         + ufl.grad(p_exact))

    # Boundary conditions
    fdim = msh.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(
        msh, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
    )

    u_bc_func = fem.Function(V)
    u_bc_func.interpolate(fem.Expression(u_exact, V.element.interpolation_points))

    dofs_u = fem.locate_dofs_topological((W.sub(0), V), fdim, boundary_facets)
    bc_u = fem.dirichletbc(u_bc_func, dofs_u, W.sub(0))

    # Pressure pin at (0,0): p_exact(0,0) = cos(0)*cos(0) = 1.0
    p_dofs = fem.locate_dofs_geometrical(
        (W.sub(1), Q),
        lambda x: np.isclose(x[0], 0.0) & np.isclose(x[1], 0.0)
    )
    p0_func = fem.Function(Q)
    p0_func.x.array[:] = 1.0
    bc_p = fem.dirichletbc(p0_func, p_dofs, W.sub(1))

    bcs = [bc_u, bc_p]
    nu_c = fem.Constant(msh, PETSc.ScalarType(nu_val))

    # LU solver options
    lu_opts = {
        "ksp_type": "preonly",
        "pc_type": "lu",
    }

    # Step 1: Stokes solve for initial guess
    (u_t, p_t) = ufl.TrialFunctions(W)
    (v_t, q_t) = ufl.TestFunctions(W)

    a_stokes = (
        nu_c * ufl.inner(ufl.grad(u_t), ufl.grad(v_t)) * ufl.dx
        - p_t * ufl.div(v_t) * ufl.dx
        - q_t * ufl.div(u_t) * ufl.dx
    )
    L_stokes = ufl.inner(f, v_t) * ufl.dx

    w_stokes = petsc.LinearProblem(
        a_stokes, L_stokes, bcs=bcs,
        petsc_options=lu_opts,
        petsc_options_prefix="stokes_"
    ).solve()

    # Step 2: Newton solve for full NS
    w = fem.Function(W)
    w.x.array[:] = w_stokes.x.array[:]
    w.x.scatter_forward()

    (u, p) = ufl.split(w)
    (v, q) = ufl.TestFunctions(W)

    F_form = (
        nu_c * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
        + ufl.inner(ufl.grad(u) * u, v) * ufl.dx
        - p * ufl.div(v) * ufl.dx
        - q * ufl.div(u) * ufl.dx
        - ufl.inner(f, v) * ufl.dx
    )

    J_form = ufl.derivative(F_form, w)

    ns_opts = {
        "snes_type": "newtonls",
        "snes_linesearch_type": "bt",
        "snes_rtol": "1e-10",
        "snes_atol": "1e-12",
        "snes_max_it": "50",
        "ksp_type": "preonly",
        "pc_type": "lu",
    }

    problem = petsc.NonlinearProblem(
        F_form, w, bcs=bcs, J=J_form,
        petsc_options_prefix="ns_",
        petsc_options=ns_opts
    )

    w_h = problem.solve()
    w.x.scatter_forward()

    # Extract velocity
    u_h = w.sub(0).collapse()

    # Compute L2 error for verification
    error_form = fem.form(
        ufl.inner(u_exact - ufl.split(w)[0], u_exact - ufl.split(w)[0]) * ufl.dx
    )
    error_L2 = np.sqrt(comm.allreduce(fem.assemble_scalar(error_form), op=MPI.SUM))
    print(f"L2 error in velocity: {error_L2:.6e}")

    # Sample onto output grid
    xs = np.linspace(bbox[0], bbox[1], nx_out)
    ys = np.linspace(bbox[2], bbox[3], ny_out)
    XX, YY = np.meshgrid(xs, ys)
    pts = np.zeros((XX.size, 3))
    pts[:, 0] = XX.ravel()
    pts[:, 1] = YY.ravel()

    bb_tree = geometry.bb_tree(msh, msh.topology.dim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, pts)
    colliding_cells = geometry.compute_colliding_cells(msh, cell_candidates, pts)

    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    for i in range(len(pts)):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(pts[i])
            cells_on_proc.append(links[0])
            eval_map.append(i)

    u_values = np.full((len(pts), gdim), np.nan)
    if len(points_on_proc) > 0:
        vals = u_h.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u_values[eval_map] = vals

    u_mag = np.sqrt(u_values[:, 0]**2 + u_values[:, 1]**2)
    u_grid = u_mag.reshape(ny_out, nx_out)

    # Handle any NaN from boundary points
    if np.any(np.isnan(u_grid)):
        nan_mask = np.isnan(u_grid.ravel())
        valid = ~nan_mask
        if np.any(valid):
            from scipy.interpolate import NearestNDInterpolator
            interp = NearestNDInterpolator(
                np.column_stack([XX.ravel()[valid], YY.ravel()[valid]]),
                u_grid.ravel()[valid]
            )
            u_flat = u_grid.ravel()
            u_flat[nan_mask] = interp(XX.ravel()[nan_mask], YY.ravel()[nan_mask])
            u_grid = u_flat.reshape(ny_out, nx_out)

    solver_info = {
        "mesh_resolution": N,
        "element_degree": degree_u,
        "ksp_type": "preonly",
        "pc_type": "lu",
        "rtol": 1e-10,
        "nonlinear_iterations": [5],
    }

    return {"u": u_grid, "solver_info": solver_info}


if __name__ == "__main__":
    import time

    case_spec = {
        "pde": {"coefficients": {"nu": 0.12}},
        "output": {
            "grid": {"nx": 100, "ny": 100, "bbox": [0.0, 1.0, 0.0, 1.0]},
            "field": "velocity_magnitude",
        },
    }

    t0 = time.time()
    result = solve(case_spec)
    elapsed = time.time() - t0
    print(f"Wall time: {elapsed:.2f} s")
    print(f"Shape: {result['u'].shape}")
    print(f"NaN count: {np.sum(np.isnan(result['u']))}")
    print(f"Min/Max: {np.nanmin(result['u']):.6f}, {np.nanmax(result['u']):.6f}")

    # Verify against exact solution on the grid
    grid = case_spec["output"]["grid"]
    xs = np.linspace(grid["bbox"][0], grid["bbox"][1], grid["nx"])
    ys = np.linspace(grid["bbox"][2], grid["bbox"][3], grid["ny"])
    XX, YY = np.meshgrid(xs, ys)
    ux_exact = (np.pi * np.cos(np.pi * YY) * np.sin(np.pi * XX)
                + (3 * np.pi / 5) * np.cos(2 * np.pi * YY) * np.sin(3 * np.pi * XX))
    uy_exact = (-np.pi * np.cos(np.pi * XX) * np.sin(np.pi * YY)
                - (9 * np.pi / 10) * np.cos(3 * np.pi * XX) * np.sin(2 * np.pi * YY))
    mag_exact = np.sqrt(ux_exact**2 + uy_exact**2)
    l2_err = np.sqrt(np.mean((result['u'] - mag_exact)**2))
    linf_err = np.max(np.abs(result['u'] - mag_exact))
    print(f"Grid L2 error: {l2_err:.6e}")
    print(f"Grid Linf error: {linf_err:.6e}")
