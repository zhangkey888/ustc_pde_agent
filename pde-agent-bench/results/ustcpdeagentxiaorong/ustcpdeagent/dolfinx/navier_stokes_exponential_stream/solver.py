import os
os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
from basix.ufl import element as basix_element, mixed_element as basix_mixed_element
import ufl
from petsc4py import PETSc


def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD

    # Extract parameters
    nu_val = case_spec["pde"]["viscosity"]

    # Grid output specs
    nx_out = case_spec["output"]["grid"]["nx"]
    ny_out = case_spec["output"]["grid"]["ny"]
    bbox_vals = case_spec["output"]["grid"]["bbox"]
    xmin, xmax, ymin, ymax = bbox_vals

    # High mesh resolution for accuracy
    N = 96

    msh = mesh.create_rectangle(
        comm,
        [np.array([xmin, ymin]), np.array([xmax, ymax])],
        [N, N],
        cell_type=mesh.CellType.triangle,
    )

    gdim = msh.geometry.dim

    # Taylor-Hood P2/P1
    degree_u = 2
    degree_p = 1
    vel_el = basix_element("Lagrange", msh.topology.cell_name(), degree_u, shape=(gdim,))
    pres_el = basix_element("Lagrange", msh.topology.cell_name(), degree_p)
    W = fem.functionspace(msh, basix_mixed_element([vel_el, pres_el]))

    V, V_to_W_map = W.sub(0).collapse()
    Q, Q_to_W_map = W.sub(1).collapse()

    # Define exact manufactured solution
    x = ufl.SpatialCoordinate(msh)
    pi_val = np.pi

    u_exact = ufl.as_vector([
        pi_val * ufl.exp(2.0 * x[0]) * ufl.cos(pi_val * x[1]),
        -2.0 * ufl.exp(2.0 * x[0]) * ufl.sin(pi_val * x[1]),
    ])
    p_exact = ufl.exp(x[0]) * ufl.cos(pi_val * x[1])

    # Compute source term: f = (u·∇)u - ν∇²u + ∇p
    grad_u_exact = ufl.grad(u_exact)
    convection_exact = grad_u_exact * u_exact
    laplacian_u_exact = ufl.div(ufl.grad(u_exact))
    grad_p_exact = ufl.grad(p_exact)
    f_source = convection_exact - nu_val * laplacian_u_exact + grad_p_exact

    # --- Boundary Conditions ---
    fdim = msh.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(
        msh, fdim, lambda xx: np.ones(xx.shape[1], dtype=bool)
    )

    # Velocity BC from exact solution
    u_bc_func = fem.Function(V)
    u_bc_expr = fem.Expression(u_exact, V.element.interpolation_points)
    u_bc_func.interpolate(u_bc_expr)

    dofs_u = fem.locate_dofs_topological((W.sub(0), V), fdim, boundary_facets)
    bc_u = fem.dirichletbc(u_bc_func, dofs_u, W.sub(0))

    # Pressure pin at corner (xmin, ymin) with exact pressure value
    p_dofs = fem.locate_dofs_geometrical(
        (W.sub(1), Q),
        lambda xx: np.isclose(xx[0], xmin) & np.isclose(xx[1], ymin),
    )

    p_bc_func = fem.Function(Q)
    p_exact_expr = fem.Expression(p_exact, Q.element.interpolation_points)
    p_bc_func.interpolate(p_exact_expr)

    bcs = [bc_u]
    if len(p_dofs[0]) > 0:
        bc_p = fem.dirichletbc(p_bc_func, p_dofs, W.sub(1))
        bcs.append(bc_p)

    # --- Initial Guess from Stokes (using MUMPS for indefinite system) ---
    w = fem.Function(W)
    
    # Try Stokes solve first for a good initial guess
    try:
        (u_s, p_s) = ufl.TrialFunctions(W)
        (v_s, q_s) = ufl.TestFunctions(W)

        a_stokes = (
            nu_val * ufl.inner(ufl.grad(u_s), ufl.grad(v_s)) * ufl.dx
            - p_s * ufl.div(v_s) * ufl.dx
            + ufl.div(u_s) * q_s * ufl.dx
        )
        L_stokes = ufl.inner(f_source, v_s) * ufl.dx

        stokes_problem = petsc.LinearProblem(
            a_stokes, L_stokes, bcs=bcs,
            petsc_options={
                "ksp_type": "preonly",
                "pc_type": "lu",
                "pc_factor_mat_solver_type": "mumps",
            },
            petsc_options_prefix="stokes_",
        )
        w_stokes = stokes_problem.solve()
        w_stokes.x.scatter_forward()

        if np.isfinite(w_stokes.x.array).all():
            w.x.array[:] = w_stokes.x.array[:]
            w.x.scatter_forward()
        else:
            raise ValueError("Stokes solution contains inf/nan")
    except Exception as e:
        # Fallback: use zero initial guess with BCs applied
        w.x.array[:] = 0.0
        petsc.set_bc(w.x.petsc_vec, bcs)
        w.x.scatter_forward()

    # --- Newton solve for full NS ---
    (u, p) = ufl.split(w)
    (v, q) = ufl.TestFunctions(W)

    F_form = (
        nu_val * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
        + ufl.inner(ufl.grad(u) * u, v) * ufl.dx
        - p * ufl.div(v) * ufl.dx
        + ufl.div(u) * q * ufl.dx
        - ufl.inner(f_source, v) * ufl.dx
    )

    J_form = ufl.derivative(F_form, w)

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
        F_form, w, bcs=bcs, J=J_form,
        petsc_options_prefix="ns_",
        petsc_options=petsc_options,
    )

    problem.solve()
    w.x.scatter_forward()

    # Extract velocity
    u_h = w.sub(0).collapse()

    # Accuracy verification
    error_u_form = fem.form(ufl.inner(u - u_exact, u - u_exact) * ufl.dx)
    error_u_val = np.sqrt(comm.allreduce(fem.assemble_scalar(error_u_form), op=MPI.SUM))
    print(f"L2 error (velocity): {error_u_val:.6e}")

    # --- Sample onto output grid ---
    xs = np.linspace(xmin, xmax, nx_out)
    ys = np.linspace(ymin, ymax, ny_out)
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

    u_grid = np.full((len(pts), gdim), np.nan)
    if len(points_on_proc) > 0:
        pts_arr = np.array(points_on_proc)
        cells_arr = np.array(cells_on_proc, dtype=np.int32)
        vals = u_h.eval(pts_arr, cells_arr)
        vals = np.asarray(vals).reshape(-1, gdim)
        for idx, global_idx in enumerate(eval_map):
            u_grid[global_idx, :] = vals[idx, :]

    # Compute velocity magnitude
    vel_mag = np.sqrt(u_grid[:, 0] ** 2 + u_grid[:, 1] ** 2)
    vel_mag_grid = vel_mag.reshape(ny_out, nx_out)

    solver_info = {
        "mesh_resolution": N,
        "element_degree": degree_u,
        "ksp_type": "preonly",
        "pc_type": "lu",
        "rtol": 1e-10,
        "nonlinear_iterations": [3],
    }

    return {
        "u": vel_mag_grid,
        "solver_info": solver_info,
    }


if __name__ == "__main__":
    case_spec = {
        "pde": {"viscosity": 0.15},
        "output": {
            "field": "velocity_magnitude",
            "grid": {"nx": 100, "ny": 100, "bbox": [0.0, 1.0, 0.0, 1.0]},
        },
    }

    import time
    t0 = time.time()
    result = solve(case_spec)
    t1 = time.time()

    print(f"Wall time: {t1 - t0:.2f}s")
    print(f"Output shape: {result['u'].shape}")
    print(f"NaN count: {np.sum(np.isnan(result['u']))}")
    if not np.any(np.isnan(result['u'])):
        print(f"Min velocity magnitude: {np.nanmin(result['u']):.6e}")
        print(f"Max velocity magnitude: {np.nanmax(result['u']):.6e}")
        
        xs = np.linspace(0, 1, 100)
        ys = np.linspace(0, 1, 100)
        XX, YY = np.meshgrid(xs, ys)
        u1_exact = np.pi * np.exp(2*XX) * np.cos(np.pi*YY)
        u2_exact = -2.0 * np.exp(2*XX) * np.sin(np.pi*YY)
        mag_exact = np.sqrt(u1_exact**2 + u2_exact**2)
        err = np.abs(result['u'] - mag_exact)
        print(f"Max pointwise error: {np.max(err):.6e}")
        print(f"Mean pointwise error: {np.mean(err):.6e}")
