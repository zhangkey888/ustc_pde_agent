import os
os.environ["OMP_NUM_THREADS"] = "4"

import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
from basix.ufl import element as basix_element, mixed_element as basix_mixed_element
import ufl
from petsc4py import PETSc

def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD
    nu_val = 0.25

    # Parse output grid spec
    grid_spec = case_spec["output"]["grid"]
    nx_out = grid_spec["nx"]
    ny_out = grid_spec["ny"]
    bbox = grid_spec["bbox"]  # [xmin, xmax, ymin, ymax]
    xmin, xmax, ymin, ymax = bbox

    # Mesh resolution - high accuracy within time budget
    N = 256

    # Create mesh
    msh = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
    gdim = msh.geometry.dim

    # Mixed function space: Taylor-Hood P2/P1
    vel_el = basix_element("Lagrange", msh.topology.cell_name(), 2, shape=(gdim,))
    pres_el = basix_element("Lagrange", msh.topology.cell_name(), 1)
    W = fem.functionspace(msh, basix_mixed_element([vel_el, pres_el]))
    V, _ = W.sub(0).collapse()
    Q, _ = W.sub(1).collapse()

    # Manufactured solution (UFL)
    x = ufl.SpatialCoordinate(msh)
    u_ex1 = x[0] * (1 - x[0]) * (1 - 2*x[1])
    u_ex2 = -x[1] * (1 - x[1]) * (1 - 2*x[0])
    u_exact = ufl.as_vector([u_ex1, u_ex2])
    p_exact = x[0] - x[1]

    # Source term f = (u·∇)u - ν∇²u + ∇p
    lap_u1 = ufl.div(ufl.grad(u_ex1))
    lap_u2 = ufl.div(ufl.grad(u_ex2))
    conv = ufl.grad(u_exact) * u_exact
    grad_p = ufl.grad(p_exact)
    f1 = conv[0] - nu_val * lap_u1 + grad_p[0]
    f2 = conv[1] - nu_val * lap_u2 + grad_p[1]
    f_expr = ufl.as_vector([f1, f2])

    # Boundary conditions - velocity on entire boundary from manufactured solution
    fdim = msh.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(
        msh, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
    )

    u_bc_func = fem.Function(V)
    u_bc_func.interpolate(
        fem.Expression(ufl.as_vector([u_ex1, u_ex2]), V.element.interpolation_points)
    )

    dofs_u = fem.locate_dofs_topological((W.sub(0), V), fdim, boundary_facets)
    bc_u = fem.dirichletbc(u_bc_func, dofs_u, W.sub(0))

    bcs = [bc_u]

    # Pressure pin at origin corner to fix gauge
    p_dofs = fem.locate_dofs_geometrical(
        (W.sub(1), Q),
        lambda x: np.isclose(x[0], 0.0) & np.isclose(x[1], 0.0)
    )
    if len(p_dofs[0]) > 0:
        p0_func = fem.Function(Q)
        p0_func.x.array[:] = 0.0
        bc_p = fem.dirichletbc(p0_func, p_dofs, W.sub(1))
        bcs.append(bc_p)

    # Step 1: Stokes solve for initial guess (drop convection term)
    (u_trial, p_trial) = ufl.TrialFunctions(W)
    (v, q) = ufl.TestFunctions(W)

    a_stokes = (nu_val * ufl.inner(ufl.grad(u_trial), ufl.grad(v)) * ufl.dx
                - p_trial * ufl.div(v) * ufl.dx
                + ufl.div(u_trial) * q * ufl.dx)
    L_stokes = ufl.inner(f_expr, v) * ufl.dx

    stokes_problem = petsc.LinearProblem(
        a_stokes, L_stokes, bcs=bcs,
        petsc_options={
            "ksp_type": "preonly",
            "pc_type": "lu",
            "pc_factor_mat_solver_type": "mumps",
        },
        petsc_options_prefix="stokes_"
    )
    w_stokes = stokes_problem.solve()
    w_stokes.x.scatter_forward()

    # Step 2: Nonlinear NS solve with Newton's method
    w = fem.Function(W)
    w.x.array[:] = w_stokes.x.array[:]
    w.x.scatter_forward()

    (u, p) = ufl.split(w)
    (v, q) = ufl.TestFunctions(W)

    F = (nu_val * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
         + ufl.inner(ufl.grad(u) * u, v) * ufl.dx
         - p * ufl.div(v) * ufl.dx
         + ufl.div(u) * q * ufl.dx
         - ufl.inner(f_expr, v) * ufl.dx)

    J = ufl.derivative(F, w)

    problem = petsc.NonlinearProblem(
        F, w, bcs=bcs, J=J,
        petsc_options_prefix="ns_",
        petsc_options={
            "snes_type": "newtonls",
            "snes_linesearch_type": "bt",
            "snes_rtol": 1e-10,
            "snes_atol": 1e-12,
            "snes_max_it": 50,
            "ksp_type": "preonly",
            "pc_type": "lu",
            "pc_factor_mat_solver_type": "mumps",
        }
    )

    w_h = problem.solve()
    w.x.scatter_forward()

    snes = problem._snes
    newton_its = int(snes.getIterationNumber())

    # Extract velocity and compute magnitude
    u_sol = w.sub(0).collapse()
    V_mag = fem.functionspace(msh, ("Lagrange", 2))
    u_mag = fem.Function(V_mag)
    u_mag.interpolate(
        fem.Expression(ufl.sqrt(ufl.inner(u_sol, u_sol)), V_mag.element.interpolation_points)
    )

    # Sample on output grid
    xs = np.linspace(xmin, xmax, nx_out)
    ys = np.linspace(ymin, ymax, ny_out)
    XX, YY = np.meshgrid(xs, ys)
    pts = np.zeros((nx_out * ny_out, 3))
    pts[:, 0] = XX.ravel()
    pts[:, 1] = YY.ravel()

    bb_tree = geometry.bb_tree(msh, msh.topology.dim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, pts)
    colliding_cells = geometry.compute_colliding_cells(msh, cell_candidates, pts)

    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    for i in range(pts.shape[0]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(pts[i])
            cells_on_proc.append(links[0])
            eval_map.append(i)

    u_grid = np.full((ny_out, nx_out), np.nan)
    if len(points_on_proc) > 0:
        vals = u_mag.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        idx = np.array(eval_map)
        row = idx // nx_out
        col = idx % nx_out
        u_grid[row, col] = vals.flatten()

    # Gather from all processes
    u_grid_flat = u_grid.ravel().copy()
    gathered = comm.allgather(u_grid_flat)
    combined = np.full_like(u_grid_flat, np.nan)
    for arr in gathered:
        mask = ~np.isnan(arr)
        combined[mask] = arr[mask]
    u_grid = combined.reshape(ny_out, nx_out)

    # Compute L2 error for verification
    u_err_expr = ufl.inner(u_sol - u_exact, u_sol - u_exact)
    L2_err_sq = fem.assemble_scalar(fem.form(u_err_expr * ufl.dx))
    L2_err = np.sqrt(comm.allreduce(L2_err_sq, op=MPI.SUM))

    solver_info = {
        "mesh_resolution": N,
        "element_degree": 2,
        "ksp_type": "preonly",
        "pc_type": "lu",
        "rtol": 1e-10,
        "iterations": 1 + newton_its,  # 1 for Stokes LU + 1 per Newton step
        "nonlinear_iterations": [newton_its],
    }

    if comm.rank == 0:
        print(f"[solver] N={N}, L2 velocity error: {L2_err:.6e}, Newton its: {newton_its}")

    return {"u": u_grid, "solver_info": solver_info}


if __name__ == "__main__":
    case_spec = {
        "output": {
            "grid": {
                "nx": 50,
                "ny": 50,
                "bbox": [0.0, 1.0, 0.0, 1.0]
            }
        }
    }
    result = solve(case_spec)
    if MPI.COMM_WORLD.rank == 0:
        print(f"Grid shape: {result['u'].shape}")
        print(f"Max velocity magnitude: {np.nanmax(result['u']):.6f}")
        print(f"NaN count: {np.sum(np.isnan(result['u']))}")
        print(f"Solver info: {result['solver_info']}")
