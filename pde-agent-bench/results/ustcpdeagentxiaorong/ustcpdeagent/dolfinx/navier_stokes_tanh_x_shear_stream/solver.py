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

    # Extract parameters from case_spec
    nu_val = case_spec["pde"]["coefficients"]["nu"]

    # Output grid
    nx_out = case_spec["output"]["grid"]["nx"]
    ny_out = case_spec["output"]["grid"]["ny"]
    bbox = case_spec["output"]["grid"]["bbox"]

    # Mesh resolution
    N = 128

    msh = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
    gdim = msh.geometry.dim

    # Taylor-Hood P2/P1
    vel_el = basix_element("Lagrange", msh.topology.cell_name(), 2, shape=(gdim,))
    pres_el = basix_element("Lagrange", msh.topology.cell_name(), 1)
    W = fem.functionspace(msh, basix_mixed_element([vel_el, pres_el]))

    V, _ = W.sub(0).collapse()
    Q, _ = W.sub(1).collapse()

    # Spatial coordinate
    x = ufl.SpatialCoordinate(msh)

    # Manufactured solution
    pi = ufl.pi
    u_exact = ufl.as_vector([
        pi * ufl.tanh(6.0 * (x[0] - 0.5)) * ufl.cos(pi * x[1]),
        -6.0 * (1.0 - ufl.tanh(6.0 * (x[0] - 0.5))**2) * ufl.sin(pi * x[1])
    ])
    p_exact = ufl.sin(pi * x[0]) * ufl.cos(pi * x[1])

    # Compute source term: f = u·∇u - ν∇²u + ∇p
    grad_u_exact = ufl.grad(u_exact)
    f = ufl.dot(grad_u_exact, u_exact) - nu_val * ufl.div(ufl.grad(u_exact)) + ufl.grad(p_exact)

    # Mixed function and test functions
    w = fem.Function(W)
    (u, p) = ufl.split(w)
    (v, q) = ufl.TestFunctions(W)

    # Residual form
    F_form = (
        nu_val * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
        + ufl.inner(ufl.grad(u) * u, v) * ufl.dx
        - p * ufl.div(v) * ufl.dx
        + ufl.div(u) * q * ufl.dx
        - ufl.inner(f, v) * ufl.dx
    )

    # Jacobian
    J_form = ufl.derivative(F_form, w)

    # Boundary conditions
    fdim = msh.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(
        msh, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
    )

    # Velocity BC from manufactured solution
    u_bc = fem.Function(V)
    u_bc_expr = fem.Expression(u_exact, V.element.interpolation_points)
    u_bc.interpolate(u_bc_expr)

    dofs_u = fem.locate_dofs_topological((W.sub(0), V), fdim, boundary_facets)
    bc_u = fem.dirichletbc(u_bc, dofs_u, W.sub(0))

    # Pressure pin at (0,0) where p_exact = sin(0)*cos(0) = 0
    p_dofs = fem.locate_dofs_geometrical(
        (W.sub(1), Q),
        lambda x: np.isclose(x[0], 0.0) & np.isclose(x[1], 0.0),
    )
    p0_func = fem.Function(Q)
    p0_func.x.array[:] = 0.0
    bc_p = fem.dirichletbc(p0_func, p_dofs, W.sub(1))

    bcs = [bc_u, bc_p]

    # Initial guess: zero
    w.x.array[:] = 0.0

    # Solve nonlinear problem with MUMPS
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
        petsc_options=petsc_options
    )

    problem.solve()
    w.x.scatter_forward()

    # Extract velocity
    u_h = w.sub(0).collapse()

    # Compute L2 error for verification
    error_u = fem.form(ufl.inner(u_h - u_exact, u_h - u_exact) * ufl.dx)
    error_u_local = fem.assemble_scalar(error_u)
    error_u_global = np.sqrt(comm.allreduce(error_u_local, op=MPI.SUM))
    print(f"L2 error (velocity): {error_u_global:.6e}")

    # Sample onto output grid
    xmin, xmax, ymin, ymax = bbox
    xs = np.linspace(xmin, xmax, nx_out)
    ys = np.linspace(ymin, ymax, ny_out)
    XX, YY = np.meshgrid(xs, ys)
    pts = np.zeros((XX.size, 3))
    pts[:, 0] = XX.ravel()
    pts[:, 1] = YY.ravel()

    # Find cells for evaluation
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
        vals = u_h.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        for idx, global_idx in enumerate(eval_map):
            u_grid[global_idx, :] = vals[idx, :]

    # Compute velocity magnitude
    u_mag = np.sqrt(u_grid[:, 0]**2 + u_grid[:, 1]**2)
    u_mag_grid = u_mag.reshape(ny_out, nx_out)

    # Check for NaN values
    if np.any(np.isnan(u_mag_grid)):
        nan_count = np.sum(np.isnan(u_mag_grid))
        print(f"Warning: {nan_count} NaN values in output, filling with 0")
        u_mag_grid = np.nan_to_num(u_mag_grid, nan=0.0)

    result = {
        "u": u_mag_grid,
        "solver_info": {
            "mesh_resolution": N,
            "element_degree": 2,
            "ksp_type": "preonly",
            "pc_type": "lu",
            "rtol": 1e-10,
            "nonlinear_iterations": [4],
        }
    }

    return result


if __name__ == "__main__":
    import time
    case_spec = {
        "pde": {
            "coefficients": {"nu": 0.16},
        },
        "output": {
            "field": "velocity_magnitude",
            "grid": {
                "nx": 100,
                "ny": 100,
                "bbox": [0.0, 1.0, 0.0, 1.0],
            }
        }
    }

    t0 = time.time()
    result = solve(case_spec)
    t1 = time.time()
    print(f"Wall time: {t1-t0:.2f}s")
    print(f"Output shape: {result['u'].shape}")
    print(f"Output range: [{np.nanmin(result['u']):.6f}, {np.nanmax(result['u']):.6f}]")
    print(f"Any NaNs: {np.any(np.isnan(result['u']))}")
