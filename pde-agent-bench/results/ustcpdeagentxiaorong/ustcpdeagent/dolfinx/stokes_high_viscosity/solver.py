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
    nu_val = case_spec["pde"]["coefficients"]["nu"]
    output_grid = case_spec["output"]["grid"]
    nx_out = output_grid["nx"]
    ny_out = output_grid["ny"]
    bbox = output_grid["bbox"]
    xmin, xmax, ymin, ymax = bbox

    # P2/P1 Taylor-Hood with N=80
    N = 80
    degree_u = 2
    degree_p = 1

    # Create mesh
    msh = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
    gdim = msh.geometry.dim

    # Mixed function space (Taylor-Hood P2/P1)
    vel_el = basix_element("Lagrange", msh.topology.cell_name(), degree_u, shape=(gdim,))
    pres_el = basix_element("Lagrange", msh.topology.cell_name(), degree_p)
    W = fem.functionspace(msh, basix_mixed_element([vel_el, pres_el]))

    V, _ = W.sub(0).collapse()
    Q, _ = W.sub(1).collapse()

    # Trial and test functions
    (u, p) = ufl.TrialFunctions(W)
    (v, q) = ufl.TestFunctions(W)

    # Spatial coordinates
    x = ufl.SpatialCoordinate(msh)
    pi = ufl.pi

    # Manufactured solution (for BCs)
    u_exact = ufl.as_vector([
        pi * ufl.cos(pi * x[1]) * ufl.sin(pi * x[0]),
        -pi * ufl.cos(pi * x[0]) * ufl.sin(pi * x[1])
    ])
    p_exact = ufl.cos(pi * x[0]) * ufl.cos(pi * x[1])

    # Analytical source term: f = -nu * laplacian(u_exact) + grad(p_exact)
    f1 = 2.0 * nu_val * pi**3 * ufl.cos(pi * x[1]) * ufl.sin(pi * x[0]) - pi * ufl.sin(pi * x[0]) * ufl.cos(pi * x[1])
    f2 = -2.0 * nu_val * pi**3 * ufl.cos(pi * x[0]) * ufl.sin(pi * x[1]) - pi * ufl.cos(pi * x[0]) * ufl.sin(pi * x[1])
    f = ufl.as_vector([f1, f2])

    # Bilinear form for Stokes
    a = (nu_val * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
         - p * ufl.div(v) * ufl.dx
         + ufl.div(u) * q * ufl.dx)

    # Linear form
    L_form = ufl.inner(f, v) * ufl.dx

    # Boundary conditions - velocity on all boundaries
    fdim = msh.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(
        msh, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
    )

    u_bc = fem.Function(V)
    u_bc.interpolate(fem.Expression(u_exact, V.element.interpolation_points))
    dofs_u = fem.locate_dofs_topological((W.sub(0), V), fdim, boundary_facets)
    bc_u = fem.dirichletbc(u_bc, dofs_u, W.sub(0))

    # Pressure pin at origin: p_exact(0,0) = cos(0)*cos(0) = 1.0
    p_dofs = fem.locate_dofs_geometrical(
        (W.sub(1), Q),
        lambda x: np.isclose(x[0], 0.0) & np.isclose(x[1], 0.0)
    )
    p0_pin = fem.Function(Q)
    p0_pin.x.array[:] = 1.0
    bc_p = fem.dirichletbc(p0_pin, p_dofs, W.sub(1))

    bcs = [bc_u, bc_p]

    # Solve with MUMPS direct solver (handles saddle-point systems)
    ksp_type = "preonly"
    pc_type = "lu"

    problem = petsc.LinearProblem(
        a, L_form, bcs=bcs,
        petsc_options={
            "ksp_type": ksp_type,
            "pc_type": pc_type,
            "pc_factor_mat_solver_type": "mumps",
        },
        petsc_options_prefix="stokes_"
    )
    w_h = problem.solve()

    # Extract velocity
    u_h = w_h.sub(0).collapse()

    # Sample velocity magnitude on output grid
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

    u_grid = np.full(nx_out * ny_out, np.nan)
    if len(points_on_proc) > 0:
        pts_eval = np.array(points_on_proc)
        cells_eval = np.array(cells_on_proc, dtype=np.int32)
        u_vals = u_h.eval(pts_eval, cells_eval)
        magnitude = np.linalg.norm(u_vals, axis=1)
        for idx, global_idx in enumerate(eval_map):
            u_grid[global_idx] = magnitude[idx]

    u_grid = u_grid.reshape(ny_out, nx_out)

    solver_info = {
        "mesh_resolution": N,
        "element_degree": degree_u,
        "ksp_type": ksp_type,
        "pc_type": pc_type,
        "rtol": 1e-10,
        "iterations": 1,
    }

    return {
        "u": u_grid,
        "solver_info": solver_info,
    }


if __name__ == "__main__":
    case_spec = {
        "pde": {
            "coefficients": {"nu": 5.0},
            "time": None,
        },
        "output": {
            "field": "velocity_magnitude",
            "grid": {
                "nx": 100,
                "ny": 100,
                "bbox": [0.0, 1.0, 0.0, 1.0],
            },
        },
    }

    import time
    t0 = time.time()
    result = solve(case_spec)
    elapsed = time.time() - t0
    print(f"Wall time: {elapsed:.2f}s")
    print(f"Output shape: {result['u'].shape}")
    print(f"NaN count: {np.isnan(result['u']).sum()}")

    # Compare with exact solution
    nx, ny = 100, 100
    xs = np.linspace(0, 1, nx)
    ys = np.linspace(0, 1, ny)
    XX, YY = np.meshgrid(xs, ys)
    u1_exact = np.pi * np.cos(np.pi * YY) * np.sin(np.pi * XX)
    u2_exact = -np.pi * np.cos(np.pi * XX) * np.sin(np.pi * YY)
    mag_exact = np.sqrt(u1_exact**2 + u2_exact**2)

    error = np.abs(result['u'] - mag_exact)
    print(f"Max pointwise error: {np.nanmax(error):.2e}")
    print(f"Mean pointwise error: {np.nanmean(error):.2e}")
    print(f"RMS error: {np.sqrt(np.nanmean(error**2)):.2e}")
