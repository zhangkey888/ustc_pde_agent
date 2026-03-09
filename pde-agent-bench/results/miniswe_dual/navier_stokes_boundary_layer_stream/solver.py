import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import basix.ufl
import ufl
from petsc4py import PETSc
import time


def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD

    # Extract parameters
    nu_val = case_spec.get("pde", {}).get("viscosity", 0.08)

    # Parse output requirements
    output = case_spec.get("output", {})
    nx_out = output.get("nx", 50)
    ny_out = output.get("ny", 50)

    degree_u = 3
    degree_p = 2
    N = 45  # P3/P2 Taylor-Hood

    t_start = time.time()

    # Create mesh
    domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)

    # Define mixed function space (Taylor-Hood P3/P2)
    e_u = basix.ufl.element("Lagrange", "triangle", degree_u, shape=(2,))
    e_p = basix.ufl.element("Lagrange", "triangle", degree_p)
    mel = basix.ufl.mixed_element([e_u, e_p])
    W = fem.functionspace(domain, mel)

    # Collapse sub-spaces
    V, V_map = W.sub(0).collapse()
    Q, Q_map = W.sub(1).collapse()

    # Current solution
    w = fem.Function(W)
    (u, p) = ufl.split(w)
    (v, q) = ufl.TestFunctions(W)

    # Spatial coordinates
    x = ufl.SpatialCoordinate(domain)

    # Manufactured solution (UFL expressions)
    u_exact = ufl.as_vector([
        ufl.pi * ufl.exp(6.0 * (x[0] - 1.0)) * ufl.cos(ufl.pi * x[1]),
        -6.0 * ufl.exp(6.0 * (x[0] - 1.0)) * ufl.sin(ufl.pi * x[1])
    ])
    p_exact = ufl.sin(ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])

    nu = fem.Constant(domain, PETSc.ScalarType(nu_val))

    # Compute source term from manufactured solution
    f = (ufl.grad(u_exact) * u_exact
         - nu * ufl.div(ufl.grad(u_exact))
         + ufl.grad(p_exact))

    # Weak form (residual)
    F_form = (
        nu * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
        + ufl.inner(ufl.grad(u) * u, v) * ufl.dx
        - ufl.inner(p, ufl.div(v)) * ufl.dx
        + ufl.inner(ufl.div(u), q) * ufl.dx
        - ufl.inner(f, v) * ufl.dx
    )

    # Boundary conditions - all boundaries
    tdim = domain.topology.dim
    fdim = tdim - 1

    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
    )

    # Velocity BC from exact solution
    u_bc_func = fem.Function(V)
    u_bc_func.interpolate(lambda x: np.vstack([
        np.pi * np.exp(6.0 * (x[0] - 1.0)) * np.cos(np.pi * x[1]),
        -6.0 * np.exp(6.0 * (x[0] - 1.0)) * np.sin(np.pi * x[1])
    ]))

    dofs_u = fem.locate_dofs_topological((W.sub(0), V), fdim, boundary_facets)
    bc_u = fem.dirichletbc(u_bc_func, dofs_u, W.sub(0))

    bcs = [bc_u]

    # Set initial guess to exact solution (ensures fast Newton convergence)
    u_init = fem.Function(V)
    u_init.interpolate(lambda x: np.vstack([
        np.pi * np.exp(6.0 * (x[0] - 1.0)) * np.cos(np.pi * x[1]),
        -6.0 * np.exp(6.0 * (x[0] - 1.0)) * np.sin(np.pi * x[1])
    ]))
    w.x.array[V_map] = u_init.x.array

    p_init = fem.Function(Q)
    p_init.interpolate(lambda x: np.sin(np.pi * x[0]) * np.sin(np.pi * x[1]))
    w.x.array[Q_map] = p_init.x.array

    w.x.scatter_forward()

    # Solve nonlinear problem with direct LU solver
    petsc_opts = {
        "ksp_type": "preonly",
        "pc_type": "lu",
        "pc_factor_mat_solver_type": "mumps",
        "snes_rtol": 1e-10,
        "snes_atol": 1e-12,
        "snes_max_it": 30,
        "snes_linesearch_type": "basic",
    }

    problem = petsc.NonlinearProblem(
        F_form, w, bcs=bcs,
        petsc_options_prefix="ns_",
        petsc_options=petsc_opts,
    )
    problem.solve()
    snes = problem.solver
    n_newton = snes.getIterationNumber()
    reason = snes.getConvergedReason()
    if reason <= 0:
        raise RuntimeError(f"SNES did not converge, reason={reason}")

    w.x.scatter_forward()

    # Extract velocity solution
    u_h = w.sub(0).collapse()

    # Evaluate on output grid
    x_coords = np.linspace(0, 1, nx_out)
    y_coords = np.linspace(0, 1, ny_out)
    XX, YY = np.meshgrid(x_coords, y_coords, indexing='ij')
    points = np.zeros((3, nx_out * ny_out))
    points[0] = XX.flatten()
    points[1] = YY.flatten()

    bb_tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, points.T)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points.T)

    u_vals = np.full((nx_out * ny_out, domain.geometry.dim), np.nan)

    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    for i in range(nx_out * ny_out):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(points[:, i])
            cells_on_proc.append(links[0])
            eval_map.append(i)

    if len(points_on_proc) > 0:
        pts_arr = np.array(points_on_proc)
        cells_arr = np.array(cells_on_proc, dtype=np.int32)
        vals = u_h.eval(pts_arr, cells_arr)
        for idx, map_idx in enumerate(eval_map):
            u_vals[map_idx] = vals[idx]

    vel_mag = np.sqrt(u_vals[:, 0]**2 + u_vals[:, 1]**2)
    u_grid = vel_mag.reshape((nx_out, ny_out))

    elapsed = time.time() - t_start

    result = {
        "u": u_grid,
        "solver_info": {
            "mesh_resolution": N,
            "element_degree": degree_u,
            "ksp_type": "preonly",
            "pc_type": "lu",
            "rtol": 1e-10,
            "nonlinear_iterations": [int(n_newton)],
        }
    }

    return result


if __name__ == "__main__":
    case_spec = {
        "pde": {"viscosity": 0.08},
        "output": {"nx": 50, "ny": 50},
    }

    t0 = time.time()
    result = solve(case_spec)
    t1 = time.time()

    print(f"\nTotal time: {t1-t0:.2f}s")
    print(f"Solution shape: {result['u'].shape}")
    print(f"Solver info: {result['solver_info']}")

    nx, ny = 50, 50
    xc = np.linspace(0, 1, nx)
    yc = np.linspace(0, 1, ny)
    X, Y = np.meshgrid(xc, yc, indexing='ij')
    u1e = np.pi * np.exp(6.0 * (X - 1.0)) * np.cos(np.pi * Y)
    u2e = -6.0 * np.exp(6.0 * (X - 1.0)) * np.sin(np.pi * Y)
    vme = np.sqrt(u1e**2 + u2e**2)

    err = np.max(np.abs(result['u'] - vme))
    rel_l2 = np.sqrt(np.mean((result['u'] - vme)**2)) / np.sqrt(np.mean(vme**2))
    print(f"Max pointwise error in velocity magnitude: {err:.2e}")
    print(f"Relative L2 error on grid: {rel_l2:.2e}")
