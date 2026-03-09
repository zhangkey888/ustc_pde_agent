import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc
import time

ScalarType = PETSc.ScalarType


def solve(case_spec: dict) -> dict:
    t_start = time.time()

    # Parse parameters
    pde = case_spec.get("pde", {})
    params = pde.get("parameters", {})
    epsilon = params.get("epsilon", 0.01)
    beta_vec = params.get("beta", [12.0, 4.0])

    output = case_spec.get("output", {})
    nx_out = output.get("nx", 50)
    ny_out = output.get("ny", 50)

    # P3 elements without SUPG - the manufactured solution is smooth
    # so standard Galerkin with P3 on a moderate mesh is sufficient
    element_degree = 3
    N = 48

    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", element_degree))

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    x = ufl.SpatialCoordinate(domain)

    # Velocity field
    beta = ufl.as_vector([beta_vec[0], beta_vec[1]])

    # Source term from manufactured solution: -epsilon * laplacian(u) + beta . grad(u) = f
    # u_exact = sin(pi*x)*sin(2*pi*y)
    # laplacian(u) = -(pi^2 + 4*pi^2)*sin(pi*x)*sin(2*pi*y) = -5*pi^2*u
    # -epsilon * laplacian(u) = 5*epsilon*pi^2*sin(pi*x)*sin(2*pi*y)
    # beta . grad(u) = beta[0]*pi*cos(pi*x)*sin(2*pi*y) + beta[1]*2*pi*sin(pi*x)*cos(2*pi*y)
    f_expr = (
        epsilon * 5.0 * ufl.pi**2 * ufl.sin(ufl.pi * x[0]) * ufl.sin(2 * ufl.pi * x[1])
        + beta_vec[0] * ufl.pi * ufl.cos(ufl.pi * x[0]) * ufl.sin(2 * ufl.pi * x[1])
        + beta_vec[1] * 2.0 * ufl.pi * ufl.sin(ufl.pi * x[0]) * ufl.cos(2 * ufl.pi * x[1])
    )

    # Standard Galerkin weak form
    a_form = (
        epsilon * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
        + ufl.inner(ufl.dot(beta, ufl.grad(u)), v) * ufl.dx
    )
    L_form = ufl.inner(f_expr, v) * ufl.dx

    # Boundary conditions: u = 0 on all boundaries
    # (sin(pi*x)*sin(2*pi*y) = 0 on boundary of [0,1]^2)
    tdim = domain.topology.dim
    fdim = tdim - 1
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
    )
    u_bc = fem.Function(V)
    u_bc.interpolate(lambda x: np.zeros(x.shape[1]))
    dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc = fem.dirichletbc(u_bc, dofs)

    # Solve with direct solver
    ksp_type = "preonly"
    pc_type = "lu"

    problem = petsc.LinearProblem(
        a_form, L_form, bcs=[bc],
        petsc_options={
            "ksp_type": ksp_type,
            "pc_type": pc_type,
        },
        petsc_options_prefix="cdiff_"
    )
    u_sol = problem.solve()

    # Evaluate on output grid
    u_grid = _evaluate_on_grid(domain, u_sol, nx_out, ny_out)

    solver_info = {
        "mesh_resolution": N,
        "element_degree": element_degree,
        "ksp_type": ksp_type,
        "pc_type": pc_type,
        "rtol": 1e-10,
        "iterations": 1,
    }

    return {
        "u": u_grid,
        "solver_info": solver_info,
    }


def _evaluate_on_grid(domain, u_func, nx, ny):
    """Evaluate solution on a uniform grid."""
    xv = np.linspace(0, 1, nx)
    yv = np.linspace(0, 1, ny)
    xx, yy = np.meshgrid(xv, yv, indexing='ij')

    points_3d = np.zeros((nx * ny, 3))
    points_3d[:, 0] = xx.ravel()
    points_3d[:, 1] = yy.ravel()

    bb_tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, points_3d)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points_3d)

    u_values = np.full(points_3d.shape[0], np.nan)
    points_on_proc = []
    cells_on_proc = []
    eval_map = []

    for i in range(points_3d.shape[0]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(points_3d[i])
            cells_on_proc.append(links[0])
            eval_map.append(i)

    if len(points_on_proc) > 0:
        pts_arr = np.array(points_on_proc)
        cells_arr = np.array(cells_on_proc, dtype=np.int32)
        vals = u_func.eval(pts_arr, cells_arr)
        u_values[eval_map] = vals.flatten()

    u_values = np.nan_to_num(u_values, nan=0.0)
    u_grid = u_values.reshape((nx, ny))
    return u_grid


if __name__ == "__main__":
    case_spec = {
        "pde": {
            "type": "convection_diffusion",
            "parameters": {
                "epsilon": 0.01,
                "beta": [12.0, 4.0],
            },
        },
        "domain": {
            "type": "unit_square",
            "x_range": [0, 1],
            "y_range": [0, 1],
        },
        "output": {
            "nx": 50,
            "ny": 50,
        },
    }

    t0 = time.time()
    result = solve(case_spec)
    elapsed = time.time() - t0

    u_grid = result["u"]
    print(f"Solution shape: {u_grid.shape}")
    print(f"Time: {elapsed:.3f}s")

    xv = np.linspace(0, 1, 50)
    yv = np.linspace(0, 1, 50)
    xx, yy = np.meshgrid(xv, yv, indexing='ij')
    u_exact = np.sin(np.pi * xx) * np.sin(2 * np.pi * yy)

    rms_error = np.sqrt(np.mean((u_grid - u_exact) ** 2))
    max_error = np.max(np.abs(u_grid - u_exact))
    l2_rel = np.sqrt(np.sum((u_grid - u_exact) ** 2) / np.sum(u_exact ** 2))
    print(f"RMS error: {rms_error:.2e}")
    print(f"Max error: {max_error:.2e}")
    print(f"L2 rel error: {l2_rel:.2e}")
    print(f"Solver info: {result['solver_info']}")
