import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc


def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD

    # Extract parameters
    pde = case_spec.get("pde", {})
    params = pde.get("parameters", {})
    epsilon = float(params.get("epsilon", 0.0))
    beta_vec = params.get("beta", [10.0, 4.0])

    domain_spec = case_spec.get("domain", {})
    bounds = domain_spec.get("bounds", [[0, 1], [0, 1]])

    output = case_spec.get("output", {})
    nx_out = output.get("nx", 50)
    ny_out = output.get("ny", 50)

    degree = 2
    N = 32

    domain = mesh.create_rectangle(
        comm,
        [np.array([bounds[0][0], bounds[1][0]]), np.array([bounds[0][1], bounds[1][1]])],
        [N, N],
        cell_type=mesh.CellType.triangle,
    )

    V = fem.functionspace(domain, ("Lagrange", degree))

    x = ufl.SpatialCoordinate(domain)

    # Exact solution for BC and source
    u_exact_ufl = ufl.sin(ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])

    # Velocity field
    beta = ufl.as_vector([PETSc.ScalarType(beta_vec[0]), PETSc.ScalarType(beta_vec[1])])

    # Source term: -eps * laplacian(u_exact) + beta . grad(u_exact)
    grad_u_exact = ufl.grad(u_exact_ufl)

    if epsilon > 1e-14:
        eps_c = fem.Constant(domain, PETSc.ScalarType(epsilon))
        laplacian_u_exact = ufl.div(ufl.grad(u_exact_ufl))
        f = -eps_c * laplacian_u_exact + ufl.dot(beta, grad_u_exact)
    else:
        f = ufl.dot(beta, grad_u_exact)

    # Trial and test functions
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    # Standard Galerkin
    if epsilon > 1e-14:
        eps_c = fem.Constant(domain, PETSc.ScalarType(epsilon))
        a_std = eps_c * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx + ufl.dot(beta, ufl.grad(u)) * v * ufl.dx
    else:
        a_std = ufl.dot(beta, ufl.grad(u)) * v * ufl.dx

    L_std = f * v * ufl.dx

    # SUPG stabilization
    h = ufl.CellDiameter(domain)
    beta_norm = ufl.sqrt(ufl.dot(beta, beta))

    # SUPG stabilization parameter
    if epsilon > 1e-14:
        eps_c = fem.Constant(domain, PETSc.ScalarType(epsilon))
        Pe_cell = beta_norm * h / (2.0 * eps_c)
        tau = h / (2.0 * beta_norm) * (1.0 / ufl.tanh(Pe_cell) - 1.0 / Pe_cell)
    else:
        # Pure advection: tau = h / (2 * |beta|)
        tau = h / (2.0 * beta_norm)

    # SUPG test function modification
    v_supg = tau * ufl.dot(beta, ufl.grad(v))

    # The strong-form operator applied to u
    if epsilon > 1e-14:
        eps_c = fem.Constant(domain, PETSc.ScalarType(epsilon))
        Lu = -eps_c * ufl.div(ufl.grad(u)) + ufl.dot(beta, ufl.grad(u))
    else:
        Lu = ufl.dot(beta, ufl.grad(u))

    a_supg = Lu * v_supg * ufl.dx
    L_supg = f * v_supg * ufl.dx

    a = a_std + a_supg
    L = L_std + L_supg

    # Boundary conditions - all boundary
    tdim = domain.topology.dim
    fdim = tdim - 1

    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
    )

    u_bc = fem.Function(V)
    u_bc.interpolate(lambda x: np.sin(np.pi * x[0]) * np.sin(np.pi * x[1]))

    dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc = fem.dirichletbc(u_bc, dofs)

    # Solve
    problem = petsc.LinearProblem(
        a,
        L,
        bcs=[bc],
        petsc_options={
            "ksp_type": "gmres",
            "pc_type": "ilu",
            "ksp_rtol": "1e-10",
            "ksp_max_it": "5000",
        },
        petsc_options_prefix="supg_",
    )

    u_sol = problem.solve()

    iterations = problem.solver.getIterationNumber()

    # Evaluate on output grid
    x_out = np.linspace(bounds[0][0], bounds[0][1], nx_out)
    y_out = np.linspace(bounds[1][0], bounds[1][1], ny_out)
    X, Y = np.meshgrid(x_out, y_out, indexing="ij")

    points_2d = np.column_stack([X.ravel(), Y.ravel()])
    points_3d = np.zeros((points_2d.shape[0], 3))
    points_3d[:, :2] = points_2d

    # Point evaluation
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
        vals = u_sol.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u_values[eval_map] = vals.flatten()

    u_grid = u_values.reshape((nx_out, ny_out))

    solver_info = {
        "mesh_resolution": N,
        "element_degree": degree,
        "ksp_type": "gmres",
        "pc_type": "ilu",
        "rtol": 1e-10,
        "iterations": int(iterations),
    }

    return {
        "u": u_grid,
        "solver_info": solver_info,
    }


if __name__ == "__main__":
    import time

    case_spec = {
        "pde": {
            "type": "convection_diffusion",
            "parameters": {
                "epsilon": 0.0,
                "beta": [10.0, 4.0],
            },
        },
        "domain": {
            "type": "rectangle",
            "bounds": [[0, 1], [0, 1]],
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

    # Compute error against exact solution
    x_out = np.linspace(0, 1, 50)
    y_out = np.linspace(0, 1, 50)
    X, Y = np.meshgrid(x_out, y_out, indexing="ij")
    u_exact = np.sin(np.pi * X) * np.sin(np.pi * Y)

    error = np.sqrt(np.mean((u_grid - u_exact) ** 2))
    max_error = np.max(np.abs(u_grid - u_exact))

    print(f"Time: {elapsed:.3f}s")
    print(f"L2 error (grid): {error:.6e}")
    print(f"Max error: {max_error:.6e}")
    print(f"Solver info: {result['solver_info']}")
    print(f"NaN count: {np.sum(np.isnan(u_grid))}")
