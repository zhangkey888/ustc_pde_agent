import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc

ScalarType = PETSc.ScalarType


def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD

    # Parse parameters
    pde = case_spec.get("pde", {})
    params = pde.get("parameters", {})
    epsilon = params.get("epsilon", 0.01)
    beta_vec = params.get("beta", [14.0, 6.0])

    domain_spec = case_spec.get("domain", {})
    output = case_spec.get("output", {})
    nx_out = output.get("nx", 50)
    ny_out = output.get("ny", 50)

    # Use degree 2 on quads with SUPG for high Peclet
    element_degree = 2
    N = 48  # Reduced from 80 - still plenty of accuracy headroom

    # Create quadrilateral mesh
    domain = mesh.create_rectangle(
        comm,
        [np.array([0.0, 0.0]), np.array([1.0, 1.0])],
        [N, N],
        cell_type=mesh.CellType.quadrilateral,
    )

    # Function space - Lagrange on quads
    V = fem.functionspace(domain, ("Lagrange", element_degree))

    # Spatial coordinate
    x = ufl.SpatialCoordinate(domain)

    # Source term derived from manufactured solution
    # u = sin(pi*x)*sin(pi*y)
    # f = -eps * laplacian(u) + beta . grad(u)
    # = 2*eps*pi^2*sin(pi*x)*sin(pi*y) + beta[0]*pi*cos(pi*x)*sin(pi*y) + beta[1]*pi*sin(pi*x)*cos(pi*y)
    pi_ = ufl.pi
    f_expr = (
        2.0 * epsilon * pi_**2 * ufl.sin(pi_ * x[0]) * ufl.sin(pi_ * x[1])
        + beta_vec[0] * pi_ * ufl.cos(pi_ * x[0]) * ufl.sin(pi_ * x[1])
        + beta_vec[1] * pi_ * ufl.sin(pi_ * x[0]) * ufl.cos(pi_ * x[1])
    )

    # Beta as UFL vector
    beta = ufl.as_vector([ScalarType(beta_vec[0]), ScalarType(beta_vec[1])])

    # Trial and test functions
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    # Standard Galerkin weak form
    a_standard = (
        epsilon * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
        + ufl.inner(ufl.dot(beta, ufl.grad(u)), v) * ufl.dx
    )
    L_standard = ufl.inner(f_expr, v) * ufl.dx

    # SUPG stabilization
    h = ufl.CellDiameter(domain)
    beta_norm = ufl.sqrt(ufl.dot(beta, beta))

    # SUPG parameter: tau = h / (2*|beta|) for convection-dominated regime
    tau = h / (2.0 * beta_norm + 1e-15)

    # SUPG test function modification
    v_supg = tau * ufl.dot(beta, ufl.grad(v))

    # Strong residual applied to trial function
    R_u = -epsilon * ufl.div(ufl.grad(u)) + ufl.dot(beta, ufl.grad(u))

    a_supg = ufl.inner(R_u, v_supg) * ufl.dx
    L_supg = ufl.inner(f_expr, v_supg) * ufl.dx

    # Total forms
    a_total = a_standard + a_supg
    L_total = L_standard + L_supg

    # Boundary conditions - u = 0 on boundary (exact solution vanishes there)
    tdim = domain.topology.dim
    fdim = tdim - 1

    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
    )
    dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)

    u_bc = fem.Function(V)
    u_bc.interpolate(lambda x: np.zeros_like(x[0]))
    bc = fem.dirichletbc(u_bc, dofs)

    # Solve with GMRES + ILU
    ksp_type = "gmres"
    pc_type = "ilu"

    problem = petsc.LinearProblem(
        a_total, L_total, bcs=[bc],
        petsc_options={
            "ksp_type": ksp_type,
            "pc_type": pc_type,
            "ksp_rtol": "1e-10",
            "ksp_atol": "1e-12",
            "ksp_max_it": "5000",
        },
        petsc_options_prefix="cdiff_",
    )
    u_sol = problem.solve()

    # Get iteration count
    iterations = problem.solver.getIterationNumber()

    # Evaluate on output grid
    x_out = np.linspace(0, 1, nx_out)
    y_out = np.linspace(0, 1, ny_out)
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
        vals = u_sol.eval(
            np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32)
        )
        u_values[eval_map] = vals.flatten()

    u_grid = u_values.reshape((nx_out, ny_out))

    return {
        "u": u_grid,
        "solver_info": {
            "mesh_resolution": N,
            "element_degree": element_degree,
            "ksp_type": ksp_type,
            "pc_type": pc_type,
            "rtol": 1e-10,
            "iterations": int(iterations),
        },
    }


if __name__ == "__main__":
    import time

    case_spec = {
        "pde": {
            "type": "convection_diffusion",
            "parameters": {
                "epsilon": 0.01,
                "beta": [14.0, 6.0],
            },
        },
        "domain": {"type": "unit_square"},
        "output": {"nx": 50, "ny": 50},
    }

    t0 = time.time()
    result = solve(case_spec)
    elapsed = time.time() - t0

    u_grid = result["u"]
    print(f"Solve time: {elapsed:.3f}s")
    print(f"Solution shape: {u_grid.shape}")
    print(f"Solution range: [{np.nanmin(u_grid):.6f}, {np.nanmax(u_grid):.6f}]")
    print(f"NaN count: {np.isnan(u_grid).sum()}")
    print(f"Solver info: {result['solver_info']}")

    # Compute error against exact solution
    x_out = np.linspace(0, 1, 50)
    y_out = np.linspace(0, 1, 50)
    X, Y = np.meshgrid(x_out, y_out, indexing="ij")
    u_exact = np.sin(np.pi * X) * np.sin(np.pi * Y)

    mask = ~np.isnan(u_grid)
    error = np.sqrt(np.mean((u_grid[mask] - u_exact[mask]) ** 2))
    max_error = np.max(np.abs(u_grid[mask] - u_exact[mask]))
    print(f"RMS error: {error:.6e}")
    print(f"Max error: {max_error:.6e}")
