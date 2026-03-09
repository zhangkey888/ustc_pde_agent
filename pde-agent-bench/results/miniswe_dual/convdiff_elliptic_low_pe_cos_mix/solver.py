import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc


def solve(case_spec: dict) -> dict:
    """Solve convection-diffusion equation: -eps*laplacian(u) + beta.grad(u) = f."""

    # Extract parameters robustly
    pde = case_spec.get("pde", {})
    params = pde.get("parameters", pde.get("params", {}))
    epsilon = params.get("epsilon", 0.2)
    beta_vec = params.get("beta", [0.8, 0.3])

    domain_spec = case_spec.get("domain", {})
    bounds = domain_spec.get("bounds", [[0, 1], [0, 1]])
    x_range = domain_spec.get("x_range", [bounds[0][0], bounds[0][1]])
    y_range = domain_spec.get("y_range", [bounds[1][0], bounds[1][1]])

    output = case_spec.get("output", {})
    nx_out = output.get("nx", 50)
    ny_out = output.get("ny", 50)

    # Fixed parameters - P2 on N=48 is already very accurate for this problem
    N = 48
    element_degree = 2
    ksp_type = "gmres"
    pc_type = "hypre"
    rtol = 1e-8

    comm = MPI.COMM_WORLD
    domain = mesh.create_rectangle(
        comm,
        [np.array([x_range[0], y_range[0]]), np.array([x_range[1], y_range[1]])],
        [N, N],
        cell_type=mesh.CellType.triangle,
    )

    V = fem.functionspace(domain, ("Lagrange", element_degree))

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    x = ufl.SpatialCoordinate(domain)

    # Manufactured solution: u_exact = cos(pi*x)*sin(pi*y)
    u_exact_ufl = ufl.cos(ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])

    # Source term: f = -eps * laplacian(u_exact) + beta . grad(u_exact)
    # laplacian(u_exact) = -2*pi^2*cos(pi*x)*sin(pi*y)
    # => -eps*laplacian = 2*eps*pi^2*cos(pi*x)*sin(pi*y)
    # grad(u_exact) = (-pi*sin(pi*x)*sin(pi*y), pi*cos(pi*x)*cos(pi*y))
    # beta.grad = beta[0]*(-pi*sin(pi*x)*sin(pi*y)) + beta[1]*(pi*cos(pi*x)*cos(pi*y))
    f_expr = (
        epsilon * 2.0 * ufl.pi ** 2 * ufl.cos(ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])
        + beta_vec[0] * (-ufl.pi * ufl.sin(ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1]))
        + beta_vec[1] * (ufl.pi * ufl.cos(ufl.pi * x[0]) * ufl.cos(ufl.pi * x[1]))
    )

    beta = ufl.as_vector(
        [
            fem.Constant(domain, PETSc.ScalarType(beta_vec[0])),
            fem.Constant(domain, PETSc.ScalarType(beta_vec[1])),
        ]
    )
    eps_const = fem.Constant(domain, PETSc.ScalarType(epsilon))

    # SUPG stabilization parameter
    h = ufl.CellDiameter(domain)
    beta_norm = ufl.sqrt(ufl.dot(beta, beta))
    Pe_cell = beta_norm * h / (2.0 * eps_const)
    tau = h / (2.0 * beta_norm) * (1.0 / ufl.tanh(Pe_cell) - 1.0 / Pe_cell)

    # Standard Galerkin bilinear form
    a_std = (
        eps_const * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
        + ufl.inner(ufl.dot(beta, ufl.grad(u)), v) * ufl.dx
    )
    L_std = f_expr * v * ufl.dx

    # SUPG additional terms
    v_supg = tau * ufl.dot(beta, ufl.grad(v))
    a_supg = a_std + ufl.inner(ufl.dot(beta, ufl.grad(u)), v_supg) * ufl.dx
    L_supg = L_std + ufl.inner(f_expr, v_supg) * ufl.dx

    # Boundary conditions - all boundaries
    tdim = domain.topology.dim
    fdim = tdim - 1

    boundary_facets = mesh.locate_entities_boundary(
        domain,
        fdim,
        lambda xx: (
            np.isclose(xx[0], x_range[0])
            | np.isclose(xx[0], x_range[1])
            | np.isclose(xx[1], y_range[0])
            | np.isclose(xx[1], y_range[1])
        ),
    )

    u_bc = fem.Function(V)
    u_bc.interpolate(lambda xx: np.cos(np.pi * xx[0]) * np.sin(np.pi * xx[1]))

    dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc = fem.dirichletbc(u_bc, dofs)

    # Solve with iterative solver, fallback to direct if needed
    try:
        problem = petsc.LinearProblem(
            a_supg,
            L_supg,
            bcs=[bc],
            petsc_options={
                "ksp_type": ksp_type,
                "pc_type": pc_type,
                "ksp_rtol": str(rtol),
                "ksp_max_it": "2000",
            },
            petsc_options_prefix="cdiff_",
        )
        u_sol = problem.solve()
        iterations = problem.solver.getIterationNumber()
    except Exception:
        ksp_type = "preonly"
        pc_type = "lu"
        problem = petsc.LinearProblem(
            a_supg,
            L_supg,
            bcs=[bc],
            petsc_options={"ksp_type": ksp_type, "pc_type": pc_type},
            petsc_options_prefix="cdiff_",
        )
        u_sol = problem.solve()
        iterations = 1

    # Evaluate on output grid
    x_coords = np.linspace(x_range[0], x_range[1], nx_out)
    y_coords = np.linspace(y_range[0], y_range[1], ny_out)
    X, Y = np.meshgrid(x_coords, y_coords, indexing="ij")
    points_2d = np.column_stack([X.ravel(), Y.ravel()])
    points_3d = np.zeros((points_2d.shape[0], 3))
    points_3d[:, :2] = points_2d

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

    solver_info = {
        "mesh_resolution": N,
        "element_degree": element_degree,
        "ksp_type": ksp_type,
        "pc_type": pc_type,
        "rtol": rtol,
        "iterations": iterations,
    }

    return {"u": u_grid, "solver_info": solver_info}
