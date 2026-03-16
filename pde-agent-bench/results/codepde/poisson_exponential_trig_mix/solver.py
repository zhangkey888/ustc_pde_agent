import numpy as np
from dolfinx import mesh, fem, default_scalar_type, geometry
from dolfinx.fem.petsc import LinearProblem
from mpi4py import MPI
import ufl
from petsc4py import PETSc


def solve(case_spec: dict) -> dict:
    # 1. Parse case_spec
    pde_config = case_spec.get("oracle_config", {}).get("pde", {})
    kappa_val = 1.0
    for coeff in pde_config.get("coefficients", []):
        if coeff.get("name") == "kappa":
            kappa_val = float(coeff["value"])

    # 2. Create mesh
    nx, ny = 80, 80
    degree = 2
    domain = mesh.create_unit_square(MPI.COMM_WORLD, nx, ny, cell_type=mesh.CellType.triangle)

    # 3. Function space
    V = fem.functionspace(domain, ("Lagrange", degree))

    # 4. Define exact solution and source term using UFL
    x = ufl.SpatialCoordinate(domain)
    pi = ufl.pi

    # Exact solution: u = exp(2*x)*cos(pi*y)
    u_exact_ufl = ufl.exp(2.0 * x[0]) * ufl.cos(pi * x[1])

    # Compute source term: f = -kappa * laplacian(u_exact)
    # grad(u_exact) = (2*exp(2x)*cos(pi*y), -pi*exp(2x)*sin(pi*y))
    # laplacian = 4*exp(2x)*cos(pi*y) - pi^2*exp(2x)*cos(pi*y) = (4 - pi^2)*exp(2x)*cos(pi*y)
    # f = -kappa * (4 - pi^2) * exp(2x)*cos(pi*y)
    kappa = fem.Constant(domain, default_scalar_type(kappa_val))
    f_expr = -kappa * ((4.0 - pi**2) * ufl.exp(2.0 * x[0]) * ufl.cos(pi * x[1]))
    # Since -kappa * laplacian(u) = f, we have f = -kappa * laplacian(u_exact)
    # But the PDE is -div(kappa grad u) = f, so f = -kappa * laplacian(u) for constant kappa
    # f = -kappa * (4 - pi^2) * exp(2x)*cos(pi*y)
    f_source = -kappa * (4.0 - pi**2) * ufl.exp(2.0 * x[0]) * ufl.cos(pi * x[1])

    # 5. Variational problem
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    a = kappa * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = f_source * v * ufl.dx

    # 6. Boundary conditions
    tdim = domain.topology.dim
    fdim = tdim - 1

    # All boundaries
    domain.topology.create_connectivity(fdim, tdim)
    boundary_facets = mesh.exterior_facets(domain.topology)

    u_bc = fem.Function(V)
    u_bc.interpolate(lambda x: np.exp(2.0 * x[0]) * np.cos(np.pi * x[1]))

    dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc = fem.dirichletbc(u_bc, dofs)

    # 7. Solve
    ksp_type = "cg"
    pc_type = "hypre"
    rtol = 1e-10

    problem = LinearProblem(
        a, L, bcs=[bc],
        petsc_options={
            "ksp_type": ksp_type,
            "pc_type": pc_type,
            "ksp_rtol": str(rtol),
            "ksp_max_it": "1000",
        },
        petsc_options_prefix="poisson_"
    )
    uh = problem.solve()

    # Get iteration count
    iterations = problem.solver.getIterationNumber()

    # 8. Extract on 50x50 uniform grid
    n_eval = 50
    xs = np.linspace(0.0, 1.0, n_eval)
    ys = np.linspace(0.0, 1.0, n_eval)
    XX, YY = np.meshgrid(xs, ys, indexing='ij')
    points = np.zeros((3, n_eval * n_eval))
    points[0, :] = XX.ravel()
    points[1, :] = YY.ravel()

    bb_tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, points.T)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points.T)

    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    for i in range(points.shape[1]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(points[:, i])
            cells_on_proc.append(links[0])
            eval_map.append(i)

    u_values = np.full(n_eval * n_eval, np.nan)
    if len(points_on_proc) > 0:
        pts_arr = np.array(points_on_proc)
        cells_arr = np.array(cells_on_proc, dtype=np.int32)
        vals = uh.eval(pts_arr, cells_arr)
        u_values[eval_map] = vals.flatten()

    u_grid = u_values.reshape((n_eval, n_eval))

    return {
        "u": u_grid,
        "solver_info": {
            "mesh_resolution": nx,
            "element_degree": degree,
            "ksp_type": ksp_type,
            "pc_type": pc_type,
            "rtol": rtol,
            "iterations": iterations,
        }
    }