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
    if "coefficients" in pde_config:
        kappa_val = pde_config["coefficients"].get("kappa", 1.0)

    # 2. Create mesh - use higher resolution for accuracy
    nx, ny = 80, 80
    domain = mesh.create_unit_square(MPI.COMM_WORLD, nx, ny, cell_type=mesh.CellType.triangle)

    # 3. Function space - degree 2 for better accuracy
    degree = 2
    V = fem.functionspace(domain, ("Lagrange", degree))

    # 4. Define exact solution and source term using UFL
    x = ufl.SpatialCoordinate(domain)
    pi = ufl.pi

    # Exact solution: u = exp(x*y)*sin(pi*x)*sin(pi*y)
    u_exact_ufl = ufl.exp(x[0] * x[1]) * ufl.sin(pi * x[0]) * ufl.sin(pi * x[1])

    # Compute source term: f = -kappa * div(grad(u_exact))
    kappa = fem.Constant(domain, default_scalar_type(kappa_val))
    f_ufl = -kappa * ufl.div(ufl.grad(u_exact_ufl))

    # 5. Variational problem
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    a = kappa * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = f_ufl * v * ufl.dx

    # 6. Boundary conditions
    # Interpolate exact solution for BC
    u_bc_func = fem.Function(V)
    u_exact_expr = fem.Expression(u_exact_ufl, V.element.interpolation_points)
    u_bc_func.interpolate(u_exact_expr)

    tdim = domain.topology.dim
    fdim = tdim - 1

    # All boundary facets
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
    )
    dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc = fem.dirichletbc(u_bc_func, dofs)

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
            "ksp_atol": "1e-14",
            "ksp_max_it": "2000",
        },
        petsc_options_prefix="poisson_"
    )
    uh = problem.solve()

    # Get iteration count
    ksp = problem.solver
    iterations = ksp.getIterationNumber()

    # 8. Extract solution on 50x50 uniform grid
    n_eval = 50
    xs = np.linspace(0.0, 1.0, n_eval)
    ys = np.linspace(0.0, 1.0, n_eval)
    XX, YY = np.meshgrid(xs, ys, indexing='ij')

    # Build points array (3, N)
    points_flat = np.zeros((3, n_eval * n_eval))
    points_flat[0, :] = XX.ravel()
    points_flat[1, :] = YY.ravel()

    bb_tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, points_flat.T)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points_flat.T)

    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    for i in range(points_flat.shape[1]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(points_flat[:, i])
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