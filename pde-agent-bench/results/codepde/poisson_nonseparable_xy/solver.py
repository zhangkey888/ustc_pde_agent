import numpy as np
from dolfinx import mesh, fem, default_scalar_type, geometry
from dolfinx.fem.petsc import LinearProblem
from mpi4py import MPI
import ufl
from petsc4py import PETSc


def solve(case_spec: dict) -> dict:
    # 1. Parse case_spec
    pde_config = case_spec.get("pde", case_spec.get("oracle_config", {}).get("pde", {}))
    kappa = 1.0

    # 2. Create mesh - use higher resolution for accuracy with non-separable solution
    nx, ny = 80, 80
    degree = 2
    domain = mesh.create_unit_square(MPI.COMM_WORLD, nx, ny, cell_type=mesh.CellType.triangle)

    # 3. Function space
    V = fem.functionspace(domain, ("Lagrange", degree))

    # 4. Define variational problem
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    x = ufl.SpatialCoordinate(domain)

    # Manufactured solution: u_exact = sin(pi*x*y)
    # -kappa * laplacian(sin(pi*x*y)) = f
    # grad(sin(pi*x*y)) = pi*y*cos(pi*x*y) e_x + pi*x*cos(pi*x*y) e_y
    # d/dx(pi*y*cos(pi*x*y)) = -pi^2*y^2*sin(pi*x*y)
    # d/dy(pi*x*cos(pi*x*y)) = pi*cos(pi*x*y) - pi^2*x^2*y*sin(pi*x*y)
    #   Wait, let me redo:
    # d/dy(pi*x*cos(pi*x*y)) = -pi*x * pi*x * sin(pi*x*y) = -pi^2*x^2*sin(pi*x*y)
    # Actually: d/dy[pi*x*cos(pi*x*y)] = pi*x * (-sin(pi*x*y)) * pi*x = -pi^2*x^2*sin(pi*x*y)
    # Wait no: d/dx[pi*y*cos(pi*x*y)] = pi*y*(-sin(pi*x*y))*pi*y = -pi^2*y^2*sin(pi*x*y)
    # laplacian = -pi^2*y^2*sin(pi*x*y) - pi^2*x^2*sin(pi*x*y) = -pi^2*(x^2+y^2)*sin(pi*x*y)
    # f = -kappa * laplacian = kappa * pi^2*(x^2+y^2)*sin(pi*x*y)

    pi = ufl.pi
    u_exact_ufl = ufl.sin(pi * x[0] * x[1])
    f_expr = kappa * pi**2 * (x[0]**2 + x[1]**2) * ufl.sin(pi * x[0] * x[1])

    a = kappa * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = f_expr * v * ufl.dx

    # 5. Boundary conditions - u = g = sin(pi*x*y) on boundary
    u_bc = fem.Function(V)
    u_bc.interpolate(lambda X: np.sin(np.pi * X[0] * X[1]))

    tdim = domain.topology.dim
    fdim = tdim - 1
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda X: np.ones(X.shape[1], dtype=bool)
    )
    dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc = fem.dirichletbc(u_bc, dofs)

    # 6. Solve
    ksp_type = "cg"
    pc_type = "hypre"
    rtol = 1e-10

    problem = LinearProblem(
        a, L, bcs=[bc],
        petsc_options={
            "ksp_type": ksp_type,
            "pc_type": pc_type,
            "ksp_rtol": rtol,
        },
        petsc_options_prefix="poisson_"
    )
    uh = problem.solve()

    # 7. Extract on 50x50 uniform grid
    n_eval = 50
    xs = np.linspace(0, 1, n_eval)
    ys = np.linspace(0, 1, n_eval)
    XX, YY = np.meshgrid(xs, ys, indexing='ij')
    points = np.zeros((3, n_eval * n_eval))
    points[0] = XX.ravel()
    points[1] = YY.ravel()

    bb_tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, points.T)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points.T)

    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    for i in range(points.shape[1]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(points.T[i])
            cells_on_proc.append(links[0])
            eval_map.append(i)

    u_values = np.full(n_eval * n_eval, np.nan)
    if len(points_on_proc) > 0:
        vals = uh.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
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
            "iterations": problem.solver.getIterationNumber() if hasattr(problem, 'solver') else -1,
        }
    }