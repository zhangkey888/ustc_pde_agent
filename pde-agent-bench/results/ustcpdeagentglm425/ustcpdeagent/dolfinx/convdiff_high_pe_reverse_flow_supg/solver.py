import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc


def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD

    # PDE parameters
    epsilon = 0.01
    beta = np.array([-12.0, 6.0], dtype=PETSc.ScalarType)
    beta_norm = np.linalg.norm(beta)

    # Output grid info
    nx_out = case_spec["output"]["grid"]["nx"]
    ny_out = case_spec["output"]["grid"]["ny"]
    bbox = case_spec["output"]["grid"]["bbox"]

    # Mesh resolution and element degree
    N = 160
    degree = 2

    # Create mesh
    domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)

    # Function space
    V = fem.functionspace(domain, ("Lagrange", degree))

    # Manufactured solution and source term (symbolic)
    x = ufl.SpatialCoordinate(domain)
    u_exact_ufl = ufl.exp(x[0]) * ufl.sin(ufl.pi * x[1])
    grad_u_ex = ufl.grad(u_exact_ufl)
    laplacian_u_ex = ufl.div(grad_u_ex)
    beta_ufl = ufl.as_vector(beta)
    f_ufl = -epsilon * laplacian_u_ex + ufl.dot(beta_ufl, grad_u_ex)

    # Dirichlet BC on entire boundary
    fdim = domain.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
    )
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)

    u_bc = fem.Function(V)
    u_bc.interpolate(
        fem.Expression(u_exact_ufl, V.element.interpolation_points)
    )
    bc = fem.dirichletbc(u_bc, boundary_dofs)

    # SUPG stabilization parameter: tau = h / (2 * |beta|)
    h = 1.0 / N
    tau_supg = h / (2.0 * beta_norm)

    # Variational form (Galerkin + SUPG)
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    a = (
        epsilon * ufl.inner(ufl.grad(u), ufl.grad(v))
        + ufl.dot(beta_ufl, ufl.grad(u)) * v
        + tau_supg * ufl.dot(beta_ufl, ufl.grad(v)) * ufl.dot(beta_ufl, ufl.grad(u))
    ) * ufl.dx

    L = (
        f_ufl * v
        + tau_supg * ufl.dot(beta_ufl, ufl.grad(v)) * f_ufl
    ) * ufl.dx

    # Solve with direct LU
    ksp_type = "preonly"
    pc_type = "lu"
    rtol = 1e-10

    petsc_options = {
        "ksp_type": ksp_type,
        "pc_type": pc_type,
    }

    problem = petsc.LinearProblem(
        a, L, bcs=[bc],
        petsc_options=petsc_options,
        petsc_options_prefix="convdiff_"
    )
    u_sol = problem.solve()
    u_sol.x.scatter_forward()

    iterations = 1  # direct solver

    # Verify: compute L2 error
    u_exact_func = fem.Function(V)
    u_exact_func.interpolate(
        fem.Expression(u_exact_ufl, V.element.interpolation_points)
    )
    error_L2_sq = fem.assemble_scalar(
        fem.form((u_sol - u_exact_func) ** 2 * ufl.dx)
    )
    error_L2 = np.sqrt(comm.allreduce(error_L2_sq, op=MPI.SUM))

    if comm.rank == 0:
        print(f"L2 error: {error_L2:.6e}")

    # Sample solution on output grid
    xmin, xmax, ymin, ymax = bbox
    xs = np.linspace(xmin, xmax, nx_out)
    ys = np.linspace(ymin, ymax, ny_out)
    XX, YY = np.meshgrid(xs, ys)
    points = np.zeros((nx_out * ny_out, 3))
    points[:, 0] = XX.ravel()
    points[:, 1] = YY.ravel()

    bb_tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, points)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points)

    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    for i in range(points.shape[0]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(points[i])
            cells_on_proc.append(links[0])
            eval_map.append(i)

    u_values = np.zeros((points.shape[0],), dtype=np.float64)
    if len(points_on_proc) > 0:
        vals = u_sol.eval(
            np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32)
        )
        u_values[eval_map] = vals.flatten()

    u_values_global = np.zeros_like(u_values)
    comm.Allreduce(u_values, u_values_global, op=MPI.SUM)

    u_grid = u_values_global.reshape(ny_out, nx_out)

    # Solver info
    solver_info = {
        "mesh_resolution": N,
        "element_degree": degree,
        "ksp_type": ksp_type,
        "pc_type": pc_type,
        "rtol": rtol,
        "iterations": iterations,
    }

    return {
        "u": u_grid,
        "solver_info": solver_info,
    }
