import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc

def solve(case_spec: dict) -> dict:
    """
    Solve -div(kappa * grad(u)) = f on [0,1]x[0,1] with u=g on boundary.
    Manufactured solution: u = sin(4*pi*x)*sin(4*pi*y), kappa=1.0
    """
    # Parse case_spec
    nx_out = case_spec["output"]["grid"]["nx"]
    ny_out = case_spec["output"]["grid"]["ny"]
    bbox = case_spec["output"]["grid"]["bbox"]  # [xmin, xmax, ymin, ymax]
    xmin, xmax, ymin, ymax = bbox

    # --- Parameters (adaptive for accuracy within time budget) ---
    mesh_res = 288
    elem_deg = 3
    ksp_type = "cg"
    pc_type = "hypre"
    rtol = 1e-10

    # --- Create mesh ---
    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_square(comm, mesh_res, mesh_res, cell_type=mesh.CellType.triangle)

    # --- Function space ---
    V = fem.functionspace(domain, ("Lagrange", elem_deg))

    # --- Define exact solution and source term symbolically ---
    x = ufl.SpatialCoordinate(domain)
    pi = np.pi
    u_exact_ufl = ufl.sin(4*pi*x[0]) * ufl.sin(4*pi*x[1])
    # f = -div(kappa*grad(u)) = -laplacian(u) since kappa=1
    # laplacian = -32*pi^2*sin(4*pi*x)*sin(4*pi*y)
    # so f = 32*pi^2*sin(4*pi*x)*sin(4*pi*y)
    f_ufl = 32 * pi**2 * ufl.sin(4*pi*x[0]) * ufl.sin(4*pi*x[1])

    # --- Boundary condition (all Dirichlet) ---
    fdim = domain.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool))
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)

    u_bc_func = fem.Function(V)
    u_bc_expr = fem.Expression(u_exact_ufl, V.element.interpolation_points)
    u_bc_func.interpolate(u_bc_expr)

    bc = fem.dirichletbc(u_bc_func, boundary_dofs)

    # --- Variational form ---
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    kappa = 1.0
    f_const = fem.Constant(domain, PETSc.ScalarType(1.0))  # placeholder

    a = kappa * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = f_ufl * v * ufl.dx

    # --- Solve ---
    problem = petsc.LinearProblem(
        a, L, bcs=[bc],
        petsc_options={
            "ksp_type": ksp_type,
            "pc_type": pc_type,
            "ksp_rtol": rtol,
            "ksp_atol": 1e-14,
        },
        petsc_options_prefix="poisson_"
    )
    u_sol = problem.solve()
    u_sol.x.scatter_forward()

    # Get iteration count
    ksp = problem.solver
    iterations = ksp.getIterationNumber()

    # --- Sample solution on output grid ---
    xs = np.linspace(xmin, xmax, nx_out)
    ys = np.linspace(ymin, ymax, ny_out)
    XX, YY = np.meshgrid(xs, ys)
    pts = np.zeros((nx_out * ny_out, 3))
    pts[:, 0] = XX.ravel()
    pts[:, 1] = YY.ravel()

    bb_tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, pts)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, pts)

    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    for i in range(pts.shape[0]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(pts[i])
            cells_on_proc.append(links[0])
            eval_map.append(i)

    u_grid = np.full((nx_out * ny_out,), np.nan)
    if len(points_on_proc) > 0:
        vals = u_sol.eval(
            np.array(points_on_proc),
            np.array(cells_on_proc, dtype=np.int32)
        )
        u_grid[eval_map] = vals.flatten()

    u_grid = u_grid.reshape(ny_out, nx_out)

    # --- Compute L2 error for verification ---
    u_ex_func = fem.Function(V)
    u_ex_func.interpolate(u_bc_expr)

    error_L2_sq = fem.assemble_scalar(
        fem.form(ufl.inner(u_sol - u_ex_func, u_sol - u_ex_func) * ufl.dx)
    )
    error_L2 = np.sqrt(MPI.COMM_WORLD.allreduce(error_L2_sq, op=MPI.SUM))

    u_ex_norm_sq = fem.assemble_scalar(
        fem.form(ufl.inner(u_ex_func, u_ex_func) * ufl.dx)
    )
    u_ex_norm = np.sqrt(MPI.COMM_WORLD.allreduce(u_ex_norm_sq, op=MPI.SUM))

    rel_L2_error = error_L2 / u_ex_norm if u_ex_norm > 1e-14 else error_L2

    if comm.rank == 0:
        print(f"L2 error: {error_L2:.6e}, relative L2 error: {rel_L2_error:.6e}")
        print(f"Mesh res: {mesh_res}, Element degree: {elem_deg}, KSP iters: {iterations}")

    # --- Build result ---
    result = {
        "u": u_grid,
        "solver_info": {
            "mesh_resolution": mesh_res,
            "element_degree": elem_deg,
            "ksp_type": ksp_type,
            "pc_type": pc_type,
            "rtol": rtol,
            "iterations": iterations,
        }
    }

    return result
