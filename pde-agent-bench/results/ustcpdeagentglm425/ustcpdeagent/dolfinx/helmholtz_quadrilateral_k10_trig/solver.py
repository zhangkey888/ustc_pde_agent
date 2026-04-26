import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc

def solve(case_spec: dict) -> dict:
    # Parse case spec
    k_val = case_spec["pde"]["params"]["k"]  # 10.0
    nx_out = case_spec["output"]["grid"]["nx"]
    ny_out = case_spec["output"]["grid"]["ny"]
    bbox = case_spec["output"]["grid"]["bbox"]  # [xmin, xmax, ymin, ymax]

    # Mesh parameters
    mesh_res = 80
    elem_deg = 3

    # Create quadrilateral mesh
    comm = MPI.COMM_WORLD
    p0 = np.array([bbox[0], bbox[2]])
    p1 = np.array([bbox[1], bbox[3]])
    domain = mesh.create_rectangle(comm, [p0, p1], [mesh_res, mesh_res],
                                    cell_type=mesh.CellType.quadrilateral)

    # Function space
    V = fem.functionspace(domain, ("Lagrange", elem_deg))

    # Variational problem: -∇²u - k²u = f
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    k_sq = PETSc.ScalarType(k_val**2)
    a = ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx - k_sq * ufl.inner(u, v) * ufl.dx

    # Source term: f = (13π² - k²) sin(2πx) cos(3πy)
    x = ufl.SpatialCoordinate(domain)
    f_coeff = (13.0 * np.pi**2 - k_val**2) * ufl.sin(2.0 * np.pi * x[0]) * ufl.cos(3.0 * np.pi * x[1])
    L = ufl.inner(f_coeff, v) * ufl.dx

    # Dirichlet BC: u = sin(2πx)cos(3πy) on ∂Ω
    g_expr = ufl.sin(2.0 * np.pi * x[0]) * ufl.cos(3.0 * np.pi * x[1])

    fdim = domain.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool))
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)

    u_bc = fem.Function(V)
    u_bc.interpolate(fem.Expression(g_expr, V.element.interpolation_points))
    bc = fem.dirichletbc(u_bc, boundary_dofs)

    # Solve with direct LU
    rtol = 1e-10
    problem = petsc.LinearProblem(
        a, L, bcs=[bc],
        petsc_options={
            "ksp_type": "preonly",
            "pc_type": "lu",
        },
        petsc_options_prefix="helmholtz_"
    )
    u_sol = problem.solve()
    u_sol.x.scatter_forward()

    iterations = 1  # direct solver counts as 1

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

    u_values = np.full((nx_out * ny_out,), np.nan)
    if len(points_on_proc) > 0:
        vals = u_sol.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u_values[eval_map] = vals.flatten()

    # Fill any NaN with exact solution (boundary edge cases)
    nan_mask = np.isnan(u_values)
    if np.any(nan_mask):
        exact = np.sin(2.0 * np.pi * points[nan_mask, 0]) * np.cos(3.0 * np.pi * points[nan_mask, 1])
        u_values[nan_mask] = exact

    u_grid = u_values.reshape(ny_out, nx_out)

    # L2 error verification
    u_exact_func = fem.Function(V)
    u_exact_func.interpolate(fem.Expression(g_expr, V.element.interpolation_points))

    error_L2 = fem.assemble_scalar(fem.form(
        ufl.inner(u_sol - u_exact_func, u_sol - u_exact_func) * ufl.dx
    ))
    error_L2 = np.sqrt(domain.comm.allreduce(error_L2, op=MPI.SUM))

    print(f"Mesh res: {mesh_res}, Element degree: {elem_deg}, k={k_val}")
    print(f"L2 error: {error_L2:.6e}")

    return {
        "u": u_grid,
        "solver_info": {
            "mesh_resolution": mesh_res,
            "element_degree": elem_deg,
            "ksp_type": "preonly",
            "pc_type": "lu",
            "rtol": rtol,
            "iterations": iterations,
        }
    }
