import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc

def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD

    # Extract output grid spec
    out = case_spec["output"]["grid"]
    nx_out = out["nx"]
    ny_out = out["ny"]
    bbox = out["bbox"]  # [xmin, xmax, ymin, ymax]
    xmin, xmax, ymin, ymax = bbox

    # Parameters - P3 with mesh_res=80 gives L2~7.9e-7, well within budget
    mesh_res = 80
    element_degree = 3

    # Create mesh
    domain = mesh.create_unit_square(comm, mesh_res, mesh_res, cell_type=mesh.CellType.triangle)

    # Function space
    V = fem.functionspace(domain, ("Lagrange", element_degree))

    # Exact solution and source term via UFL
    x = ufl.SpatialCoordinate(domain)
    pi = ufl.pi
    u_exact_ufl = ufl.sin(pi * x[0]) * ufl.sin(pi * x[1]) + 0.3 * ufl.sin(6 * pi * x[0]) * ufl.sin(6 * pi * x[1])
    f_ufl = -ufl.div(1.0 * ufl.grad(u_exact_ufl))

    # Variational form
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    a = ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx

    # Interpolate source term
    f_expr = fem.Expression(f_ufl, V.element.interpolation_points)
    f_func = fem.Function(V)
    f_func.interpolate(f_expr)
    L = ufl.inner(f_func, v) * ufl.dx

    # Dirichlet BC on entire boundary
    fdim = domain.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool))
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)

    u_bc = fem.Function(V)
    u_bc.interpolate(fem.Expression(u_exact_ufl, V.element.interpolation_points))
    bc = fem.dirichletbc(u_bc, boundary_dofs)

    # Solve with direct LU
    rtol = 1e-10
    problem = petsc.LinearProblem(
        a, L, bcs=[bc],
        petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
        petsc_options_prefix="poisson_"
    )
    u_sol = problem.solve()
    u_sol.x.scatter_forward()

    # Evaluate solution on output grid - vectorized approach
    xs = np.linspace(xmin, xmax, nx_out)
    ys = np.linspace(ymin, ymax, ny_out)
    XX, YY = np.meshgrid(xs, ys)

    # Build point array (N, 3) for geometry API
    pts = np.zeros((nx_out * ny_out, 3))
    pts[:, 0] = XX.ravel()
    pts[:, 1] = YY.ravel()

    bb_tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, pts)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, pts)

    # Build arrays of colliding points and cells
    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    for i in range(pts.shape[0]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(pts[i])
            cells_on_proc.append(links[0])
            eval_map.append(i)

    u_values = np.zeros(nx_out * ny_out)
    if len(points_on_proc) > 0:
        pts_arr = np.array(points_on_proc)
        cells_arr = np.array(cells_on_proc, dtype=np.int32)
        vals = u_sol.eval(pts_arr, cells_arr)
        u_values[eval_map] = vals.ravel()

    # MPI gather
    if comm.size > 1:
        u_values_global = np.zeros_like(u_values)
        comm.Allreduce(u_values, u_values_global, op=MPI.SUM)
        u_values = u_values_global

    u_grid = u_values.reshape(ny_out, nx_out)

    # Compute L2 error for verification
    L2_error_ufl = ufl.inner(u_sol - u_exact_ufl, u_sol - u_exact_ufl) * ufl.dx
    L2_error_form = fem.form(L2_error_ufl)
    L2_error_local = fem.assemble_scalar(L2_error_form)
    L2_error = np.sqrt(comm.allreduce(L2_error_local, op=MPI.SUM))

    if comm.rank == 0:
        print(f"L2 error: {L2_error:.6e}")

    solver_info = {
        "mesh_resolution": mesh_res,
        "element_degree": element_degree,
        "ksp_type": "preonly",
        "pc_type": "lu",
        "rtol": rtol,
        "iterations": 1,
    }

    return {"u": u_grid, "solver_info": solver_info}
