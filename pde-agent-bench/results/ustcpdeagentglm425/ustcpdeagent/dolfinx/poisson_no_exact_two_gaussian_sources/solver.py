import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc

def solve(case_spec: dict) -> dict:
    # Parse output grid specs
    grid_spec = case_spec["output"]["grid"]
    nx_out = grid_spec["nx"]
    ny_out = grid_spec["ny"]
    bbox = grid_spec["bbox"]  # [xmin, xmax, ymin, ymax]
    xmin, xmax, ymin, ymax = bbox

    # --- Numerical parameters ---
    mesh_res = 384
    elem_deg = 2
    ksp_type = "cg"
    pc_type = "hypre"
    rtol = 1e-10

    comm = MPI.COMM_WORLD

    # --- Create mesh ---
    domain = mesh.create_unit_square(comm, mesh_res, mesh_res, cell_type=mesh.CellType.triangle)
    tdim = domain.topology.dim
    fdim = tdim - 1

    # --- Function space ---
    V = fem.functionspace(domain, ("Lagrange", elem_deg))

    # --- Source term: two Gaussian sources ---
    x = ufl.SpatialCoordinate(domain)
    f_expr = ufl.exp(-250.0 * ((x[0] - 0.25)**2 + (x[1] - 0.25)**2)) + \
             ufl.exp(-250.0 * ((x[0] - 0.75)**2 + (x[1] - 0.7)**2))

    # --- Variational form: -div(kappa * grad(u)) = f ---
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    kappa = 1.0
    a = kappa * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx

    f_func = fem.Function(V)
    f_func.interpolate(fem.Expression(f_expr, V.element.interpolation_points))
    L = ufl.inner(f_func, v) * ufl.dx

    # --- Boundary conditions: u = 0 on all boundaries ---
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
    )
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    u_bc = fem.Function(V)
    u_bc.x.array[:] = 0.0
    bc = fem.dirichletbc(u_bc, boundary_dofs)

    # --- Solve ---
    problem = petsc.LinearProblem(
        a, L, bcs=[bc],
        petsc_options={
            "ksp_type": ksp_type,
            "pc_type": pc_type,
            "pc_hypre_type": "boomeramg",
            "ksp_rtol": rtol,
            "ksp_atol": 1e-12,
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
    pts = np.column_stack([XX.ravel(), YY.ravel(), np.zeros(nx_out * ny_out)])

    bb_tree = geometry.bb_tree(domain, tdim)
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

    u_values = np.full((pts.shape[0],), np.nan)
    if len(points_on_proc) > 0:
        vals = u_sol.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u_values[eval_map] = vals.flatten()

    # Handle parallel: gather from all procs
    if comm.size > 1:
        from mpi4py import MPI as MPI_mod
        recv_buf = comm.allreduce(u_values, op=MPI_mod.SUM)
        # Replace NaNs with values from other procs
        nan_mask = np.isnan(u_values)
        u_values[nan_mask] = recv_buf[nan_mask]

    u_grid = u_values.reshape(ny_out, nx_out)

    # --- Return results ---
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
