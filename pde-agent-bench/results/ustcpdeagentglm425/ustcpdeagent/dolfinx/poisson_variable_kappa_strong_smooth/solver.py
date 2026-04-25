import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc

ScalarType = PETSc.ScalarType

def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD

    # Extract output grid info
    nx_out = case_spec["output"]["grid"]["nx"]
    ny_out = case_spec["output"]["grid"]["ny"]
    bbox = case_spec["output"]["grid"]["bbox"]  # [xmin, xmax, ymin, ymax]
    xmin, xmax, ymin, ymax = bbox

    # Choose mesh resolution and element degree for accuracy within time limit
    mesh_res = 170
    elem_deg = 3

    # Create mesh
    domain = mesh.create_unit_square(comm, mesh_res, mesh_res, cell_type=mesh.CellType.triangle)

    # Function space
    V = fem.functionspace(domain, ("Lagrange", elem_deg))

    # Spatial coordinates for UFL
    x = ufl.SpatialCoordinate(domain)

    # Exact (manufactured) solution
    u_exact = ufl.sin(3 * ufl.pi * x[0]) * ufl.sin(2 * ufl.pi * x[1])

    # Variable coefficient kappa
    kappa = 1.0 + 0.9 * ufl.sin(2 * ufl.pi * x[0]) * ufl.sin(2 * ufl.pi * x[1])

    # Source term f = -div(kappa * grad(u_exact))
    # Compute symbolically using UFL
    grad_u_exact = ufl.grad(u_exact)
    kappa_grad_u = kappa * grad_u_exact
    div_kappa_grad_u = ufl.div(kappa_grad_u)
    f_expr_ufl = -div_kappa_grad_u

    # Boundary condition g = u_exact on boundary
    g_expr_ufl = u_exact

    # Variational form
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    a = ufl.inner(kappa * ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = f_expr_ufl * v * ufl.dx

    # Boundary conditions: u = g on entire boundary
    fdim = domain.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
    )
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)

    # Interpolate g onto a function for BC
    u_bc = fem.Function(V)
    u_bc.interpolate(
        fem.Expression(g_expr_ufl, V.element.interpolation_points)
    )
    bc = fem.dirichletbc(u_bc, boundary_dofs)

    # Solve
    ksp_type = "cg"
    pc_type = "hypre"
    rtol = 1e-12

    petsc_options = {
        "ksp_type": ksp_type,
        "pc_type": pc_type,
        "pc_hypre_type": "boomeramg",
        "ksp_rtol": str(rtol),
        "ksp_atol": "1e-14",
        "ksp_max_it": "500",
    }

    problem = petsc.LinearProblem(
        a, L, bcs=[bc],
        petsc_options=petsc_options,
        petsc_options_prefix="poisson_"
    )
    u_sol = problem.solve()
    u_sol.x.scatter_forward()

    # Get iteration count
    ksp = problem.solver
    iterations = ksp.getIterationNumber()

    # Evaluate solution on the output grid
    xs = np.linspace(xmin, xmax, nx_out)
    ys = np.linspace(ymin, ymax, ny_out)
    XX, YY = np.meshgrid(xs, ys)
    points = np.zeros((ny_out * nx_out, 3))
    points[:, 0] = XX.ravel()
    points[:, 1] = YY.ravel()
    points[:, 2] = 0.0

    # Use bounding box tree for point evaluation
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
        pts_arr = np.array(points_on_proc)
        cells_arr = np.array(cells_on_proc, dtype=np.int32)
        vals = u_sol.eval(pts_arr, cells_arr)
        u_values[eval_map] = vals.flatten()

    # In MPI parallel, we need to gather from all processes
    if comm.size > 1:
        # Gather all values
        all_values = comm.gather(u_values, root=0)
        all_maps = comm.gather(eval_map, root=0)
        if comm.rank == 0:
            final_values = np.zeros((points.shape[0],), dtype=np.float64)
            for vals, mp in zip(all_values, all_maps):
                if len(mp) > 0:
                    final_values[mp] = vals[mp]
            u_grid = final_values.reshape(ny_out, nx_out)
        else:
            u_grid = None
        u_grid = comm.bcast(u_grid, root=0)
    else:
        u_grid = u_values.reshape(ny_out, nx_out)

    # Compute L2 error for verification
    L2_error = fem.assemble_scalar(
        fem.form((u_sol - u_exact) ** 2 * ufl.dx)
    )
    L2_error = np.sqrt(comm.allreduce(L2_error, op=MPI.SUM))

    # Compute H1 semi-error
    H1_semi_error = fem.assemble_scalar(
        fem.form(ufl.inner(ufl.grad(u_sol - u_exact), ufl.grad(u_sol - u_exact)) * ufl.dx)
    )
    H1_semi_error = np.sqrt(comm.allreduce(H1_semi_error, op=MPI.SUM))

    solver_info = {
        "mesh_resolution": mesh_res,
        "element_degree": elem_deg,
        "ksp_type": ksp_type,
        "pc_type": pc_type,
        "rtol": rtol,
        "iterations": iterations,
    }

    return {
        "u": u_grid,
        "solver_info": solver_info,
    }
