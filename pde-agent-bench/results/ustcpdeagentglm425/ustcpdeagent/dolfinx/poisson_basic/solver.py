import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc

ScalarType = PETSc.ScalarType

def solve(case_spec: dict) -> dict:
    # Extract output grid info
    grid_info = case_spec["output"]["grid"]
    nx_out = grid_info["nx"]
    ny_out = grid_info["ny"]
    bbox = grid_info["bbox"]  # [xmin, xmax, ymin, ymax]
    xmin, xmax, ymin, ymax = bbox

    # Choose solver parameters for maximum accuracy within time budget
    mesh_res = 128
    elem_degree = 4
    ksp_type = "cg"
    pc_type = "hypre"
    rtol = 1e-12

    comm = MPI.COMM_WORLD

    # Create mesh
    domain = mesh.create_unit_square(comm, mesh_res, mesh_res, cell_type=mesh.CellType.triangle)

    # Function space
    V = fem.functionspace(domain, ("Lagrange", elem_degree))

    # Manufactured solution: u = sin(pi*x)*sin(pi*y)
    x = ufl.SpatialCoordinate(domain)
    u_exact = ufl.sin(ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])

    # RHS: f = 2*pi^2*sin(pi*x)*sin(pi*y)  (from -div(grad(u)) = f with kappa=1)
    f_expr = 2.0 * ufl.pi**2 * ufl.sin(ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])

    # Dirichlet BC on entire boundary
    fdim = domain.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
    )
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)

    u_bc = fem.Function(V)
    u_bc.interpolate(
        fem.Expression(u_exact, V.element.interpolation_points)
    )
    bc = fem.dirichletbc(u_bc, boundary_dofs)

    # Variational form
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    kappa = 1.0
    a = kappa * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = ufl.inner(f_expr, v) * ufl.dx

    # Solve
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

    # Get solver iteration info
    ksp = problem.solver
    iterations = ksp.getIterationNumber()

    # Evaluate on output grid
    xs = np.linspace(xmin, xmax, nx_out)
    ys = np.linspace(ymin, ymax, ny_out)
    XX, YY = np.meshgrid(xs, ys)
    pts = np.column_stack([XX.ravel(), YY.ravel(), np.zeros(nx_out * ny_out)])

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

    u_values = np.zeros((nx_out * ny_out,))
    if len(points_on_proc) > 0:
        vals = u_sol.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u_values[eval_map] = vals.flatten()

    # Gather across processes if parallel
    if comm.size > 1:
        u_values_global = np.zeros_like(u_values)
        comm.Allreduce(u_values, u_values_global, op=MPI.SUM)
        u_values = u_values_global

    u_grid = u_values.reshape(ny_out, nx_out)

    # Compute L2 error for verification
    err_degree = max(elem_degree + 3, 6)
    V_err = fem.functionspace(domain, ("Lagrange", err_degree))
    u_exact_func = fem.Function(V_err)
    u_exact_func.interpolate(
        fem.Expression(u_exact, V_err.element.interpolation_points)
    )
    u_sol_high = fem.Function(V_err)
    u_sol_high.interpolate(u_sol)
    error_expr = ufl.inner(u_sol_high - u_exact_func, u_sol_high - u_exact_func) * ufl.dx
    error_form = fem.form(error_expr)
    l2_error_sq = fem.assemble_scalar(error_form)
    l2_error = np.sqrt(comm.allreduce(l2_error_sq, op=MPI.SUM))

    if comm.rank == 0:
        print(f"L2 error: {l2_error:.6e}")
        print(f"Solver iterations: {iterations}")

    result = {
        "u": u_grid,
        "solver_info": {
            "mesh_resolution": mesh_res,
            "element_degree": elem_degree,
            "ksp_type": ksp_type,
            "pc_type": pc_type,
            "rtol": rtol,
            "iterations": iterations,
        }
    }

    return result


if __name__ == "__main__":
    case_spec = {
        "output": {
            "grid": {
                "nx": 50,
                "ny": 50,
                "bbox": [0.0, 1.0, 0.0, 1.0]
            }
        },
        "pde": {}
    }
    result = solve(case_spec)
    print(f"Output shape: {result['u'].shape}")
    print(f"Max value: {np.max(result['u']):.6e}")
    print(f"Solver info: {result['solver_info']}")
