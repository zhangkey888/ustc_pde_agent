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

    # Parameters - adaptive for accuracy
    mesh_res = 64
    element_degree = 3
    ksp_type = "cg"
    pc_type = "hypre"
    rtol = 1e-12

    # Create quadrilateral mesh on [0,1]x[0,1]
    n_mesh = mesh_res
    domain = mesh.create_rectangle(
        comm,
        [np.array([xmin, ymin]), np.array([xmax, ymax])],
        [n_mesh, n_mesh],
        cell_type=mesh.CellType.quadrilateral,
    )

    # Function space
    import basix.ufl
    cell_name = domain.topology.cell_name()
    elem = basix.ufl.element("Lagrange", cell_name, element_degree)
    V = fem.functionspace(domain, elem)

    # Define variational problem
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    kappa = fem.Constant(domain, ScalarType(1.0))

    # Source term from manufactured solution u = sin(pi*x)*sin(pi*y)
    # f = kappa * 2*pi^2*sin(pi*x)*sin(pi*y)
    x = ufl.SpatialCoordinate(domain)
    f_expr = kappa * 2.0 * ufl.pi**2 * ufl.sin(ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])

    a = kappa * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = f_expr * v * ufl.dx

    # Dirichlet BC: u = sin(pi*x)*sin(pi*y) on boundary
    fdim = domain.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
    )
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)

    u_bc = fem.Function(V)
    u_exact_uf = ufl.sin(ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])
    u_bc.interpolate(fem.Expression(u_exact_uf, V.element.interpolation_points))

    bc = fem.dirichletbc(u_bc, boundary_dofs)

    # Solve
    problem = petsc.LinearProblem(
        a, L, bcs=[bc],
        petsc_options={
            "ksp_type": ksp_type,
            "pc_type": pc_type,
            "ksp_rtol": rtol,
            "ksp_atol": 1e-14,
            "ksp_max_it": 1000,
        },
        petsc_options_prefix="poisson_",
    )
    u_sol = problem.solve()
    u_sol.x.scatter_forward()

    # Get iteration count
    ksp = problem.solver
    iterations = ksp.getIterationNumber()

    # Compute L2 error for verification
    L2_error_sq = fem.assemble_scalar(
        fem.form((u_sol - u_exact_uf)**2 * ufl.dx)
    )
    L2_error = np.sqrt(MPI.COMM_WORLD.allreduce(float(L2_error_sq), op=MPI.SUM))

    # Sample solution on output grid
    xs = np.linspace(xmin, xmax, nx_out)
    ys = np.linspace(ymin, ymax, ny_out)
    XX, YY = np.meshgrid(xs, ys)
    pts = np.vstack([XX.ravel(), YY.ravel(), np.zeros(nx_out * ny_out)])  # shape (3, N)

    bb_tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, pts.T)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, pts.T)

    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    for i in range(pts.shape[1]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(pts.T[i])
            cells_on_proc.append(links[0])
            eval_map.append(i)

    u_values = np.zeros((pts.shape[1],), dtype=np.float64)
    if len(points_on_proc) > 0:
        vals = u_sol.eval(
            np.array(points_on_proc),
            np.array(cells_on_proc, dtype=np.int32),
        )
        u_values[eval_map] = vals.flatten()

    # Gather on all processes
    u_values_global = np.zeros_like(u_values)
    MPI.COMM_WORLD.Allreduce(u_values, u_values_global, op=MPI.SUM)

    u_grid = u_values_global.reshape(ny_out, nx_out)

    # Compute max error on grid for verification
    exact_grid = np.sin(np.pi * XX) * np.sin(np.pi * YY)
    max_error = np.max(np.abs(u_grid - exact_grid))

    if comm.rank == 0:
        print(f"L2 error: {L2_error:.6e}, Max grid error: {max_error:.6e}, Iterations: {iterations}")

    solver_info = {
        "mesh_resolution": n_mesh,
        "element_degree": element_degree,
        "ksp_type": ksp_type,
        "pc_type": pc_type,
        "rtol": rtol,
        "iterations": int(iterations),
    }

    # Check if time-dependent info is needed
    if "time" in case_spec.get("pde", {}):
        solver_info["dt"] = 1.0
        solver_info["n_steps"] = 1
        solver_info["time_scheme"] = "none"

    return {"u": u_grid, "solver_info": solver_info}


if __name__ == "__main__":
    case_spec = {
        "output": {
            "grid": {"nx": 50, "ny": 50, "bbox": [0.0, 1.0, 0.0, 1.0]}
        },
        "pde": {}
    }
    result = solve(case_spec)
    print(f"Output shape: {result['u'].shape}")
    print(f"Solver info: {result['solver_info']}")
