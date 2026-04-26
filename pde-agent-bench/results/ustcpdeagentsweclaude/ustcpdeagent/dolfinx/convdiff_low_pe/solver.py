import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
import ufl
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc

ScalarType = PETSc.ScalarType


def _sample_function_on_grid(domain, uh, grid_spec):
    nx = int(grid_spec["nx"])
    ny = int(grid_spec["ny"])
    xmin, xmax, ymin, ymax = grid_spec["bbox"]

    xs = np.linspace(xmin, xmax, nx, dtype=np.float64)
    ys = np.linspace(ymin, ymax, ny, dtype=np.float64)
    XX, YY = np.meshgrid(xs, ys, indexing="xy")
    pts = np.column_stack(
        [XX.ravel(), YY.ravel(), np.zeros(nx * ny, dtype=np.float64)]
    )

    tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(tree, pts)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, pts)

    values = np.full(nx * ny, np.nan, dtype=np.float64)
    points_on_proc = []
    cells_on_proc = []
    ids_on_proc = []

    for i in range(pts.shape[0]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(pts[i])
            cells_on_proc.append(links[0])
            ids_on_proc.append(i)

    if points_on_proc:
        vals = uh.eval(
            np.array(points_on_proc, dtype=np.float64),
            np.array(cells_on_proc, dtype=np.int32),
        ).reshape(-1)
        values[np.array(ids_on_proc, dtype=np.int32)] = np.real(vals)

    gathered = domain.comm.allgather(values)
    merged = np.full_like(values, np.nan)
    for arr in gathered:
        mask = np.isnan(merged) & ~np.isnan(arr)
        merged[mask] = arr[mask]

    if np.isnan(merged).any():
        raise RuntimeError("Failed to evaluate solution at one or more sample points")

    return merged.reshape(ny, nx)


def solve(case_spec: dict) -> dict:
    """
    Solve -eps*Delta(u) + beta.grad(u) = f on the unit square with Dirichlet data
    from the manufactured exact solution u = sin(pi x) sin(pi y).

    Returns:
      {
        "u": ndarray shape (ny, nx),
        "solver_info": {...}
      }
    """
    comm = MPI.COMM_WORLD

    # Fixed high-accuracy configuration chosen to satisfy the strict accuracy target
    # while keeping setup simple and robust for this small steady benchmark.
    mesh_resolution = 32
    element_degree = 2
    ksp_type = "preonly"
    pc_type = "lu"
    rtol = 1.0e-12

    domain = mesh.create_unit_square(
        comm, mesh_resolution, mesh_resolution, cell_type=mesh.CellType.triangle
    )
    V = fem.functionspace(domain, ("Lagrange", element_degree))

    x = ufl.SpatialCoordinate(domain)
    pi = np.pi
    eps = 0.2
    beta_vec = np.array([1.0, 0.5], dtype=np.float64)
    beta = fem.Constant(domain, np.array(beta_vec, dtype=ScalarType))

    u_exact_ufl = ufl.sin(pi * x[0]) * ufl.sin(pi * x[1])
    f_ufl = -eps * ufl.div(ufl.grad(u_exact_ufl)) + ufl.dot(beta, ufl.grad(u_exact_ufl))

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    # Standard Galerkin form; Pe is moderate and this manufactured case is smooth.
    a = (
        eps * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
        + ufl.dot(beta, ufl.grad(u)) * v * ufl.dx
    )
    L = f_ufl * v * ufl.dx

    # Optional mild SUPG for robustness on advective bias
    h = ufl.CellDiameter(domain)
    beta_norm = ufl.sqrt(ufl.dot(beta, beta))
    tau = h / (2.0 * beta_norm + 4.0 * eps / h)
    r_trial = -eps * ufl.div(ufl.grad(u)) + ufl.dot(beta, ufl.grad(u))
    a += tau * r_trial * ufl.dot(beta, ufl.grad(v)) * ufl.dx
    L += tau * f_ufl * ufl.dot(beta, ufl.grad(v)) * ufl.dx

    u_bc = fem.Function(V)
    u_bc.interpolate(lambda X: np.sin(pi * X[0]) * np.sin(pi * X[1]))

    fdim = domain.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda X: np.ones(X.shape[1], dtype=bool)
    )
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc = fem.dirichletbc(u_bc, boundary_dofs)

    problem = petsc.LinearProblem(
        a,
        L,
        bcs=[bc],
        petsc_options_prefix="convdiff_",
        petsc_options={
            "ksp_type": ksp_type,
            "pc_type": pc_type,
        },
    )
    uh = problem.solve()
    uh.x.scatter_forward()

    # Accuracy verification against manufactured exact solution
    u_exact = fem.Function(V)
    u_exact.interpolate(lambda X: np.sin(pi * X[0]) * np.sin(pi * X[1]))
    l2_error = np.sqrt(
        comm.allreduce(
            fem.assemble_scalar(fem.form((uh - u_exact) ** 2 * ufl.dx)),
            op=MPI.SUM,
        )
    )
    linf_nodal = np.max(np.abs(uh.x.array - u_exact.x.array))
    linf_nodal = comm.allreduce(linf_nodal, op=MPI.MAX)

    u_grid = _sample_function_on_grid(domain, uh, case_spec["output"]["grid"])

    solver_info = {
        "mesh_resolution": int(mesh_resolution),
        "element_degree": int(element_degree),
        "ksp_type": ksp_type,
        "pc_type": pc_type,
        "rtol": float(rtol),
        "iterations": 1,
        "verification_L2_error": float(l2_error),
        "verification_max_nodal_error": float(linf_nodal),
    }

    return {"u": u_grid, "solver_info": solver_info}


if __name__ == "__main__":
    case_spec = {
        "output": {"grid": {"nx": 64, "ny": 64, "bbox": [0.0, 1.0, 0.0, 1.0]}}
    }
    result = solve(case_spec)
    if MPI.COMM_WORLD.rank == 0:
        print(result["u"].shape)
        print(result["solver_info"])
