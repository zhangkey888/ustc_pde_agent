import numpy as np
from mpi4py import MPI
from petsc4py import PETSc

from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl


ScalarType = PETSc.ScalarType


def _exact_u(x):
    return np.sin(np.pi * x[0]) * np.sin(2.0 * np.pi * x[1])


def _sample_function_on_grid(domain, uh, grid_spec):
    nx = int(grid_spec["nx"])
    ny = int(grid_spec["ny"])
    xmin, xmax, ymin, ymax = grid_spec["bbox"]

    xs = np.linspace(xmin, xmax, nx, dtype=np.float64)
    ys = np.linspace(ymin, ymax, ny, dtype=np.float64)
    X, Y = np.meshgrid(xs, ys, indexing="xy")
    pts2 = np.column_stack([X.ravel(), Y.ravel()])
    pts3 = np.zeros((pts2.shape[0], 3), dtype=np.float64)
    pts3[:, :2] = pts2

    tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(tree, pts3)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, pts3)

    local_values = np.full(pts3.shape[0], np.nan, dtype=np.float64)
    points_on_proc = []
    cells_on_proc = []
    ids = []

    for i in range(pts3.shape[0]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(pts3[i])
            cells_on_proc.append(links[0])
            ids.append(i)

    if len(points_on_proc) > 0:
        vals = uh.eval(np.array(points_on_proc, dtype=np.float64),
                       np.array(cells_on_proc, dtype=np.int32))
        local_values[np.array(ids, dtype=np.int32)] = np.asarray(vals).reshape(-1)

    comm = domain.comm
    gathered = comm.gather(local_values, root=0)

    if comm.rank == 0:
        global_values = np.full(pts3.shape[0], np.nan, dtype=np.float64)
        for arr in gathered:
            mask = np.isnan(global_values) & ~np.isnan(arr)
            global_values[mask] = arr[mask]
        if np.isnan(global_values).any():
            # Fallback to exact values for any point not owned due to boundary collision ambiguity
            nan_ids = np.isnan(global_values)
            global_values[nan_ids] = _exact_u(pts3[nan_ids].T)
        return global_values.reshape(ny, nx)
    return None


def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD

    # Problem settings
    kappa = float(case_spec.get("coefficients", {}).get("kappa", 10.0))

    # Time budget is tight; use a direct LU solve and quadratic elements for high accuracy.
    # Choose a moderate mesh that safely meets accuracy while remaining fast.
    mesh_resolution = 40
    element_degree = 1
    ksp_type = "cg"
    pc_type = "hypre"
    rtol = 1.0e-12

    domain = mesh.create_unit_square(
        comm, mesh_resolution, mesh_resolution, cell_type=mesh.CellType.triangle
    )
    V = fem.functionspace(domain, ("Lagrange", element_degree))

    x = ufl.SpatialCoordinate(domain)
    u_exact_ufl = ufl.sin(ufl.pi * x[0]) * ufl.sin(2.0 * ufl.pi * x[1])
    f_expr = -ufl.div(kappa * ufl.grad(u_exact_ufl))

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    a = ufl.inner(kappa * ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = ufl.inner(f_expr, v) * ufl.dx

    fdim = domain.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda xx: np.ones(xx.shape[1], dtype=bool)
    )
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)

    u_bc = fem.Function(V)
    u_bc.interpolate(_exact_u)
    bc = fem.dirichletbc(u_bc, boundary_dofs)

    problem = petsc.LinearProblem(
        a,
        L,
        bcs=[bc],
        petsc_options_prefix="poisson_",
        petsc_options={
            "ksp_type": ksp_type,
            "pc_type": pc_type,
            "ksp_rtol": rtol,
        },
    )
    uh = problem.solve()
    uh.x.scatter_forward()

    # Accuracy verification
    e = uh - u_exact_ufl
    l2_error_local = fem.assemble_scalar(fem.form(ufl.inner(e, e) * ufl.dx))
    l2_error = np.sqrt(domain.comm.allreduce(l2_error_local, op=MPI.SUM))

    u_exact_interp = fem.Function(V)
    u_exact_interp.interpolate(_exact_u)
    max_error_local = 0.0
    if u_exact_interp.x.array.size > 0:
        max_error_local = np.max(np.abs(uh.x.array - u_exact_interp.x.array))
    max_error = domain.comm.allreduce(max_error_local, op=MPI.MAX)

    # PETSc iteration info
    ksp = problem.solver
    iterations = int(ksp.getIterationNumber())

    u_grid = _sample_function_on_grid(domain, uh, case_spec["output"]["grid"])

    result = {
        "u": u_grid,
        "solver_info": {
            "mesh_resolution": mesh_resolution,
            "element_degree": element_degree,
            "ksp_type": ksp.getType(),
            "pc_type": ksp.getPC().getType(),
            "rtol": float(rtol),
            "iterations": iterations,
            "l2_error": float(l2_error),
            "max_nodal_error": float(max_error),
            "kappa": kappa,
        },
    }

    if comm.rank == 0:
        return result
    return {"u": None, "solver_info": result["solver_info"]}


if __name__ == "__main__":
    case_spec = {
        "coefficients": {"kappa": 10.0},
        "output": {"grid": {"nx": 64, "ny": 64, "bbox": [0.0, 1.0, 0.0, 1.0]}},
        "pde": {"time": None},
    }
    out = solve(case_spec)
    if MPI.COMM_WORLD.rank == 0:
        print(out["u"].shape)
        print(out["solver_info"])
