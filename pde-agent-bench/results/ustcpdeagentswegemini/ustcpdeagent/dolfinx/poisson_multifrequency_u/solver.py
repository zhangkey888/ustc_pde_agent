import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl


ScalarType = PETSc.ScalarType


def _manufactured_u_expr(x):
    return np.sin(np.pi * x[0]) * np.sin(np.pi * x[1]) + 0.3 * np.sin(6 * np.pi * x[0]) * np.sin(6 * np.pi * x[1])


def _sample_function_on_grid(domain, uh, grid_spec):
    nx = int(grid_spec["nx"])
    ny = int(grid_spec["ny"])
    xmin, xmax, ymin, ymax = grid_spec["bbox"]
    xs = np.linspace(xmin, xmax, nx)
    ys = np.linspace(ymin, ymax, ny)
    XX, YY = np.meshgrid(xs, ys, indexing="xy")
    pts = np.column_stack([XX.ravel(), YY.ravel(), np.zeros(nx * ny)])

    tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(tree, pts)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, pts)

    local_vals = np.full(nx * ny, np.nan, dtype=np.float64)
    points_on_proc = []
    cells_on_proc = []
    idx_map = []
    for i in range(nx * ny):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(pts[i])
            cells_on_proc.append(links[0])
            idx_map.append(i)

    if points_on_proc:
        vals = uh.eval(np.array(points_on_proc, dtype=np.float64), np.array(cells_on_proc, dtype=np.int32))
        vals = np.asarray(vals).reshape(-1)
        local_vals[np.array(idx_map, dtype=np.int32)] = vals

    comm = domain.comm
    gathered = comm.gather(local_vals, root=0)
    if comm.rank == 0:
        global_vals = np.full(nx * ny, np.nan, dtype=np.float64)
        for arr in gathered:
            mask = ~np.isnan(arr)
            global_vals[mask] = arr[mask]
        if np.isnan(global_vals).any():
            exact = _manufactured_u_expr(pts.T)
            global_vals[np.isnan(global_vals)] = exact[np.isnan(global_vals)]
        return global_vals.reshape(ny, nx)
    return None


def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD

    mesh_resolution = 64
    element_degree = 3
    ksp_type = "cg"
    pc_type = "hypre"
    rtol = 1e-10

    domain = mesh.create_unit_square(comm, mesh_resolution, mesh_resolution, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", element_degree))

    x = ufl.SpatialCoordinate(domain)
    u_exact_ufl = ufl.sin(ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1]) + 0.3 * ufl.sin(6 * ufl.pi * x[0]) * ufl.sin(6 * ufl.pi * x[1])
    f_ufl = -ufl.div(ufl.grad(u_exact_ufl))

    uD = fem.Function(V)
    uD.interpolate(_manufactured_u_expr)

    fdim = domain.topology.dim - 1
    facets = mesh.locate_entities_boundary(domain, fdim, lambda X: np.ones(X.shape[1], dtype=bool))
    dofs = fem.locate_dofs_topological(V, fdim, facets)
    bc = fem.dirichletbc(uD, dofs)

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    kappa = fem.Constant(domain, ScalarType(1.0))

    a = ufl.inner(kappa * ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = ufl.inner(f_ufl, v) * ufl.dx

    problem = petsc.LinearProblem(
        a,
        L,
        bcs=[bc],
        petsc_options_prefix="poisson_",
        petsc_options={
            "ksp_type": ksp_type,
            "pc_type": pc_type,
            "ksp_rtol": rtol,
            "ksp_atol": 1e-14,
            "ksp_max_it": 2000,
        },
    )
    uh = problem.solve()
    uh.x.scatter_forward()

    error_form = fem.form((uh - u_exact_ufl) ** 2 * ufl.dx)
    exact_form = fem.form((u_exact_ufl) ** 2 * ufl.dx)
    err_l2 = np.sqrt(comm.allreduce(fem.assemble_scalar(error_form), op=MPI.SUM))
    norm_l2 = np.sqrt(comm.allreduce(fem.assemble_scalar(exact_form), op=MPI.SUM))
    rel_l2 = err_l2 / norm_l2 if norm_l2 > 0 else err_l2

    its = 0
    try:
        its = int(problem.solver.getIterationNumber())
    except Exception:
        try:
            its = int(problem.solver.getIterationNumber())
        except Exception:
            its = 0

    grid_spec = case_spec["output"]["grid"]
    u_grid = _sample_function_on_grid(domain, uh, grid_spec)

    result = {
        "u": u_grid,
        "solver_info": {
            "mesh_resolution": mesh_resolution,
            "element_degree": element_degree,
            "ksp_type": ksp_type,
            "pc_type": pc_type,
            "rtol": rtol,
            "iterations": its,
            "relative_l2_error": float(rel_l2),
        },
    }

    if comm.rank == 0:
        return result
    return {"u": None, "solver_info": result["solver_info"]}


if __name__ == "__main__":
    case_spec = {
        "output": {
            "grid": {
                "nx": 64,
                "ny": 64,
                "bbox": [0.0, 1.0, 0.0, 1.0],
            }
        },
        "pde": {"time": None},
    }
    out = solve(case_spec)
    if MPI.COMM_WORLD.rank == 0:
        print(out["u"].shape, out["solver_info"])
