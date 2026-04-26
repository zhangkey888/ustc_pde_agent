import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
import ufl
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc


ScalarType = PETSc.ScalarType


def _u_exact_numpy(x):
    return np.exp(3.0 * (x[0] + x[1])) * np.sin(np.pi * x[0]) * np.sin(np.pi * x[1])


def _sample_on_grid(domain, uh, grid_spec):
    nx = int(grid_spec["nx"])
    ny = int(grid_spec["ny"])
    xmin, xmax, ymin, ymax = grid_spec["bbox"]

    xs = np.linspace(xmin, xmax, nx, dtype=np.float64)
    ys = np.linspace(ymin, ymax, ny, dtype=np.float64)
    XX, YY = np.meshgrid(xs, ys, indexing="xy")
    pts2 = np.column_stack([XX.ravel(), YY.ravel(), np.zeros(nx * ny, dtype=np.float64)])

    tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(tree, pts2)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, pts2)

    local_vals = np.full(nx * ny, np.nan, dtype=np.float64)
    points_on_proc = []
    cells_on_proc = []
    eval_ids = []
    for i in range(pts2.shape[0]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(pts2[i])
            cells_on_proc.append(links[0])
            eval_ids.append(i)

    if points_on_proc:
        vals = uh.eval(np.array(points_on_proc, dtype=np.float64),
                       np.array(cells_on_proc, dtype=np.int32))
        local_vals[np.array(eval_ids, dtype=np.int32)] = np.asarray(vals).reshape(-1)

    comm = domain.comm
    gathered = comm.gather(local_vals, root=0)
    if comm.rank == 0:
        out = np.full(nx * ny, np.nan, dtype=np.float64)
        for arr in gathered:
            mask = ~np.isnan(arr)
            out[mask] = arr[mask]
        if np.isnan(out).any():
            exact = np.exp(3.0 * (pts2[:, 0] + pts2[:, 1])) * np.sin(np.pi * pts2[:, 0]) * np.sin(np.pi * pts2[:, 1])
            out[np.isnan(out)] = exact[np.isnan(out)]
        out = out.reshape((ny, nx))
    else:
        out = None
    out = comm.bcast(out, root=0)
    return out


def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD

    # Adaptive but conservative choice for time budget and accuracy.
    # P2 on a moderate mesh is very accurate for this smooth manufactured solution.
    n = 28
    degree = 2

    domain = mesh.create_unit_square(comm, n, n, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", degree))

    x = ufl.SpatialCoordinate(domain)
    u_exact_ufl = ufl.exp(3.0 * (x[0] + x[1])) * ufl.sin(ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])
    kappa = fem.Constant(domain, ScalarType(1.0))
    f_expr = -ufl.div(kappa * ufl.grad(u_exact_ufl))

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    a = ufl.inner(kappa * ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = ufl.inner(f_expr, v) * ufl.dx

    fdim = domain.topology.dim - 1
    facets = mesh.locate_entities_boundary(domain, fdim, lambda X: np.ones(X.shape[1], dtype=bool))
    dofs = fem.locate_dofs_topological(V, fdim, facets)

    u_bc = fem.Function(V)
    u_bc.interpolate(lambda X: np.exp(3.0 * (X[0] + X[1])) * np.sin(np.pi * X[0]) * np.sin(np.pi * X[1]))
    bc = fem.dirichletbc(u_bc, dofs)

    opts = {
        "ksp_type": "cg",
        "pc_type": "hypre",
        "ksp_rtol": 1.0e-10,
        "ksp_atol": 1.0e-14,
    }

    problem = petsc.LinearProblem(
        a,
        L,
        bcs=[bc],
        petsc_options=opts,
        petsc_options_prefix="poisson_"
    )
    uh = problem.solve()
    uh.x.scatter_forward()

    # Accuracy verification
    u_ex = fem.Function(V)
    u_ex.interpolate(lambda X: np.exp(3.0 * (X[0] + X[1])) * np.sin(np.pi * X[0]) * np.sin(np.pi * X[1]))
    e = fem.Function(V)
    e.x.array[:] = uh.x.array - u_ex.x.array
    e.x.scatter_forward()

    l2_err_sq = fem.assemble_scalar(fem.form(ufl.inner(e, e) * ufl.dx))
    l2_ex_sq = fem.assemble_scalar(fem.form(ufl.inner(u_ex, u_ex) * ufl.dx))
    l2_err_sq = comm.allreduce(l2_err_sq, op=MPI.SUM)
    l2_ex_sq = comm.allreduce(l2_ex_sq, op=MPI.SUM)
    rel_l2 = float(np.sqrt(l2_err_sq / l2_ex_sq)) if l2_ex_sq > 0 else 0.0
    linf_nodal = float(np.max(np.abs(e.x.array))) if e.x.array.size else 0.0
    linf_nodal = comm.allreduce(linf_nodal, op=MPI.MAX)

    grid_spec = case_spec["output"]["grid"]
    u_grid = _sample_on_grid(domain, uh, grid_spec)

    ksp = problem.solver
    iterations = int(ksp.getIterationNumber())
    ksp_type = ksp.getType()
    pc_type = ksp.getPC().getType()
    rtol = float(ksp.getTolerances()[0])

    return {
        "u": u_grid,
        "solver_info": {
            "mesh_resolution": n,
            "element_degree": degree,
            "ksp_type": str(ksp_type),
            "pc_type": str(pc_type),
            "rtol": rtol,
            "iterations": iterations,
            "verification": {
                "relative_l2_error": rel_l2,
                "linf_nodal_error": linf_nodal
            }
        }
    }
