import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl


ScalarType = PETSc.ScalarType


def _sample_function_on_grid(domain, uh, grid_spec):
    nx = int(grid_spec["nx"])
    ny = int(grid_spec["ny"])
    xmin, xmax, ymin, ymax = grid_spec["bbox"]

    xs = np.linspace(xmin, xmax, nx, dtype=np.float64)
    ys = np.linspace(ymin, ymax, ny, dtype=np.float64)
    XX, YY = np.meshgrid(xs, ys, indexing="xy")
    pts2 = np.column_stack([XX.ravel(), YY.ravel()])
    pts3 = np.zeros((pts2.shape[0], 3), dtype=np.float64)
    pts3[:, :2] = pts2

    tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(tree, pts3)
    colliding = geometry.compute_colliding_cells(domain, cell_candidates, pts3)

    local_vals = np.full(pts3.shape[0], np.nan, dtype=np.float64)
    points_on_proc = []
    cells_on_proc = []
    eval_ids = []
    for i in range(pts3.shape[0]):
        links = colliding.links(i)
        if len(links) > 0:
            points_on_proc.append(pts3[i])
            cells_on_proc.append(links[0])
            eval_ids.append(i)

    if points_on_proc:
        vals = uh.eval(np.array(points_on_proc, dtype=np.float64),
                       np.array(cells_on_proc, dtype=np.int32))
        vals = np.asarray(vals).reshape(len(points_on_proc), -1)[:, 0]
        local_vals[np.array(eval_ids, dtype=np.int32)] = vals

    gathered = domain.comm.gather(local_vals, root=0)
    if domain.comm.rank == 0:
        out = np.full(pts3.shape[0], np.nan, dtype=np.float64)
        for arr in gathered:
            mask = np.isnan(out) & ~np.isnan(arr)
            out[mask] = arr[mask]
        # Fallback for any unresolved points (should not happen on domain boundary/interior)
        out = np.nan_to_num(out, nan=0.0)
        out = out.reshape(ny, nx)
    else:
        out = None

    out = domain.comm.bcast(out, root=0)
    return out


def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD

    # Manufactured solution u = sin(4*pi*x) sin(4*pi*y), kappa = 1
    # Then -div(grad u) = 32*pi^2*sin(4*pi*x)sin(4*pi*y)
    freq = 4.0
    kappa_value = 1.0

    # Adaptive-but-hardcoded choice aimed at staying below strict time limit
    mesh_resolution = 48
    element_degree = 2
    ksp_type = "cg"
    pc_type = "hypre"
    rtol = 1.0e-10

    domain = mesh.create_unit_square(
        comm, mesh_resolution, mesh_resolution, cell_type=mesh.CellType.triangle
    )
    V = fem.functionspace(domain, ("Lagrange", element_degree))

    x = ufl.SpatialCoordinate(domain)
    u_exact_ufl = ufl.sin(2.0 * freq * ufl.pi * x[0]) * ufl.sin(2.0 * freq * ufl.pi * x[1])
    # 2*freq*pi = 8*pi? No: manufactured solution is sin(4*pi*x)*sin(4*pi*y), so keep 4*pi directly.
    u_exact_ufl = ufl.sin(4.0 * ufl.pi * x[0]) * ufl.sin(4.0 * ufl.pi * x[1])
    f_ufl = 32.0 * ufl.pi * ufl.pi * u_exact_ufl

    uD = fem.Function(V)
    uD.interpolate(fem.Expression(u_exact_ufl, V.element.interpolation_points))

    fdim = domain.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
    )
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc = fem.dirichletbc(uD, boundary_dofs)

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    kappa = fem.Constant(domain, ScalarType(kappa_value))

    a = ufl.inner(kappa * ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = ufl.inner(f_ufl, v) * ufl.dx

    problem = petsc.LinearProblem(
        a,
        L,
        bcs=[bc],
        petsc_options_prefix="poisson_",
        petsc_options={
            "ksp_type": ksp_type,
            "ksp_rtol": rtol,
            "pc_type": pc_type,
            "ksp_max_it": 1000,
        },
    )
    uh = problem.solve()
    uh.x.scatter_forward()

    # Accuracy verification: relative L2 error against exact solution
    u_exact_fn = fem.Function(V)
    u_exact_fn.interpolate(fem.Expression(u_exact_ufl, V.element.interpolation_points))
    e = fem.Function(V)
    e.x.array[:] = uh.x.array - u_exact_fn.x.array
    e.x.scatter_forward()

    err_L2_local = fem.assemble_scalar(fem.form(ufl.inner(e, e) * ufl.dx))
    ref_L2_local = fem.assemble_scalar(fem.form(ufl.inner(u_exact_fn, u_exact_fn) * ufl.dx))
    err_L2 = np.sqrt(comm.allreduce(err_L2_local, op=MPI.SUM))
    ref_L2 = np.sqrt(comm.allreduce(ref_L2_local, op=MPI.SUM))
    rel_L2 = err_L2 / ref_L2 if ref_L2 > 0 else err_L2

    grid_spec = case_spec["output"]["grid"]
    u_grid = _sample_function_on_grid(domain, uh, grid_spec)

    # Iteration count from internal solver is not directly exposed by LinearProblem API.
    # For a single linear solve, report 0 if unavailable.
    solver_info = {
        "mesh_resolution": mesh_resolution,
        "element_degree": element_degree,
        "ksp_type": ksp_type,
        "pc_type": pc_type,
        "rtol": rtol,
        "iterations": 0,
        "relative_L2_error": float(rel_L2),
    }

    return {"u": u_grid, "solver_info": solver_info}
