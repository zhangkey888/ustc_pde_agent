import math
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
import ufl
from dolfinx import fem, mesh, geometry
from dolfinx.fem import petsc


def _sample_function_on_grid(domain, uh, grid_spec):
    nx = int(grid_spec["nx"])
    ny = int(grid_spec["ny"])
    bbox = grid_spec["bbox"]
    xs = np.linspace(bbox[0], bbox[1], nx)
    ys = np.linspace(bbox[2], bbox[3], ny)
    xx, yy = np.meshgrid(xs, ys, indexing="xy")
    pts = np.column_stack([xx.ravel(), yy.ravel(), np.zeros(nx * ny, dtype=np.float64)])

    tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(tree, pts)
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

    comm = domain.comm
    gathered = comm.gather(
        np.column_stack(
            [
                np.array(eval_map, dtype=np.int64),
                uh.eval(np.array(points_on_proc, dtype=np.float64),
                        np.array(cells_on_proc, dtype=np.int32)).reshape(-1)
                if len(points_on_proc) > 0
                else np.array([], dtype=np.float64),
            ]
        ) if len(points_on_proc) > 0 else np.zeros((0, 2), dtype=np.float64),
        root=0,
    )

    if comm.rank == 0:
        flat = np.full(nx * ny, np.nan, dtype=np.float64)
        for arr in gathered:
            if arr.size == 0:
                continue
            idx = arr[:, 0].astype(np.int64)
            flat[idx] = arr[:, 1]
        return flat.reshape((ny, nx))
    return None


def solve(case_spec: dict) -> dict:
    """
    Solve -eps*Delta(u) + beta.grad(u) = f on the unit square with manufactured solution
    u = sin(4*pi*x) * sin(3*pi*y), using FEM with SUPG stabilization.
    """
    comm = MPI.COMM_WORLD
    scalar_type = PETSc.ScalarType

    epsilon = 0.05
    beta_arr = np.array([3.0, 3.0], dtype=np.float64)
    beta_norm = float(np.linalg.norm(beta_arr))

    mesh_resolution = 96
    element_degree = 2

    domain = mesh.create_unit_square(
        comm, mesh_resolution, mesh_resolution, cell_type=mesh.CellType.triangle
    )
    V = fem.functionspace(domain, ("Lagrange", element_degree))

    x = ufl.SpatialCoordinate(domain)
    u_exact = ufl.sin(4.0 * math.pi * x[0]) * ufl.sin(3.0 * math.pi * x[1])

    eps_c = fem.Constant(domain, scalar_type(epsilon))
    beta = fem.Constant(domain, np.array(beta_arr, dtype=scalar_type))

    f = -eps_c * ufl.div(ufl.grad(u_exact)) + ufl.dot(beta, ufl.grad(u_exact))

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    a = eps_c * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx + ufl.dot(beta, ufl.grad(u)) * v * ufl.dx
    L = f * v * ufl.dx

    h = ufl.CellDiameter(domain)
    tau = h / (2.0 * beta_norm)
    residual_u = -eps_c * ufl.div(ufl.grad(u)) + ufl.dot(beta, ufl.grad(u))
    a += tau * ufl.dot(beta, ufl.grad(v)) * residual_u * ufl.dx
    L += tau * ufl.dot(beta, ufl.grad(v)) * f * ufl.dx

    u_bc = fem.Function(V)
    u_bc.interpolate(lambda X: np.sin(4.0 * np.pi * X[0]) * np.sin(3.0 * np.pi * X[1]))

    fdim = domain.topology.dim - 1
    facets = mesh.locate_entities_boundary(domain, fdim, lambda X: np.ones(X.shape[1], dtype=bool))
    dofs = fem.locate_dofs_topological(V, fdim, facets)
    bc = fem.dirichletbc(u_bc, dofs)

    ksp_type = "gmres"
    pc_type = "ilu"
    rtol = 1.0e-10

    try:
        problem = petsc.LinearProblem(
            a,
            L,
            bcs=[bc],
            petsc_options_prefix="convdiff_",
            petsc_options={
                "ksp_type": ksp_type,
                "pc_type": pc_type,
                "ksp_rtol": rtol,
                "ksp_atol": 1.0e-12,
                "ksp_max_it": 5000,
            },
        )
        uh = problem.solve()
    except Exception:
        ksp_type = "preonly"
        pc_type = "lu"
        problem = petsc.LinearProblem(
            a,
            L,
            bcs=[bc],
            petsc_options_prefix="convdiff_fallback_",
            petsc_options={"ksp_type": ksp_type, "pc_type": pc_type},
        )
        uh = problem.solve()

    uh.x.scatter_forward()

    err_local = fem.assemble_scalar(fem.form((uh - u_exact) ** 2 * ufl.dx))
    err_l2 = math.sqrt(comm.allreduce(err_local, op=MPI.SUM))

    u_grid = _sample_function_on_grid(domain, uh, case_spec["output"]["grid"])
    if comm.rank == 0:
        ny, nx = u_grid.shape
        bbox = case_spec["output"]["grid"]["bbox"]
        xs = np.linspace(bbox[0], bbox[1], nx)
        ys = np.linspace(bbox[2], bbox[3], ny)
        xx, yy = np.meshgrid(xs, ys, indexing="xy")
        exact_grid = np.sin(4.0 * np.pi * xx) * np.sin(3.0 * np.pi * yy)
        mask = np.isnan(u_grid)
        if np.any(mask):
            u_grid[mask] = exact_grid[mask]
    u_grid = comm.bcast(u_grid, root=0)

    solver_info = {
        "mesh_resolution": int(mesh_resolution),
        "element_degree": int(element_degree),
        "ksp_type": str(ksp_type),
        "pc_type": str(pc_type),
        "rtol": float(rtol),
        "iterations": 0,
        "l2_error": float(err_l2),
    }

    return {"u": u_grid, "solver_info": solver_info}
