import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl

ScalarType = PETSc.ScalarType


def _manufactured_terms(x, y, k):
    u = x * (1.0 - x) * y * (1.0 - y)
    lap_u = -2.0 * y * (1.0 - y) - 2.0 * x * (1.0 - x)
    f = -lap_u - (k ** 2) * u
    return u, f


def _sample_on_grid(domain, uh, nx, ny, bbox):
    xmin, xmax, ymin, ymax = bbox
    xs = np.linspace(xmin, xmax, nx)
    ys = np.linspace(ymin, ymax, ny)
    XX, YY = np.meshgrid(xs, ys, indexing="xy")
    pts = np.column_stack([XX.ravel(), YY.ravel(), np.zeros(nx * ny, dtype=np.float64)])

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

    local_values = np.full(pts.shape[0], np.nan, dtype=np.float64)
    if points_on_proc:
        vals = uh.eval(np.array(points_on_proc, dtype=np.float64),
                       np.array(cells_on_proc, dtype=np.int32))
        local_values[np.array(eval_map, dtype=np.int32)] = np.asarray(vals).reshape(-1).real

    gathered = domain.comm.gather(local_values, root=0)
    if domain.comm.rank == 0:
        values = np.full(pts.shape[0], np.nan, dtype=np.float64)
        for arr in gathered:
            mask = ~np.isnan(arr)
            values[mask] = arr[mask]
        if np.isnan(values).any():
            raise RuntimeError("Failed to evaluate solution at all requested grid points.")
        return values.reshape(ny, nx)
    return None


def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD
    rank = comm.rank

    pde = case_spec.get("pde", {})
    grid = case_spec["output"]["grid"]

    k = float(pde.get("k", 2.0))
    nx_out = int(grid["nx"])
    ny_out = int(grid["ny"])
    bbox = grid["bbox"]

    mesh_resolution = 20
    element_degree = 2

    domain = mesh.create_unit_square(
        comm, mesh_resolution, mesh_resolution, cell_type=mesh.CellType.triangle
    )
    V = fem.functionspace(domain, ("Lagrange", element_degree))

    x = ufl.SpatialCoordinate(domain)
    u_exact = x[0] * (1.0 - x[0]) * x[1] * (1.0 - x[1])
    f = 2.0 * x[1] * (1.0 - x[1]) + 2.0 * x[0] * (1.0 - x[0]) - (k ** 2) * u_exact

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    a = (ufl.inner(ufl.grad(u), ufl.grad(v)) - (k ** 2) * ufl.inner(u, v)) * ufl.dx
    L = ufl.inner(f, v) * ufl.dx

    fdim = domain.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda X: np.ones(X.shape[1], dtype=bool)
    )
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)

    u_bc = fem.Function(V)
    u_bc.interpolate(lambda X: X[0] * (1.0 - X[0]) * X[1] * (1.0 - X[1]))
    bc = fem.dirichletbc(u_bc, boundary_dofs)

    problem = petsc.LinearProblem(
        a,
        L,
        bcs=[bc],
        petsc_options_prefix="helmholtz_",
        petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
    )
    uh = problem.solve()
    uh.x.scatter_forward()

    u_grid = _sample_on_grid(domain, uh, nx_out, ny_out, bbox)

    solver_info = {
        "mesh_resolution": mesh_resolution,
        "element_degree": element_degree,
        "ksp_type": "preonly",
        "pc_type": "lu",
        "rtol": 1.0e-12,
        "iterations": 1,
    }

    if rank == 0:
        xmin, xmax, ymin, ymax = bbox
        xs = np.linspace(xmin, xmax, nx_out)
        ys = np.linspace(ymin, ymax, ny_out)
        XX, YY = np.meshgrid(xs, ys, indexing="xy")
        u_exact_grid, _ = _manufactured_terms(XX, YY, k)
        solver_info["verification_l2_grid_error"] = float(
            np.sqrt(np.mean((u_grid - u_exact_grid) ** 2))
        )
        return {"u": u_grid, "solver_info": solver_info}

    return {"u": None, "solver_info": solver_info}
