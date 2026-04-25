import math
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl

ScalarType = PETSc.ScalarType


def _boundary_values(x):
    return np.sin(3.0 * np.pi * x[0]) + np.cos(2.0 * np.pi * x[1])


def _sample_function_to_grid(domain, uh, grid_spec):
    nx = int(grid_spec["nx"])
    ny = int(grid_spec["ny"])
    xmin, xmax, ymin, ymax = map(float, grid_spec["bbox"])
    xs = np.linspace(xmin, xmax, nx)
    ys = np.linspace(ymin, ymax, ny)
    XX, YY = np.meshgrid(xs, ys, indexing="xy")
    pts = np.column_stack([XX.ravel(), YY.ravel(), np.zeros(nx * ny, dtype=np.float64)])

    tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(tree, pts)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, pts)

    local_vals = np.full(nx * ny, np.nan, dtype=np.float64)
    points_on_proc, cells, eval_map = [], [], []
    for i in range(pts.shape[0]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(pts[i])
            cells.append(links[0])
            eval_map.append(i)

    if points_on_proc:
        vals = uh.eval(np.array(points_on_proc, dtype=np.float64), np.array(cells, dtype=np.int32))
        local_vals[np.asarray(eval_map, dtype=np.int32)] = np.real(np.asarray(vals).reshape(-1))

    gathered = domain.comm.gather(local_vals, root=0)
    if domain.comm.rank == 0:
        combined = np.full(nx * ny, np.nan, dtype=np.float64)
        for arr in gathered:
            mask = np.isfinite(arr)
            combined[mask] = arr[mask]
        return combined.reshape(ny, nx)
    return None


def _solve_problem(mesh_resolution, degree, k, manufactured=False):
    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_square(comm, mesh_resolution, mesh_resolution, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", degree))

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    x = ufl.SpatialCoordinate(domain)

    fdim = domain.topology.dim - 1
    facets = mesh.locate_entities_boundary(domain, fdim, lambda X: np.ones(X.shape[1], dtype=bool))
    dofs = fem.locate_dofs_topological(V, fdim, facets)
    u_bc = fem.Function(V)
    u_bc.interpolate(_boundary_values)
    bc = fem.dirichletbc(u_bc, dofs)

    if manufactured:
        f = ((3.0 * np.pi) ** 2 - k**2) * ufl.sin(3.0 * ufl.pi * x[0]) + ((2.0 * np.pi) ** 2 - k**2) * ufl.cos(2.0 * ufl.pi * x[1])
    else:
        f = fem.Constant(domain, ScalarType(0.0))

    a = (ufl.inner(ufl.grad(u), ufl.grad(v)) - (k**2) * ufl.inner(u, v)) * ufl.dx
    L = ufl.inner(f, v) * ufl.dx

    problem = petsc.LinearProblem(
        a,
        L,
        bcs=[bc],
        petsc_options_prefix="helmholtz_",
        petsc_options={
            "ksp_type": "preonly",
            "pc_type": "lu",
        },
    )
    uh = problem.solve()
    uh.x.scatter_forward()

    solver_info = {
        "mesh_resolution": int(mesh_resolution),
        "element_degree": int(degree),
        "ksp_type": "preonly",
        "pc_type": "lu",
        "rtol": 1.0e-12,
        "iterations": 1,
    }
    return domain, uh, solver_info


def _verification_report(k):
    comm = MPI.COMM_WORLD
    n = 20
    domain, uh, _ = _solve_problem(n, 2, k, manufactured=True)
    V = uh.function_space
    x = ufl.SpatialCoordinate(domain)
    u_exact = fem.Function(V)
    u_exact.interpolate(fem.Expression(ufl.sin(3.0 * ufl.pi * x[0]) + ufl.cos(2.0 * ufl.pi * x[1]), V.element.interpolation_points))
    err = fem.Function(V)
    err.x.array[:] = uh.x.array - u_exact.x.array
    l2_local = fem.assemble_scalar(fem.form(ufl.inner(err, err) * ufl.dx))
    l2_err = math.sqrt(comm.allreduce(l2_local, op=MPI.SUM))
    return {"type": "manufactured_solution_l2", "mesh_resolution": n, "l2_error": float(l2_err)}


def solve(case_spec: dict) -> dict:
    k = float(case_spec.get("pde", {}).get("wavenumber", 12.0))
    grid_spec = case_spec["output"]["grid"]

    verification = _verification_report(k)

    mesh_resolution = 56
    degree = 2
    domain, uh, solver_info = _solve_problem(mesh_resolution, degree, k, manufactured=False)
    u_grid = _sample_function_to_grid(domain, uh, grid_spec)

    if domain.comm.rank == 0:
        solver_info["verification"] = verification
        return {"u": u_grid, "solver_info": solver_info}
    return {"u": None, "solver_info": solver_info}
