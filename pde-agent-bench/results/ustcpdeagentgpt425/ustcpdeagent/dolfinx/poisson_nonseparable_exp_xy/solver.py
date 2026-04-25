import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl

ScalarType = PETSc.ScalarType


def _build_problem(nx: int, degree: int):
    domain = mesh.create_unit_square(MPI.COMM_WORLD, nx, nx, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", degree))

    x = ufl.SpatialCoordinate(domain)
    pi = ufl.pi
    u_exact = ufl.exp(x[0] * x[1]) * ufl.sin(pi * x[0]) * ufl.sin(pi * x[1])
    f = -ufl.div(ufl.grad(u_exact))

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    a = ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = ufl.inner(f, v) * ufl.dx

    fdim = domain.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda X: np.ones(X.shape[1], dtype=bool)
    )
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)

    u_bc = fem.Function(V)
    u_bc.interpolate(fem.Expression(u_exact, V.element.interpolation_points))
    bc = fem.dirichletbc(u_bc, boundary_dofs)

    return domain, a, L, bc


def _solve_linear(a, L, bc):
    try:
        problem = petsc.LinearProblem(
            a,
            L,
            bcs=[bc],
            petsc_options_prefix="poisson_",
            petsc_options={
                "ksp_type": "cg",
                "pc_type": "hypre",
                "ksp_rtol": 1e-10,
            },
        )
        uh = problem.solve()
        return uh, "cg", "hypre", 1e-10, int(problem.solver.getIterationNumber())
    except Exception:
        problem = petsc.LinearProblem(
            a,
            L,
            bcs=[bc],
            petsc_options_prefix="poisson_lu_",
            petsc_options={
                "ksp_type": "preonly",
                "pc_type": "lu",
            },
        )
        uh = problem.solve()
        return uh, "preonly", "lu", 1e-10, int(problem.solver.getIterationNumber())


def _sample_on_grid(u_func, bbox, nx, ny):
    domain = u_func.function_space.mesh
    xs = np.linspace(bbox[0], bbox[1], nx)
    ys = np.linspace(bbox[2], bbox[3], ny)
    XX, YY = np.meshgrid(xs, ys, indexing="xy")
    pts = np.column_stack([XX.ravel(), YY.ravel(), np.zeros(nx * ny)])

    tree = geometry.bb_tree(domain, domain.topology.dim)
    candidates = geometry.compute_collisions_points(tree, pts)
    colliding = geometry.compute_colliding_cells(domain, candidates, pts)

    values = np.full(nx * ny, np.nan, dtype=np.float64)
    points_on_proc = []
    cells_on_proc = []
    ids = []

    for i in range(pts.shape[0]):
        links = colliding.links(i)
        if len(links) > 0:
            points_on_proc.append(pts[i])
            cells_on_proc.append(links[0])
            ids.append(i)

    if points_on_proc:
        vals = u_func.eval(
            np.asarray(points_on_proc, dtype=np.float64),
            np.asarray(cells_on_proc, dtype=np.int32),
        )
        vals = np.asarray(vals).reshape(len(points_on_proc), -1)[:, 0]
        values[np.asarray(ids, dtype=np.int32)] = vals

    if domain.comm.size > 1:
        global_vals = np.empty_like(values)
        domain.comm.Allreduce(values, global_vals, op=MPI.SUM)
        values = global_vals

    return values.reshape(ny, nx)


def solve(case_spec: dict) -> dict:
    grid = case_spec["output"]["grid"]
    nx_out = int(grid["nx"])
    ny_out = int(grid["ny"])
    bbox = grid["bbox"]

    mesh_resolution = 40
    element_degree = 2

    domain, a, L, bc = _build_problem(mesh_resolution, element_degree)
    uh, ksp_type, pc_type, rtol, iterations = _solve_linear(a, L, bc)
    uh.x.scatter_forward()

    u_grid = _sample_on_grid(uh, bbox, nx_out, ny_out)

    xs = np.linspace(bbox[0], bbox[1], nx_out)
    ys = np.linspace(bbox[2], bbox[3], ny_out)
    XX, YY = np.meshgrid(xs, ys, indexing="xy")
    u_exact_grid = np.exp(XX * YY) * np.sin(np.pi * XX) * np.sin(np.pi * YY)

    solver_info = {
        "mesh_resolution": mesh_resolution,
        "element_degree": element_degree,
        "ksp_type": ksp_type,
        "pc_type": pc_type,
        "rtol": rtol,
        "iterations": iterations,
        "verification_max_abs_error_on_output_grid": float(np.max(np.abs(u_grid - u_exact_grid))),
        "verification_rmse_on_output_grid": float(np.sqrt(np.mean((u_grid - u_exact_grid) ** 2))),
    }

    return {"u": u_grid, "solver_info": solver_info}
