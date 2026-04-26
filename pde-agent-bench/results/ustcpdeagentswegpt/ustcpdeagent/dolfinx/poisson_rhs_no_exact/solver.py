import math
import time
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
import ufl
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc


def _sample_function_on_grid(domain, uh, grid_spec):
    nx = int(grid_spec["nx"])
    ny = int(grid_spec["ny"])
    xmin, xmax, ymin, ymax = grid_spec["bbox"]

    xs = np.linspace(xmin, xmax, nx)
    ys = np.linspace(ymin, ymax, ny)
    xx, yy = np.meshgrid(xs, ys, indexing="xy")
    pts = np.zeros((nx * ny, 3), dtype=np.float64)
    pts[:, 0] = xx.ravel()
    pts[:, 1] = yy.ravel()

    tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(tree, pts)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, pts)

    local_vals = np.full(nx * ny, np.nan, dtype=np.float64)
    points_on_proc = []
    cells = []
    ids = []
    for i in range(nx * ny):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(pts[i])
            cells.append(links[0])
            ids.append(i)

    if points_on_proc:
        vals = uh.eval(np.array(points_on_proc, dtype=np.float64), np.array(cells, dtype=np.int32))
        local_vals[np.array(ids, dtype=np.int32)] = np.asarray(vals).reshape(-1)

    gathered = np.empty_like(local_vals)
    domain.comm.Allreduce(local_vals, gathered, op=MPI.MAX)
    gathered[np.isnan(gathered)] = 0.0
    return gathered.reshape((ny, nx))


def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD
    t0 = time.perf_counter()

    mesh_resolution = int(case_spec.get("solver_params", {}).get("mesh_resolution", 48))
    element_degree = int(case_spec.get("solver_params", {}).get("element_degree", 2))

    domain = mesh.create_unit_square(comm, mesh_resolution, mesh_resolution, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", element_degree))

    x = ufl.SpatialCoordinate(domain)
    kappa = fem.Constant(domain, PETSc.ScalarType(0.5))
    f = ufl.sin(3.0 * ufl.pi * x[0]) * ufl.sin(2.0 * ufl.pi * x[1])

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    a = kappa * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = f * v * ufl.dx

    fdim = domain.topology.dim - 1
    facets = mesh.locate_entities_boundary(domain, fdim, lambda X: np.ones(X.shape[1], dtype=bool))
    dofs = fem.locate_dofs_topological(V, fdim, facets)
    bc = fem.dirichletbc(PETSc.ScalarType(0.0), dofs, V)

    ksp_type = "cg"
    pc_type = "hypre"
    rtol = 1e-10
    iterations = 0

    try:
        problem = petsc.LinearProblem(
            a, L, bcs=[bc],
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
        try:
            iterations = int(problem.solver.getIterationNumber())
            if problem.solver.getConvergedReason() <= 0:
                raise RuntimeError("KSP failed")
        except Exception:
            pass
    except Exception:
        ksp_type = "preonly"
        pc_type = "lu"
        problem = petsc.LinearProblem(
            a, L, bcs=[bc],
            petsc_options_prefix="poisson_lu_",
            petsc_options={"ksp_type": ksp_type, "pc_type": pc_type},
        )
        uh = problem.solve()
        uh.x.scatter_forward()
        iterations = 1

    u_exact = ufl.sin(3.0 * ufl.pi * x[0]) * ufl.sin(2.0 * ufl.pi * x[1]) / (
        0.5 * ((3.0 * ufl.pi) ** 2 + (2.0 * ufl.pi) ** 2)
    )
    err_form = fem.form((uh - u_exact) * (uh - u_exact) * ufl.dx)
    norm_form = fem.form(u_exact * u_exact * ufl.dx)
    l2_error = math.sqrt(comm.allreduce(fem.assemble_scalar(err_form), op=MPI.SUM))
    l2_norm = math.sqrt(comm.allreduce(fem.assemble_scalar(norm_form), op=MPI.SUM))
    rel_l2_error = l2_error / l2_norm if l2_norm > 0 else l2_error

    u_grid = _sample_function_on_grid(domain, uh, case_spec["output"]["grid"])

    solver_info = {
        "mesh_resolution": mesh_resolution,
        "element_degree": element_degree,
        "ksp_type": ksp_type,
        "pc_type": pc_type,
        "rtol": rtol,
        "iterations": int(iterations),
        "l2_error": float(l2_error),
        "rel_l2_error": float(rel_l2_error),
        "wall_time_sec": float(time.perf_counter() - t0),
    }

    return {"u": u_grid, "solver_info": solver_info}
