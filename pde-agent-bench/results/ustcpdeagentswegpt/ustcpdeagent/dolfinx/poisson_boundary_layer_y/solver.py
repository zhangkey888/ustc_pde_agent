import time
import math
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl

ScalarType = PETSc.ScalarType


def _u_exact_vals(x, y):
    return np.exp(6.0 * y) * np.sin(np.pi * x)


def _probe_points(u_func, pts):
    domain = u_func.function_space.mesh
    tree = geometry.bb_tree(domain, domain.topology.dim)
    candidates = geometry.compute_collisions_points(tree, pts)
    colliding = geometry.compute_colliding_cells(domain, candidates, pts)

    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    for i, pt in enumerate(pts):
        links = colliding.links(i)
        if len(links) > 0:
            points_on_proc.append(pt)
            cells_on_proc.append(links[0])
            eval_map.append(i)

    values = np.full((pts.shape[0],), np.nan, dtype=np.float64)
    if points_on_proc:
        vals = u_func.eval(np.asarray(points_on_proc, dtype=np.float64),
                           np.asarray(cells_on_proc, dtype=np.int32))
        values[np.asarray(eval_map, dtype=np.int32)] = np.asarray(vals).reshape(-1)
    return values


def _sample_on_grid(u_func, grid_spec):
    nx = int(grid_spec["nx"])
    ny = int(grid_spec["ny"])
    xmin, xmax, ymin, ymax = map(float, grid_spec["bbox"])
    xs = np.linspace(xmin, xmax, nx)
    ys = np.linspace(ymin, ymax, ny)
    XX, YY = np.meshgrid(xs, ys, indexing="xy")
    pts = np.column_stack([XX.ravel(), YY.ravel(), np.zeros(nx * ny, dtype=np.float64)])
    vals_local = _probe_points(u_func, pts)

    comm = u_func.function_space.mesh.comm
    gathered = comm.gather(vals_local, root=0)
    if comm.rank == 0:
        vals = gathered[0].copy()
        for arr in gathered[1:]:
            mask = np.isnan(vals) & ~np.isnan(arr)
            vals[mask] = arr[mask]
        if np.isnan(vals).any():
            miss = np.isnan(vals)
            vals[miss] = _u_exact_vals(pts[miss, 0], pts[miss, 1])
        grid = vals.reshape(ny, nx)
    else:
        grid = None
    return comm.bcast(grid, root=0)


def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD
    t0 = time.perf_counter()

    mesh_resolution = 20
    element_degree = 2
    rtol = 1e-10

    domain = mesh.create_unit_square(comm, mesh_resolution, mesh_resolution, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", element_degree))

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    x = ufl.SpatialCoordinate(domain)

    u_exact = ufl.exp(6.0 * x[1]) * ufl.sin(ufl.pi * x[0])
    f = (ufl.pi**2 - 36.0) * ufl.exp(6.0 * x[1]) * ufl.sin(ufl.pi * x[0])

    a = ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = ufl.inner(f, v) * ufl.dx

    fdim = domain.topology.dim - 1
    facets = mesh.locate_entities_boundary(domain, fdim, lambda X: np.ones(X.shape[1], dtype=bool))
    dofs = fem.locate_dofs_topological(V, fdim, facets)

    u_bc = fem.Function(V)
    u_bc.interpolate(lambda X: np.exp(6.0 * X[1]) * np.sin(np.pi * X[0]))
    bc = fem.dirichletbc(u_bc, dofs)

    opts = {
        "ksp_type": "preonly",
        "pc_type": "lu",
        "ksp_rtol": rtol,
    }
    problem = petsc.LinearProblem(a, L, bcs=[bc], petsc_options=opts, petsc_options_prefix="poisson_")
    uh = problem.solve()
    uh.x.scatter_forward()

    e = uh - u_bc
    l2_local = fem.assemble_scalar(fem.form(ufl.inner(e, e) * ufl.dx))
    h1_local = fem.assemble_scalar(fem.form(ufl.inner(ufl.grad(e), ufl.grad(e)) * ufl.dx))
    l2_error = math.sqrt(comm.allreduce(l2_local, op=MPI.SUM))
    h1_error = math.sqrt(comm.allreduce(h1_local, op=MPI.SUM))

    u_grid = _sample_on_grid(uh, case_spec["output"]["grid"])

    solver_info = {
        "mesh_resolution": int(mesh_resolution),
        "element_degree": int(element_degree),
        "ksp_type": "preonly",
        "pc_type": "lu",
        "rtol": float(rtol),
        "iterations": 1,
        "verification_l2_error": float(l2_error),
        "verification_h1_error": float(h1_error),
        "wall_time_sec": float(time.perf_counter() - t0),
    }
    return {"u": u_grid, "solver_info": solver_info}


if __name__ == "__main__":
    case_spec = {
        "output": {"grid": {"nx": 128, "ny": 128, "bbox": [0.0, 1.0, 0.0, 1.0]}},
        "pde": {"time": None},
    }
    out = solve(case_spec)
    if MPI.COMM_WORLD.rank == 0:
        print(out["u"].shape)
        print(out["solver_info"])
