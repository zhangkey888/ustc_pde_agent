import time
import math
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl

ScalarType = PETSc.ScalarType


def _sample_on_grid(domain, uh, grid_spec):
    nx = int(grid_spec["nx"])
    ny = int(grid_spec["ny"])
    xmin, xmax, ymin, ymax = grid_spec["bbox"]
    xs = np.linspace(xmin, xmax, nx, dtype=np.float64)
    ys = np.linspace(ymin, ymax, ny, dtype=np.float64)
    XX, YY = np.meshgrid(xs, ys, indexing="xy")
    pts = np.column_stack([XX.ravel(), YY.ravel(), np.zeros(nx * ny, dtype=np.float64)])

    tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(tree, pts)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, pts)

    local_vals = np.full(nx * ny, np.nan, dtype=np.float64)
    points_on_proc = []
    cells_on_proc = []
    ids = []
    for i in range(nx * ny):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(pts[i])
            cells_on_proc.append(links[0])
            ids.append(i)

    if points_on_proc:
        vals = uh.eval(np.array(points_on_proc, dtype=np.float64), np.array(cells_on_proc, dtype=np.int32))
        local_vals[np.array(ids, dtype=np.int32)] = np.asarray(vals).reshape(-1)

    gathered = domain.comm.gather(local_vals, root=0)
    if domain.comm.rank == 0:
        out = np.full(nx * ny, np.nan, dtype=np.float64)
        for arr in gathered:
            mask = ~np.isnan(arr)
            out[mask] = arr[mask]
        out[np.isnan(out)] = 0.0
        out = out.reshape(ny, nx)
    else:
        out = None
    return domain.comm.bcast(out, root=0)


def _solve_once(n, degree, eps_val, beta_vec):
    comm = MPI.COMM_WORLD
    msh = mesh.create_unit_square(comm, n, n, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(msh, ("Lagrange", degree))

    x = ufl.SpatialCoordinate(msh)
    pi = ufl.pi
    u_exact = ufl.sin(pi * x[0]) * ufl.sin(2.0 * pi * x[1])

    eps_c = fem.Constant(msh, ScalarType(eps_val))
    beta_c = fem.Constant(msh, np.array(beta_vec, dtype=np.float64))

    f_expr = -eps_c * ufl.div(ufl.grad(u_exact)) + ufl.dot(beta_c, ufl.grad(u_exact))

    u_bc = fem.Function(V)
    u_bc.interpolate(fem.Expression(u_exact, V.element.interpolation_points))

    fdim = msh.topology.dim - 1
    facets = mesh.locate_entities_boundary(msh, fdim, lambda X: np.ones(X.shape[1], dtype=bool))
    dofs = fem.locate_dofs_topological(V, fdim, facets)
    bc = fem.dirichletbc(u_bc, dofs)

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    h = ufl.CellDiameter(msh)
    bnorm = ufl.sqrt(ufl.dot(beta_c, beta_c) + 1.0e-30)
    Pe = bnorm * h / (2.0 * eps_c)
    tau = h / (2.0 * bnorm) * (ufl.cosh(Pe) / ufl.sinh(Pe) - 1.0 / Pe)

    a = eps_c * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx + ufl.dot(beta_c, ufl.grad(u)) * v * ufl.dx
    L = f_expr * v * ufl.dx

    residual_u = -eps_c * ufl.div(ufl.grad(u)) + ufl.dot(beta_c, ufl.grad(u))
    a += tau * ufl.dot(beta_c, ufl.grad(v)) * residual_u * ufl.dx
    L += tau * ufl.dot(beta_c, ufl.grad(v)) * f_expr * ufl.dx

    opts = {
        "ksp_type": "gmres",
        "pc_type": "ilu",
        "ksp_rtol": 1.0e-10,
        "ksp_atol": 1.0e-12,
        "ksp_max_it": 2000,
    }

    problem = petsc.LinearProblem(
        a, L, bcs=[bc], petsc_options=opts, petsc_options_prefix=f"cd_{n}_{degree}_"
    )
    uh = problem.solve()
    uh.x.scatter_forward()

    err = uh - u_bc
    l2_local = fem.assemble_scalar(fem.form(ufl.inner(err, err) * ufl.dx))
    l2 = math.sqrt(comm.allreduce(l2_local, op=MPI.SUM))

    ksp = problem.solver
    return msh, uh, {
        "mesh_resolution": n,
        "element_degree": degree,
        "ksp_type": str(ksp.getType()),
        "pc_type": str(ksp.getPC().getType()),
        "rtol": float(ksp.getTolerances()[0]),
        "iterations": int(ksp.getIterationNumber()),
        "L2_error_vs_manufactured": float(l2),
    }


def solve(case_spec: dict) -> dict:
    eps_val = float(case_spec.get("pde", {}).get("epsilon", 0.03))
    beta_vec = case_spec.get("pde", {}).get("beta", [5.0, 2.0])
    beta_vec = [float(beta_vec[0]), float(beta_vec[1])]

    degree = 2
    start = time.perf_counter()
    best = None
    for n in [40, 52, 64]:
        best = _solve_once(n, degree, eps_val, beta_vec)
        if time.perf_counter() - start > 1.6:
            break

    msh, uh, info = best
    u_grid = _sample_on_grid(msh, uh, case_spec["output"]["grid"])
    info["stabilization"] = "SUPG"

    return {"u": u_grid, "solver_info": info}
