import time
import numpy as np
from mpi4py import MPI
import ufl
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc

def _defaults(case_spec):
    case_spec = {} if case_spec is None else dict(case_spec)
    case_spec.setdefault("output", {})
    case_spec["output"].setdefault("grid", {})
    g = case_spec["output"]["grid"]
    g.setdefault("nx", 129); g.setdefault("ny", 129); g.setdefault("bbox", [0.0, 1.0, 0.0, 1.0])
    return case_spec

def _sample(domain, uh, grid):
    nx, ny = int(grid["nx"]), int(grid["ny"])
    xmin, xmax, ymin, ymax = map(float, grid["bbox"])
    xs = np.linspace(xmin, xmax, nx); ys = np.linspace(ymin, ymax, ny)
    XX, YY = np.meshgrid(xs, ys, indexing="xy")
    pts = np.column_stack([XX.ravel(), YY.ravel(), np.zeros(nx * ny)])
    tree = geometry.bb_tree(domain, domain.topology.dim)
    cand = geometry.compute_collisions_points(tree, pts)
    coll = geometry.compute_colliding_cells(domain, cand, pts)
    out = np.full(nx * ny, np.nan)
    p = []; c = []; ids = []
    for i in range(pts.shape[0]):
        links = coll.links(i)
        if len(links) > 0:
            p.append(pts[i]); c.append(links[0]); ids.append(i)
    if p:
        vals = uh.eval(np.array(p, dtype=np.float64), np.array(c, dtype=np.int32))
        out[np.array(ids, dtype=np.int32)] = np.linalg.norm(vals, axis=1)
    gathered = domain.comm.gather(out, root=0)
    if domain.comm.rank == 0:
        final = np.full(nx * ny, np.nan)
        for arr in gathered:
            m = ~np.isnan(arr); final[m] = arr[m]
        if np.isnan(final).any():
            raise RuntimeError("Point evaluation failed")
        final = final.reshape(ny, nx)
    else:
        final = None
    return domain.comm.bcast(final, root=0)

def _solve_level(case_spec, n, degree, rtol):
    case_spec = _defaults(case_spec)
    comm = MPI.COMM_WORLD
    t0 = time.perf_counter()
    domain = mesh.create_unit_square(comm, n, n, cell_type=mesh.CellType.triangle)
    gdim = domain.geometry.dim
    V = fem.functionspace(domain, ("Lagrange", degree, (gdim,)))
    E = 1.0; nu = 0.3
    mu = E / (2.0 * (1.0 + nu))
    lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    def eps(w): return ufl.sym(ufl.grad(w))
    def sigma(w): return 2.0 * mu * eps(w) + lam * ufl.tr(eps(w)) * ufl.Identity(gdim)
    u = ufl.TrialFunction(V); v = ufl.TestFunction(V); x = ufl.SpatialCoordinate(domain)
    uD_expr = ufl.as_vector((ufl.sin(ufl.pi * x[1]), 0.0 * x[0]))
    uD = fem.Function(V)
    uD.interpolate(fem.Expression(uD_expr, V.element.interpolation_points))
    f = fem.Constant(domain, np.array((0.0, 0.0), dtype=np.float64))
    fdim = domain.topology.dim - 1
    facets = mesh.locate_entities_boundary(domain, fdim, lambda X: np.ones(X.shape[1], dtype=bool))
    dofs = fem.locate_dofs_topological(V, fdim, facets)
    bc = fem.dirichletbc(uD, dofs)
    a = ufl.inner(sigma(u), eps(v)) * ufl.Measure("dx", domain=domain)
    L = ufl.inner(f, v) * ufl.Measure("dx", domain=domain)
    cfgs = [
        {"ksp_type": "cg", "pc_type": "hypre", "ksp_rtol": rtol, "ksp_atol": 1.0e-14, "ksp_max_it": 10000},
        {"ksp_type": "gmres", "pc_type": "hypre", "ksp_rtol": rtol, "ksp_atol": 1.0e-14, "ksp_max_it": 10000},
        {"ksp_type": "preonly", "pc_type": "lu"},
    ]
    last = None
    for i, opts in enumerate(cfgs):
        try:
            prob = petsc.LinearProblem(a, L, bcs=[bc], petsc_options=opts, petsc_options_prefix=f"elas_{i}_")
            uh = prob.solve()
            uh.x.scatter_forward()
            ksp = prob.solver
            break
        except Exception as e:
            last = e
    else:
        raise last
    exact = fem.Function(V)
    exact.interpolate(fem.Expression(uD_expr, V.element.interpolation_points))
    err_form = fem.form(ufl.inner(uh - exact, uh - exact) * ufl.Measure("dx", domain=domain))
    err = np.sqrt(comm.allreduce(fem.assemble_scalar(err_form), op=MPI.SUM))
    u_grid = _sample(domain, uh, case_spec["output"]["grid"])
    wall = time.perf_counter() - t0
    return {"u": u_grid, "solver_info": {"mesh_resolution": n, "element_degree": degree, "ksp_type": str(ksp.getType()), "pc_type": str(ksp.getPC().getType()), "rtol": rtol, "iterations": int(ksp.getIterationNumber()), "l2_error_check": float(err), "wall_time_check": float(wall)}}
def solve(case_spec):
    case_spec = _defaults(case_spec)
    degree = 2
    rtol = 1.0e-10
    coarse_n = 96
    fine_n = 160
    coarse = _solve_level(case_spec, coarse_n, degree, rtol)
    fine = _solve_level(case_spec, fine_n, degree, rtol)
    diff = fine["u"] - coarse["u"]
    conv_err = float(np.sqrt(np.mean(diff * diff)))
    info = dict(fine["solver_info"])
    info["iterations"] = int(coarse["solver_info"]["iterations"] + fine["solver_info"]["iterations"])
    info["mesh_resolution"] = fine_n
    info["convergence_error_check"] = conv_err
    info["wall_time_check"] = float(coarse["solver_info"]["wall_time_check"] + fine["solver_info"]["wall_time_check"])
    return {"u": fine["u"], "solver_info": info}


if __name__ == "__main__":
    out = solve({"output": {"grid": {"nx": 129, "ny": 129, "bbox": [0.0, 1.0, 0.0, 1.0]}}})
    if MPI.COMM_WORLD.rank == 0:
        print(f"L2_ERROR: {out['solver_info']['convergence_error_check']:.12e}")
        print(f"WALL_TIME: {out['solver_info']['wall_time_check']:.6f}")
