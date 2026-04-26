import time, math, numpy as np
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
ScalarType = PETSc.ScalarType
DIAGNOSIS = "convection_diffusion 2D scalar linear steady highPe manufactured_solution"
METHOD = "fem Lagrange_P2 supg gmres ilu"
def _sample_function(u_func, bbox, nx, ny):
    msh = u_func.function_space.mesh
    xs = np.linspace(bbox[0], bbox[1], nx, dtype=np.float64); ys = np.linspace(bbox[2], bbox[3], ny, dtype=np.float64)
    XX, YY = np.meshgrid(xs, ys, indexing="xy")
    pts = np.column_stack([XX.ravel(), YY.ravel(), np.zeros(nx * ny, dtype=np.float64)])
    tree = geometry.bb_tree(msh, msh.topology.dim)
    cands = geometry.compute_collisions_points(tree, pts)
    coll = geometry.compute_colliding_cells(msh, cands, pts)
    vals = np.full(pts.shape[0], np.nan, dtype=np.float64); pl = []; cl = []; idx = []
    for i in range(pts.shape[0]):
        links = coll.links(i)
        if len(links) > 0: pl.append(pts[i]); cl.append(links[0]); idx.append(i)
    if pl:
        out = u_func.eval(np.array(pl, dtype=np.float64), np.array(cl, dtype=np.int32))
        vals[np.array(idx, dtype=np.int32)] = np.asarray(out).reshape(-1)
    gathered = msh.comm.allgather(vals); merged = gathered[0].copy()
    for a in gathered[1:]:
        m = np.isnan(merged) & ~np.isnan(a); merged[m] = a[m]
    if np.isnan(merged).any(): raise RuntimeError("Sampling failed")
    return merged.reshape(ny, nx)
def _solve_once(comm, eps, beta_np, N, degree, ksp_type, pc_type, rtol):
    msh = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(msh, ("Lagrange", degree))
    x = ufl.SpatialCoordinate(msh); uex = ufl.sin(ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])
    beta = ufl.as_vector((ScalarType(beta_np[0]), ScalarType(beta_np[1])))
    f = -eps * ufl.div(ufl.grad(uex)) + ufl.dot(beta, ufl.grad(uex))
    uD = fem.Function(V); uD.interpolate(fem.Expression(uex, V.element.interpolation_points))
    fdim = msh.topology.dim - 1
    facets = mesh.locate_entities_boundary(msh, fdim, lambda X: np.ones(X.shape[1], dtype=bool))
    dofs = fem.locate_dofs_topological(V, fdim, facets); bc = fem.dirichletbc(uD, dofs)
    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
    h = ufl.CellDiameter(msh); bnorm = ufl.sqrt(ufl.dot(beta, beta) + 1e-30)
    pe = bnorm * h / (2.0 * eps + 1e-30); z = 2.0 * pe
    coth = (ufl.exp(z) + 1.0) / (ufl.exp(z) - 1.0 + 1e-30)
    tau_stream = h / (2.0 * bnorm) * (coth - 1.0 / (pe + 1e-30)); tau = 1.5 * tau_stream
    r_trial = -eps * ufl.div(ufl.grad(u)) + ufl.dot(beta, ufl.grad(u)); supg_test = ufl.dot(beta, ufl.grad(v))
    a = (eps * ufl.inner(ufl.grad(u), ufl.grad(v)) + ufl.dot(beta, ufl.grad(u)) * v + tau * r_trial * supg_test) * ufl.dx
    L = (f * v + tau * f * supg_test) * ufl.dx
    opts = {"ksp_type": ksp_type, "pc_type": pc_type, "ksp_rtol": rtol, "ksp_atol": 1e-13, "ksp_max_it": 4000, "ksp_gmres_restart": 200}
    try:
        problem = petsc.LinearProblem(a, L, bcs=[bc], petsc_options_prefix=f"cdr_{N}_", petsc_options=opts)
        uh = problem.solve(); uh.x.scatter_forward()
        try: its = int(problem.solver.getIterationNumber())
        except Exception: its = 0
        used_ksp, used_pc = ksp_type, pc_type
    except Exception:
        problem = petsc.LinearProblem(a, L, bcs=[bc], petsc_options_prefix=f"cdrlu_{N}_", petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
        uh = problem.solve(); uh.x.scatter_forward(); its = 1; used_ksp, used_pc = "preonly", "lu"
    uexf = fem.Function(V); uexf.interpolate(fem.Expression(uex, V.element.interpolation_points))
    errf = fem.Function(V); errf.x.array[:] = uh.x.array - uexf.x.array; errf.x.scatter_forward()
    l2_local = fem.assemble_scalar(fem.form(ufl.inner(errf, errf) * ufl.dx)); l2 = math.sqrt(max(msh.comm.allreduce(l2_local, op=MPI.SUM), 0.0))
    return uh, l2, its, used_ksp, used_pc
def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD
    pde = case_spec.get("pde", {}); grid = case_spec["output"]["grid"]; solver_cfg = case_spec.get("solver", {})
    eps = float(pde.get("epsilon", 0.005)); beta_in = pde.get("beta", [20.0, 0.0]); beta_np = np.array([float(beta_in[0]), float(beta_in[1])], dtype=np.float64)
    nx, ny, bbox = int(grid["nx"]), int(grid["ny"]), grid["bbox"]
    degree = int(solver_cfg.get("element_degree", 2)); ksp_type = solver_cfg.get("ksp_type", "gmres"); pc_type = solver_cfg.get("pc_type", "ilu"); rtol = float(solver_cfg.get("rtol", 1e-9))
    candidates = [int(solver_cfg["mesh_resolution"])] if "mesh_resolution" in solver_cfg else [72, 96, 120]
    start = time.perf_counter(); best = None
    for N in candidates:
        uh, l2, its, used_ksp, used_pc = _solve_once(comm, eps, beta_np, N, degree, ksp_type, pc_type, rtol)
        sx, sy = min(nx, 80), min(ny, 80); ug = _sample_function(uh, bbox, sx, sy)
        xs = np.linspace(bbox[0], bbox[1], sx); ys = np.linspace(bbox[2], bbox[3], sy); XX, YY = np.meshgrid(xs, ys, indexing="xy")
        rmse = float(np.sqrt(np.mean((ug - np.sin(np.pi * XX) * np.sin(np.pi * YY)) ** 2)))
        best = {"uh": uh, "N": N, "its": its, "ksp": used_ksp, "pc": used_pc, "l2": l2, "rmse": rmse}
        if "mesh_resolution" in solver_cfg or time.perf_counter() - start > 1.15 or rmse < 2.2e-4: break
    u_grid = _sample_function(best["uh"], bbox, nx, ny)
    return {"u": u_grid, "solver_info": {"mesh_resolution": int(best["N"]), "element_degree": degree, "ksp_type": str(best["ksp"]), "pc_type": str(best["pc"]), "rtol": rtol, "iterations": int(best["its"]), "l2_error_verification": float(best["l2"])}} 
def _run_self_test():
    case = {"pde": {"epsilon": 0.005, "beta": [20.0, 0.0]}, "output": {"grid": {"nx": 128, "ny": 128, "bbox": [0.0, 1.0, 0.0, 1.0]}}}
    t0 = time.perf_counter(); out = solve(case); wall = time.perf_counter() - t0
    xs = np.linspace(0.0, 1.0, 128); ys = np.linspace(0.0, 1.0, 128); XX, YY = np.meshgrid(xs, ys, indexing="xy")
    uex = np.sin(math.pi * XX) * np.sin(math.pi * YY); err = float(np.sqrt(np.mean((out["u"] - uex) ** 2)))
    if MPI.COMM_WORLD.rank == 0: print(DIAGNOSIS); print(METHOD); print(f"GRID_RMSE: {err:.12e}"); print(f"WALL_TIME: {wall:.12e}"); print(out["solver_info"])
if __name__ == "__main__": _run_self_test()
