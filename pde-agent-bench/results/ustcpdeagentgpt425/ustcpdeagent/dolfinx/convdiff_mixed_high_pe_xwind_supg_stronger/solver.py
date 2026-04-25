import time, math, numpy as np
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl

ScalarType = PETSc.ScalarType

DIAGNOSIS = "DIAGNOSIS: convection_diffusion, 2D, scalar, linear, steady, high-Pe, all-Dirichlet, manufactured_solution"
METHOD = "METHOD: FEM, Lagrange_P2, SUPG, GMRES, ILU"

def _sample_function(u_func, bbox, nx, ny):
    msh = u_func.function_space.mesh
    xs = np.linspace(bbox[0], bbox[1], nx)
    ys = np.linspace(bbox[2], bbox[3], ny)
    XX, YY = np.meshgrid(xs, ys, indexing="xy")
    pts = np.column_stack([XX.ravel(), YY.ravel(), np.zeros(nx * ny, dtype=np.float64)])
    tree = geometry.bb_tree(msh, msh.topology.dim)
    cands = geometry.compute_collisions_points(tree, pts)
    coll = geometry.compute_colliding_cells(msh, cands, pts)
    vals = np.full(pts.shape[0], np.nan, dtype=np.float64)
    p_local, c_local, idx = [], [], []
    for i in range(pts.shape[0]):
        links = coll.links(i)
        if len(links) > 0:
            p_local.append(pts[i]); c_local.append(links[0]); idx.append(i)
    if p_local:
        out = u_func.eval(np.array(p_local, dtype=np.float64), np.array(c_local, dtype=np.int32))
        vals[np.array(idx, dtype=np.int32)] = np.asarray(out).reshape(-1)
    gathered = msh.comm.allgather(vals)
    merged = gathered[0].copy()
    for a in gathered[1:]:
        m = np.isnan(merged) & ~np.isnan(a)
        merged[m] = a[m]
    if np.isnan(merged).any():
        raise RuntimeError("Sampling failed")
    return merged.reshape(ny, nx)

def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD
    pde = case_spec.get("pde", {})
    eps = float(pde.get("epsilon", 0.005))
    beta_in = pde.get("beta", [20.0, 0.0])
    beta0, beta1 = float(beta_in[0]), float(beta_in[1])
    solver_cfg = case_spec.get("solver", {})
    N = int(solver_cfg.get("mesh_resolution", 96))
    degree = int(solver_cfg.get("element_degree", 2))
    ksp_type = solver_cfg.get("ksp_type", "gmres")
    pc_type = solver_cfg.get("pc_type", "ilu")
    rtol = float(solver_cfg.get("rtol", 1e-9))

    msh = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(msh, ("Lagrange", degree))
    x = ufl.SpatialCoordinate(msh)
    pi = ufl.pi
    uex = ufl.sin(pi * x[0]) * ufl.sin(pi * x[1])
    beta = ufl.as_vector((ScalarType(beta0), ScalarType(beta1)))
    f = -eps * ufl.div(ufl.grad(uex)) + ufl.dot(beta, ufl.grad(uex))

    uD = fem.Function(V)
    uD.interpolate(fem.Expression(uex, V.element.interpolation_points))
    fdim = msh.topology.dim - 1
    facets = mesh.locate_entities_boundary(msh, fdim, lambda X: np.ones(X.shape[1], dtype=bool))
    dofs = fem.locate_dofs_topological(V, fdim, facets)
    bc = fem.dirichletbc(uD, dofs)

    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
    h = ufl.CellDiameter(msh)
    bnorm = ufl.sqrt(ufl.dot(beta, beta))
    Pe = bnorm * h / (2.0 * eps + 1e-16)
    z = 2.0 * Pe
    cothPe = (ufl.exp(z) + 1.0) / (ufl.exp(z) - 1.0 + 1e-16)
    tau = h / (2.0 * bnorm + 1e-16) * (cothPe - 1.0 / (Pe + 1e-16))

    strong_u = -eps * ufl.div(ufl.grad(u)) + ufl.dot(beta, ufl.grad(u))
    a = (eps * ufl.inner(ufl.grad(u), ufl.grad(v)) + ufl.dot(beta, ufl.grad(u)) * v) * ufl.dx + tau * strong_u * ufl.dot(beta, ufl.grad(v)) * ufl.dx
    L = (f * v) * ufl.dx + tau * f * ufl.dot(beta, ufl.grad(v)) * ufl.dx

    problem = petsc.LinearProblem(
        a, L, bcs=[bc], petsc_options_prefix="cdr_",
        petsc_options={"ksp_type": ksp_type, "pc_type": pc_type, "ksp_rtol": rtol, "ksp_atol": 1e-12, "ksp_max_it": 4000, "ksp_gmres_restart": 200}
    )
    uh = problem.solve()
    uh.x.scatter_forward()
    its = 0
    try:
        its = int(problem.solver.getIterationNumber())
    except Exception:
        pass

    grid = case_spec["output"]["grid"]
    u_grid = _sample_function(uh, grid["bbox"], int(grid["nx"]), int(grid["ny"]))
    return {"u": u_grid, "solver_info": {"mesh_resolution": N, "element_degree": degree, "ksp_type": ksp_type, "pc_type": pc_type, "rtol": rtol, "iterations": its}}

def _run_self_test():
    case = {"pde": {"epsilon": 0.005, "beta": [20.0, 0.0]}, "output": {"grid": {"nx": 128, "ny": 128, "bbox": [0.0, 1.0, 0.0, 1.0]}}}
    t0 = time.perf_counter()
    out = solve(case)
    wall = time.perf_counter() - t0
    nx, ny = case["output"]["grid"]["nx"], case["output"]["grid"]["ny"]
    bbox = case["output"]["grid"]["bbox"]
    xs = np.linspace(bbox[0], bbox[1], nx)
    ys = np.linspace(bbox[2], bbox[3], ny)
    XX, YY = np.meshgrid(xs, ys, indexing="xy")
    u_exact = np.sin(math.pi * XX) * np.sin(math.pi * YY)
    err = np.sqrt(np.mean((out["u"] - u_exact) ** 2))
    if MPI.COMM_WORLD.rank == 0:
        print(DIAGNOSIS)
        print(METHOD)
        print(f"L2_ERROR: {err:.12e}")
        print(f"WALL_TIME: {wall:.12e}")
        print(out["solver_info"])

if __name__ == "__main__":
    _run_self_test()
