import time
import math
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl

ScalarType = PETSc.ScalarType

DIAGNOSIS_CARD = "DIAGNOSIS: convection_diffusion, 2D, rectangle, scalar, linear, steady, high-Pe, all_dirichlet, manufactured_solution"
METHOD_CARD = "METHOD: fem, Lagrange_P2, supg, steady, linear, gmres, ilu"

def _make_exact_and_rhs(msh, epsilon, beta):
    x = ufl.SpatialCoordinate(msh)
    u_exact = ufl.sin(2 * ufl.pi * x[0]) * ufl.sin(2 * ufl.pi * x[1])
    f_expr = -epsilon * ufl.div(ufl.grad(u_exact)) + ufl.dot(beta, ufl.grad(u_exact))
    return u_exact, f_expr

def _build_solver(nx, degree=2, epsilon=0.02, beta_vec=(6.0, 2.0), rtol=1e-9):
    msh = mesh.create_unit_square(MPI.COMM_WORLD, nx, nx, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(msh, ("Lagrange", degree))
    beta = fem.Constant(msh, np.array(beta_vec, dtype=np.float64))
    eps_c = fem.Constant(msh, ScalarType(epsilon))
    u_exact_ufl, f_ufl = _make_exact_and_rhs(msh, eps_c, beta)

    uD = fem.Function(V)
    uD.interpolate(fem.Expression(u_exact_ufl, V.element.interpolation_points))

    fdim = msh.topology.dim - 1
    facets = mesh.locate_entities_boundary(msh, fdim, lambda X: np.ones(X.shape[1], dtype=bool))
    bdofs = fem.locate_dofs_topological(V, fdim, facets)
    bc = fem.dirichletbc(uD, bdofs)

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    h = ufl.CellDiameter(msh)
    beta_norm = ufl.sqrt(ufl.dot(beta, beta) + 1.0e-16)
    pe = beta_norm * h / (2.0 * eps_c + 1.0e-16)
    tau = h / (2.0 * beta_norm) * (ufl.cosh(pe) / ufl.sinh(pe) - 1.0 / pe)

    a = (
        eps_c * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
        + ufl.dot(beta, ufl.grad(u)) * v * ufl.dx
        + tau * ufl.dot(beta, ufl.grad(v)) * (-eps_c * ufl.div(ufl.grad(u)) + ufl.dot(beta, ufl.grad(u))) * ufl.dx
    )
    L = f_ufl * v * ufl.dx + tau * ufl.dot(beta, ufl.grad(v)) * f_ufl * ufl.dx

    problem = petsc.LinearProblem(
        a, L, bcs=[bc],
        petsc_options_prefix=f"convdiff_{nx}_",
        petsc_options={"ksp_type": "gmres", "ksp_rtol": rtol, "pc_type": "ilu"},
    )
    uh = problem.solve()
    uh.x.scatter_forward()
    ksp = problem.solver
    return msh, uh, u_exact_ufl, {"iterations": int(ksp.getIterationNumber()), "ksp_type": ksp.getType(), "pc_type": ksp.getPC().getType()}

def _l2_error(msh, uh, u_exact_ufl):
    err2_local = fem.assemble_scalar(fem.form((uh - u_exact_ufl) ** 2 * ufl.dx))
    err2 = msh.comm.allreduce(err2_local, op=MPI.SUM)
    return math.sqrt(err2)

def _sample_on_grid(msh, uh, grid):
    nx = int(grid["nx"])
    ny = int(grid["ny"])
    xmin, xmax, ymin, ymax = grid["bbox"]
    xs = np.linspace(xmin, xmax, nx)
    ys = np.linspace(ymin, ymax, ny)
    XX, YY = np.meshgrid(xs, ys, indexing="xy")
    pts = np.column_stack([XX.ravel(), YY.ravel(), np.zeros(nx * ny)])
    tree = geometry.bb_tree(msh, msh.topology.dim)
    candidates = geometry.compute_collisions_points(tree, pts)
    colliding = geometry.compute_colliding_cells(msh, candidates, pts)

    values = np.full(nx * ny, np.nan, dtype=np.float64)
    local_pts, local_cells, ids = [], [], []
    for i in range(pts.shape[0]):
        links = colliding.links(i)
        if len(links) > 0:
            local_pts.append(pts[i])
            local_cells.append(links[0])
            ids.append(i)
    if local_pts:
        vals = uh.eval(np.array(local_pts, dtype=np.float64), np.array(local_cells, dtype=np.int32))
        values[np.array(ids, dtype=np.int32)] = np.asarray(vals).reshape(len(local_pts), -1)[:, 0]
    values = np.where(np.isnan(values), 0.0, values)
    values = msh.comm.allreduce(values, op=MPI.SUM)
    return values.reshape(ny, nx)

def solve(case_spec: dict) -> dict:
    pde = case_spec.get("pde", {})
    grid = case_spec.get("output", {}).get("grid", {"nx": 64, "ny": 64, "bbox": [0.0, 1.0, 0.0, 1.0]})
    epsilon = float(pde.get("epsilon", 0.02))
    beta_in = pde.get("beta", [6.0, 2.0])
    beta_vec = (float(beta_in[0]), float(beta_in[1]))
    wall_budget = float(case_spec.get("time_limit", 11.717))
    adaptive_budget = 0.5 * wall_budget

    configs = [(48, 2, 1e-8), (64, 2, 1e-8), (80, 2, 1e-9), (96, 2, 1e-9), (112, 2, 1e-9)]
    chosen = None
    t_start = time.perf_counter()
    for nx, degree, rtol in configs:
        t0 = time.perf_counter()
        msh, uh, u_exact_ufl, info = _build_solver(nx=nx, degree=degree, epsilon=epsilon, beta_vec=beta_vec, rtol=rtol)
        l2 = _l2_error(msh, uh, u_exact_ufl)
        chosen = (msh, uh, nx, degree, rtol, info, l2)
        elapsed = time.perf_counter() - t0
        if elapsed > adaptive_budget or (time.perf_counter() - t_start) > adaptive_budget:
            break

    msh, uh, nx, degree, rtol, info, l2 = chosen
    u_grid = _sample_on_grid(msh, uh, grid)
    solver_info = {
        "mesh_resolution": int(nx),
        "element_degree": int(degree),
        "ksp_type": str(info["ksp_type"]),
        "pc_type": str(info["pc_type"]),
        "rtol": float(rtol),
        "iterations": int(info["iterations"]),
    }
    return {"u": u_grid, "solver_info": solver_info, "diagnosis": DIAGNOSIS_CARD, "method": METHOD_CARD, "l2_error": float(l2)}

if __name__ == "__main__":
    case_spec = {
        "pde": {"epsilon": 0.02, "beta": [6.0, 2.0], "time": None},
        "output": {"grid": {"nx": 64, "ny": 64, "bbox": [0.0, 1.0, 0.0, 1.0]}},
        "time_limit": 11.717,
    }
    t0 = time.perf_counter()
    result = solve(case_spec)
    wall = time.perf_counter() - t0
    print("L2_ERROR:", result["l2_error"])
    print("WALL_TIME:", wall)
    print("U_SHAPE:", result["u"].shape)
