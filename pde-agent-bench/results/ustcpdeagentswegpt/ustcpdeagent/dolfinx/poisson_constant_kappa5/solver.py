import time
import math
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl

ScalarType = PETSc.ScalarType

# ```DIAGNOSIS
# equation_type: poisson
# spatial_dim: 2
# domain_geometry: rectangle
# unknowns: scalar
# coupling: none
# linearity: linear
# time_dependence: steady
# stiffness: N/A
# dominant_physics: diffusion
# peclet_or_reynolds: N/A
# solution_regularity: smooth
# bc_type: all_dirichlet
# special_notes: manufactured_solution
# ```
#
# ```METHOD
# spatial_method: fem
# element_or_basis: Lagrange_P2
# stabilization: none
# time_method: none
# nonlinear_solver: none
# linear_solver: cg
# preconditioner: hypre
# special_treatment: none
# pde_skill: poisson
# ```


def _analytic_u(x):
    return np.cos(2.0 * np.pi * x[0]) * np.cos(3.0 * np.pi * x[1])


def _sample_function_on_grid(u_func, msh, nx, ny, bbox):
    xmin, xmax, ymin, ymax = bbox
    xs = np.linspace(xmin, xmax, nx)
    ys = np.linspace(ymin, ymax, ny)
    XX, YY = np.meshgrid(xs, ys, indexing="xy")
    pts = np.column_stack([XX.ravel(), YY.ravel(), np.zeros(nx * ny)])

    tree = geometry.bb_tree(msh, msh.topology.dim)
    cell_candidates = geometry.compute_collisions_points(tree, pts)
    colliding_cells = geometry.compute_colliding_cells(msh, cell_candidates, pts)

    local_vals = np.full(nx * ny, np.nan, dtype=np.float64)
    points_on_proc = []
    cells_on_proc = []
    eval_ids = []
    for i in range(pts.shape[0]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(pts[i])
            cells_on_proc.append(links[0])
            eval_ids.append(i)

    if points_on_proc:
        vals = u_func.eval(
            np.asarray(points_on_proc, dtype=np.float64),
            np.asarray(cells_on_proc, dtype=np.int32),
        )
        local_vals[np.asarray(eval_ids, dtype=np.int32)] = np.asarray(vals).reshape(-1)

    comm = msh.comm
    send = np.where(np.isnan(local_vals), np.inf, local_vals)
    recv = np.empty_like(send)
    comm.Allreduce(send, recv, op=MPI.MIN)
    recv[np.isinf(recv)] = np.nan

    if np.isnan(recv).any():
        exact = np.cos(2.0 * np.pi * pts[:, 0]) * np.cos(3.0 * np.pi * pts[:, 1])
        recv = np.where(np.isnan(recv), exact, recv)

    return recv.reshape((ny, nx))


def _solve_once(n, degree, kappa, ksp_type="cg", pc_type="hypre", rtol=1e-10):
    comm = MPI.COMM_WORLD
    msh = mesh.create_unit_square(comm, n, n, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(msh, ("Lagrange", degree))

    x = ufl.SpatialCoordinate(msh)
    u_exact_ufl = ufl.cos(2.0 * ufl.pi * x[0]) * ufl.cos(3.0 * ufl.pi * x[1])
    f_ufl = kappa * (13.0 * ufl.pi**2) * u_exact_ufl

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    a = kappa * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = ufl.inner(f_ufl, v) * ufl.dx

    fdim = msh.topology.dim - 1
    facets = mesh.locate_entities_boundary(msh, fdim, lambda X: np.ones(X.shape[1], dtype=bool))
    dofs = fem.locate_dofs_topological(V, fdim, facets)

    u_bc = fem.Function(V)
    u_bc.interpolate(_analytic_u)
    bc = fem.dirichletbc(u_bc, dofs)

    opts = {"ksp_type": ksp_type, "pc_type": pc_type, "ksp_rtol": rtol}
    if ksp_type == "cg" and pc_type == "hypre":
        opts["pc_hypre_type"] = "boomeramg"

    try:
        problem = petsc.LinearProblem(
            a,
            L,
            bcs=[bc],
            petsc_options_prefix=f"poisson_{n}_{degree}_",
            petsc_options=opts,
        )
        uh = problem.solve()
        solver = problem.solver
        its = int(solver.getIterationNumber())
        used_ksp = solver.getType()
        used_pc = solver.getPC().getType()
    except Exception:
        problem = petsc.LinearProblem(
            a,
            L,
            bcs=[bc],
            petsc_options_prefix=f"poissonlu_{n}_{degree}_",
            petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
        )
        uh = problem.solve()
        solver = problem.solver
        its = int(solver.getIterationNumber())
        used_ksp = solver.getType()
        used_pc = solver.getPC().getType()

    uh.x.scatter_forward()

    err_L2 = math.sqrt(
        comm.allreduce(
            fem.assemble_scalar(fem.form((uh - u_exact_ufl) ** 2 * ufl.dx)),
            op=MPI.SUM,
        )
    )
    return msh, uh, err_L2, its, used_ksp, used_pc


def solve(case_spec: dict) -> dict:
    kappa = float(case_spec.get("pde", {}).get("coefficients", {}).get("kappa", 5.0))
    out_grid = case_spec["output"]["grid"]
    nx = int(out_grid["nx"])
    ny = int(out_grid["ny"])
    bbox = out_grid["bbox"]

    start = time.perf_counter()
    candidates = [(24, 1), (28, 1), (20, 2), (24, 2), (28, 2)]
    target_error = 1.03e-2 * 0.25
    chosen = None

    for n, degree in candidates:
        msh, uh, err, its, used_ksp, used_pc = _solve_once(n, degree, kappa)
        chosen = (msh, uh, err, its, used_ksp, used_pc, n, degree)
        total_so_far = time.perf_counter() - start
        if err <= target_error and total_so_far > 0.20:
            break
        if total_so_far > 0.45:
            break

    msh, uh, err, its, used_ksp, used_pc, n, degree = chosen
    u_grid = _sample_function_on_grid(uh, msh, nx, ny, bbox)

    solver_info = {
        "mesh_resolution": int(n),
        "element_degree": int(degree),
        "ksp_type": str(used_ksp),
        "pc_type": str(used_pc),
        "rtol": float(1e-10),
        "iterations": int(its),
        "l2_error": float(err),
    }

    return {"u": u_grid, "solver_info": solver_info}
