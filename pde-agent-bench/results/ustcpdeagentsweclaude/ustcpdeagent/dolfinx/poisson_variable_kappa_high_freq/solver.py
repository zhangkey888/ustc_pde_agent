import time
import math
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl

ScalarType = PETSc.ScalarType

"""
DIAGNOSIS
equation_type: poisson
spatial_dim: 2
domain_geometry: rectangle
unknowns: scalar
coupling: none
linearity: linear
time_dependence: steady
stiffness: N/A
dominant_physics: diffusion
peclet_or_reynolds: N/A
solution_regularity: smooth
bc_type: all_dirichlet
special_notes: manufactured_solution, variable_coeff
"""

"""
METHOD
spatial_method: fem
element_or_basis: Lagrange_P2
stabilization: none
time_method: none
nonlinear_solver: none
linear_solver: cg
preconditioner: hypre
special_treatment: none
pde_skill: poisson
"""


def _sample_function_on_grid(domain, uh, grid):
    nx = int(grid["nx"])
    ny = int(grid["ny"])
    xmin, xmax, ymin, ymax = grid["bbox"]
    xs = np.linspace(xmin, xmax, nx)
    ys = np.linspace(ymin, ymax, ny)
    XX, YY = np.meshgrid(xs, ys, indexing="xy")
    pts2 = np.column_stack([XX.ravel(), YY.ravel(), np.zeros(nx * ny, dtype=np.float64)])

    tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(tree, pts2)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, pts2)

    values = np.full(nx * ny, np.nan, dtype=np.float64)
    points_local = []
    cells_local = []
    ids_local = []
    for i in range(pts2.shape[0]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_local.append(pts2[i])
            cells_local.append(links[0])
            ids_local.append(i)

    if len(points_local) > 0:
        vals = uh.eval(np.array(points_local, dtype=np.float64), np.array(cells_local, dtype=np.int32))
        vals = np.asarray(vals).reshape(-1)
        values[np.array(ids_local, dtype=np.int32)] = vals

    comm = domain.comm
    gvalues = np.empty_like(values)
    comm.Allreduce(values, gvalues, op=MPI.SUM)
    return gvalues.reshape(ny, nx)


def _build_and_solve(n, degree, ksp_type="cg", pc_type="hypre", rtol=1e-10):
    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_square(comm, n, n, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", degree))

    x = ufl.SpatialCoordinate(domain)
    pi = ufl.pi

    u_exact_ufl = ufl.sin(2 * pi * x[0]) * ufl.sin(2 * pi * x[1])
    kappa = 1.0 + 0.3 * ufl.sin(8 * pi * x[0]) * ufl.sin(8 * pi * x[1])
    f_expr = -ufl.div(kappa * ufl.grad(u_exact_ufl))

    uD = fem.Function(V)
    uD.interpolate(fem.Expression(u_exact_ufl, V.element.interpolation_points))

    fdim = domain.topology.dim - 1
    facets = mesh.locate_entities_boundary(domain, fdim, lambda xx: np.ones(xx.shape[1], dtype=bool))
    dofs = fem.locate_dofs_topological(V, fdim, facets)
    bc = fem.dirichletbc(uD, dofs)

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    a = ufl.inner(kappa * ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = ufl.inner(f_expr, v) * ufl.dx

    opts = {
        "ksp_type": ksp_type,
        "pc_type": pc_type,
        "ksp_rtol": rtol,
    }
    if ksp_type == "cg" and pc_type == "hypre":
        opts["pc_hypre_type"] = "boomeramg"

    start = time.perf_counter()
    problem = petsc.LinearProblem(
        a, L, bcs=[bc],
        petsc_options=opts,
        petsc_options_prefix="poisson_var_",
    )
    uh = problem.solve()
    uh.x.scatter_forward()
    elapsed = time.perf_counter() - start

    Vex = fem.functionspace(domain, ("Lagrange", max(degree + 2, 4)))
    uex = fem.Function(Vex)
    uex.interpolate(fem.Expression(u_exact_ufl, Vex.element.interpolation_points))

    uh_high = fem.Function(Vex)
    uh_high.interpolate(uh)

    err_L2 = math.sqrt(domain.comm.allreduce(
        fem.assemble_scalar(fem.form((uh_high - uex) ** 2 * ufl.dx)),
        op=MPI.SUM
    ))

    ksp = problem.solver
    its = int(ksp.getIterationNumber())
    return domain, uh, err_L2, elapsed, its


def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD
    time_limit = 1.622

    candidates = [
        (36, 2, "cg", "hypre", 1e-10),
        (48, 2, "cg", "hypre", 1e-10),
        (56, 2, "cg", "hypre", 1e-10),
        (40, 3, "cg", "hypre", 1e-10),
    ]

    chosen = None
    history = []
    started = time.perf_counter()

    for cand in candidates:
        n, degree, ksp_type, pc_type, rtol = cand
        try:
            domain, uh, err, elapsed, its = _build_and_solve(n, degree, ksp_type, pc_type, rtol)
            history.append((cand, err, elapsed, its))
            remaining = time_limit - (time.perf_counter() - started)
            chosen = (domain, uh, n, degree, ksp_type, pc_type, rtol, err, elapsed, its)
            if err <= 3.55e-03 and remaining < 0.45:
                break
            if elapsed > 0.9 * time_limit:
                break
        except Exception:
            try:
                domain, uh, err, elapsed, its = _build_and_solve(n, degree, "preonly", "lu", 1e-10)
                history.append(((n, degree, "preonly", "lu", 1e-10), err, elapsed, its))
                chosen = (domain, uh, n, degree, "preonly", "lu", 1e-10, err, elapsed, its)
                break
            except Exception:
                continue

    if chosen is None:
        raise RuntimeError("Failed to solve Poisson problem with all candidate configurations")

    domain, uh, n, degree, ksp_type, pc_type, rtol, err, elapsed, its = chosen

    grid = case_spec["output"]["grid"]
    u_grid = _sample_function_on_grid(domain, uh, grid)

    solver_info = {
        "mesh_resolution": int(n),
        "element_degree": int(degree),
        "ksp_type": str(ksp_type),
        "pc_type": str(pc_type),
        "rtol": float(rtol),
        "iterations": int(its),
        "l2_error": float(err),
        "solve_wall_time": float(elapsed),
    }

    return {"u": u_grid, "solver_info": solver_info}


if __name__ == "__main__":
    case_spec = {
        "output": {
            "grid": {"nx": 64, "ny": 64, "bbox": [0.0, 1.0, 0.0, 1.0]}
        },
        "pde": {"time": None},
    }
    out = solve(case_spec)
    print(out["u"].shape)
    print(out["solver_info"])
