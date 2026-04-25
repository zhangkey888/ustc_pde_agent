import time
import math
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl

ScalarType = PETSc.ScalarType

# DIAGNOSIS
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

# METHOD
# spatial_method: fem
# element_or_basis: Lagrange_P2
# stabilization: none
# time_method: none
# nonlinear_solver: none
# linear_solver: cg
# preconditioner: hypre
# special_treatment: none
# pde_skill: poisson


def _probe_points(u_func, pts):
    domain = u_func.function_space.mesh
    tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(tree, pts)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, pts)

    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    for i in range(pts.shape[0]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(pts[i])
            cells_on_proc.append(links[0])
            eval_map.append(i)

    values = np.full((pts.shape[0],), np.nan, dtype=np.float64)
    if points_on_proc:
        vals = u_func.eval(
            np.array(points_on_proc, dtype=np.float64),
            np.array(cells_on_proc, dtype=np.int32),
        )
        values[np.array(eval_map, dtype=np.int32)] = np.asarray(vals).reshape(-1)
    return values


def _sample_on_uniform_grid(u_func, grid_spec):
    nx = int(grid_spec["nx"])
    ny = int(grid_spec["ny"])
    xmin, xmax, ymin, ymax = map(float, grid_spec["bbox"])

    xs = np.linspace(xmin, xmax, nx, dtype=np.float64)
    ys = np.linspace(ymin, ymax, ny, dtype=np.float64)
    xx, yy = np.meshgrid(xs, ys, indexing="xy")
    pts = np.column_stack(
        [xx.ravel(), yy.ravel(), np.zeros(nx * ny, dtype=np.float64)]
    )

    vals_local = _probe_points(u_func, pts)
    comm = u_func.function_space.mesh.comm
    vals = comm.allreduce(np.nan_to_num(vals_local, nan=0.0), op=MPI.SUM)
    found = comm.allreduce((~np.isnan(vals_local)).astype(np.int32), op=MPI.SUM)
    if np.any(found == 0):
        raise RuntimeError("Point evaluation failed for some output grid points")
    return vals.reshape(ny, nx)


def _manufactured_forms(domain, V):
    x = ufl.SpatialCoordinate(domain)
    u_exact = ufl.sin(ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])
    f = 2.0 * ufl.pi * ufl.pi * u_exact

    u_D = fem.Function(V)
    u_D.interpolate(fem.Expression(u_exact, V.element.interpolation_points))

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    kappa = fem.Constant(domain, ScalarType(1.0))
    a = ufl.inner(kappa * ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = ufl.inner(f, v) * ufl.dx
    return u_exact, u_D, a, L


def _boundary_condition(domain, V, u_D):
    fdim = domain.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
    )
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    return fem.dirichletbc(u_D, boundary_dofs)


def _solve_once(n, degree, ksp_type="cg", pc_type="hypre", rtol=1e-10):
    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_square(
        comm, n, n, cell_type=mesh.CellType.triangle
    )
    V = fem.functionspace(domain, ("Lagrange", degree))
    u_exact, u_D, a, L = _manufactured_forms(domain, V)
    bc = _boundary_condition(domain, V, u_D)

    problem = petsc.LinearProblem(
        a,
        L,
        bcs=[bc],
        petsc_options_prefix="poisson_basic_",
        petsc_options={
            "ksp_type": ksp_type,
            "pc_type": pc_type,
            "ksp_rtol": rtol,
        },
    )
    uh = problem.solve()
    uh.x.scatter_forward()

    e = uh - u_exact
    l2_sq_local = fem.assemble_scalar(fem.form(ufl.inner(e, e) * ufl.dx))
    h1_semi_sq_local = fem.assemble_scalar(
        fem.form(ufl.inner(ufl.grad(e), ufl.grad(e)) * ufl.dx)
    )
    l2_error = math.sqrt(comm.allreduce(l2_sq_local, op=MPI.SUM))
    h1_error = math.sqrt(comm.allreduce(l2_sq_local + h1_semi_sq_local, op=MPI.SUM))

    a_form = fem.form(a)
    L_form = fem.form(L)
    A = petsc.assemble_matrix(a_form, bcs=[bc])
    A.assemble()
    b = petsc.create_vector(L_form.function_spaces)
    with b.localForm() as loc:
        loc.set(0)
    petsc.assemble_vector(b, L_form)
    petsc.apply_lifting(b, [a_form], bcs=[[bc]])
    b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    petsc.set_bc(b, [bc])

    ksp = PETSc.KSP().create(comm)
    ksp.setOperators(A)
    ksp.setType(ksp_type)
    ksp.getPC().setType(pc_type)
    ksp.setTolerances(rtol=rtol)
    ksp.setFromOptions()
    xvec = A.createVecRight()
    xvec.set(0.0)
    ksp.solve(b, xvec)
    iterations = int(ksp.getIterationNumber())

    return {
        "u": uh,
        "l2_error": float(l2_error),
        "h1_error": float(h1_error),
        "iterations": iterations,
    }


def solve(case_spec: dict) -> dict:
    t0 = time.perf_counter()
    time_budget = 7.124
    safety_factor = 0.75

    candidates = [
        {"n": 20, "degree": 1, "ksp_type": "cg", "pc_type": "hypre", "rtol": 1e-10},
        {"n": 32, "degree": 1, "ksp_type": "cg", "pc_type": "hypre", "rtol": 1e-10},
        {"n": 24, "degree": 2, "ksp_type": "cg", "pc_type": "hypre", "rtol": 1e-11},
        {"n": 32, "degree": 2, "ksp_type": "cg", "pc_type": "hypre", "rtol": 1e-11},
        {"n": 40, "degree": 2, "ksp_type": "cg", "pc_type": "hypre", "rtol": 1e-12},
        {"n": 48, "degree": 2, "ksp_type": "cg", "pc_type": "hypre", "rtol": 1e-12},
    ]

    best_cfg = None
    best_res = None
    last_dt = None

    for cfg in candidates:
        if time.perf_counter() - t0 > time_budget * safety_factor:
            break
        start = time.perf_counter()
        try:
            res = _solve_once(**cfg)
        except Exception:
            fallback = dict(cfg)
            fallback["ksp_type"] = "preonly"
            fallback["pc_type"] = "lu"
            res = _solve_once(**fallback)
            cfg = fallback
        elapsed = time.perf_counter() - start
        last_dt = elapsed
        best_cfg = cfg
        best_res = res
        if res["l2_error"] <= 1.92e-3 and (time.perf_counter() - t0 + 2.0 * elapsed > time_budget * safety_factor):
            break

    if best_cfg is None or best_res is None:
        raise RuntimeError("No successful solve was completed")

    u_grid = _sample_on_uniform_grid(best_res["u"], case_spec["output"]["grid"])
    solver_info = {
        "mesh_resolution": int(best_cfg["n"]),
        "element_degree": int(best_cfg["degree"]),
        "ksp_type": str(best_cfg["ksp_type"]),
        "pc_type": str(best_cfg["pc_type"]),
        "rtol": float(best_cfg["rtol"]),
        "iterations": int(best_res["iterations"]),
        "l2_error_vs_exact": float(best_res["l2_error"]),
        "h1_error_vs_exact": float(best_res["h1_error"]),
        "wall_time_sec": float(time.perf_counter() - t0),
    }
    if last_dt is not None:
        solver_info["last_solve_stage_sec"] = float(last_dt)

    return {"u": u_grid, "solver_info": solver_info}


if __name__ == "__main__":
    case_spec = {
        "pde": {"type": "poisson", "time": None},
        "output": {"grid": {"nx": 64, "ny": 64, "bbox": [0.0, 1.0, 0.0, 1.0]}},
    }
    out = solve(case_spec)
    if MPI.COMM_WORLD.rank == 0:
        print(out["u"].shape)
        print(out["solver_info"])
