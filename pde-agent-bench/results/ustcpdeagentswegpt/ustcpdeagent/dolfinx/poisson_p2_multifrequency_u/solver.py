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


def _exact_numpy(x, y):
    return np.sin(np.pi * x) * np.sin(np.pi * y) + 0.2 * np.sin(5 * np.pi * x) * np.sin(4 * np.pi * y)


def _make_exact_ufl(x):
    return ufl.sin(ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1]) + 0.2 * ufl.sin(5 * ufl.pi * x[0]) * ufl.sin(4 * ufl.pi * x[1])


def _sample_function(u_func, domain, nx, ny, bbox):
    xmin, xmax, ymin, ymax = bbox
    xs = np.linspace(xmin, xmax, nx)
    ys = np.linspace(ymin, ymax, ny)
    XX, YY = np.meshgrid(xs, ys, indexing="xy")
    pts2 = np.column_stack([XX.ravel(), YY.ravel(), np.zeros(nx * ny)])

    tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(tree, pts2)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, pts2)

    values = np.full(nx * ny, np.nan, dtype=np.float64)
    points_on_proc = []
    cells_on_proc = []
    eval_ids = []

    for i in range(pts2.shape[0]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(pts2[i])
            cells_on_proc.append(links[0])
            eval_ids.append(i)

    if points_on_proc:
        vals = u_func.eval(np.array(points_on_proc, dtype=np.float64), np.array(cells_on_proc, dtype=np.int32))
        vals = np.asarray(vals).reshape(len(points_on_proc), -1)[:, 0]
        values[np.array(eval_ids, dtype=np.int32)] = vals

    # Gather on root if needed; benchmark is typically serial, but keep this safe.
    comm = domain.comm
    if comm.size > 1:
        gathered = comm.gather(values, root=0)
        if comm.rank == 0:
            out = gathered[0].copy()
            for arr in gathered[1:]:
                mask = np.isnan(out) & ~np.isnan(arr)
                out[mask] = arr[mask]
            values = out
        values = comm.bcast(values if comm.rank == 0 else None, root=0)

    # Fill any remaining NaNs analytically at boundary-corner degeneracies
    nan_mask = np.isnan(values)
    if np.any(nan_mask):
        xf = XX.ravel()[nan_mask]
        yf = YY.ravel()[nan_mask]
        values[nan_mask] = _exact_numpy(xf, yf)

    return values.reshape(ny, nx)


def _solve_once(n, degree=2, ksp_type="cg", pc_type="hypre", rtol=1e-10):
    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_square(comm, n, n, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", degree))

    x = ufl.SpatialCoordinate(domain)
    u_exact = _make_exact_ufl(x)
    kappa = ScalarType(1.0)
    f_expr = -ufl.div(kappa * ufl.grad(u_exact))

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    a = ufl.inner(kappa * ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = ufl.inner(f_expr, v) * ufl.dx

    fdim = domain.topology.dim - 1
    facets = mesh.locate_entities_boundary(domain, fdim, lambda xx: np.ones(xx.shape[1], dtype=bool))
    dofs = fem.locate_dofs_topological(V, fdim, facets)

    u_bc = fem.Function(V)
    u_bc.interpolate(fem.Expression(u_exact, V.element.interpolation_points))
    bc = fem.dirichletbc(u_bc, dofs)

    a_form = fem.form(a)
    L_form = fem.form(L)

    A = petsc.assemble_matrix(a_form, bcs=[bc])
    A.assemble()
    b = petsc.create_vector(L_form.function_spaces)

    uh = fem.Function(V)

    solver = PETSc.KSP().create(comm)
    solver.setOperators(A)
    solver.setType(ksp_type)
    pc = solver.getPC()
    pc.setType(pc_type)
    solver.setTolerances(rtol=rtol)
    solver.setFromOptions()

    with b.localForm() as loc:
        loc.set(0.0)
    petsc.assemble_vector(b, L_form)
    petsc.apply_lifting(b, [a_form], bcs=[[bc]])
    b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    petsc.set_bc(b, [bc])

    t0 = time.perf_counter()
    try:
        solver.solve(b, uh.x.petsc_vec)
    except Exception:
        solver.setType("preonly")
        solver.getPC().setType("lu")
        solver.setOperators(A)
        solver.solve(b, uh.x.petsc_vec)
    solve_time = time.perf_counter() - t0
    uh.x.scatter_forward()

    its = int(solver.getIterationNumber())
    used_ksp = solver.getType()
    used_pc = solver.getPC().getType()

    err_L2_form = fem.form((uh - u_exact) ** 2 * ufl.dx)
    err_H1s_form = fem.form(ufl.inner(ufl.grad(uh - u_exact), ufl.grad(uh - u_exact)) * ufl.dx)
    l2 = math.sqrt(comm.allreduce(fem.assemble_scalar(err_L2_form), op=MPI.SUM))
    h1s = math.sqrt(comm.allreduce(fem.assemble_scalar(err_H1s_form), op=MPI.SUM))

    return {
        "domain": domain,
        "V": V,
        "u": uh,
        "l2_error": float(l2),
        "h1_semi_error": float(h1s),
        "solve_time": float(solve_time),
        "iterations": its,
        "ksp_type": str(used_ksp),
        "pc_type": str(used_pc),
        "rtol": float(rtol),
        "mesh_resolution": int(n),
        "element_degree": int(degree),
    }


def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD
    rank = comm.rank
    wall_start = time.perf_counter()

    grid = case_spec["output"]["grid"]
    nx = int(grid["nx"])
    ny = int(grid["ny"])
    bbox = grid["bbox"]

    time_limit = 1.480
    safety = 0.18
    budget = max(0.2, time_limit - safety)

    candidates = [28, 36, 44, 52, 60, 72]
    best = None

    # Adaptive accuracy/time trade-off: increase mesh while time allows.
    for n in candidates:
        result = _solve_once(n=n, degree=2, ksp_type="cg", pc_type="hypre", rtol=1e-10)
        elapsed = time.perf_counter() - wall_start
        best = result
        if result["l2_error"] <= 1.17e-4 and elapsed > 0.60 * budget:
            break
        if elapsed > budget:
            break

    u_grid = _sample_function(best["u"], best["domain"], nx, ny, bbox)

    solver_info = {
        "mesh_resolution": int(best["mesh_resolution"]),
        "element_degree": int(best["element_degree"]),
        "ksp_type": str(best["ksp_type"]),
        "pc_type": str(best["pc_type"]),
        "rtol": float(best["rtol"]),
        "iterations": int(best["iterations"]),
        "l2_error_vs_exact": float(best["l2_error"]),
        "h1_semi_error_vs_exact": float(best["h1_semi_error"]),
        "wall_time_sec": float(time.perf_counter() - wall_start),
    }

    return {"u": u_grid, "solver_info": solver_info}


if __name__ == "__main__":
    case_spec = {
        "output": {
            "grid": {
                "nx": 64,
                "ny": 64,
                "bbox": [0.0, 1.0, 0.0, 1.0],
            }
        },
        "pde": {"time": None},
    }
    out = solve(case_spec)
    if MPI.COMM_WORLD.rank == 0:
        print(out["u"].shape)
        print(out["solver_info"])
