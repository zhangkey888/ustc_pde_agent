import time
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
    xmin, xmax, ymin, ymax = map(float, grid_spec["bbox"])

    xs = np.linspace(xmin, xmax, nx)
    ys = np.linspace(ymin, ymax, ny)
    xx, yy = np.meshgrid(xs, ys, indexing="xy")
    pts = np.zeros((nx * ny, 3), dtype=np.float64)
    pts[:, 0] = xx.ravel()
    pts[:, 1] = yy.ravel()

    tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(tree, pts)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, pts)

    local_vals = np.zeros(nx * ny, dtype=np.float64)
    local_found = np.zeros(nx * ny, dtype=np.int32)

    points_on_proc = []
    cells_on_proc = []
    idx = []
    for i in range(nx * ny):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(pts[i])
            cells_on_proc.append(links[0])
            idx.append(i)

    if points_on_proc:
        vals = uh.eval(np.array(points_on_proc, dtype=np.float64), np.array(cells_on_proc, dtype=np.int32))
        vals = np.asarray(vals).reshape(len(points_on_proc), -1)[:, 0]
        idx = np.array(idx, dtype=np.int32)
        local_vals[idx] = vals
        local_found[idx] = 1

    global_vals = np.zeros_like(local_vals)
    global_found = np.zeros_like(local_found)
    domain.comm.Allreduce(local_vals, global_vals, op=MPI.SUM)
    domain.comm.Allreduce(local_found, global_found, op=MPI.SUM)

    if np.any(global_found == 0):
        missing = int(np.count_nonzero(global_found == 0))
        raise RuntimeError(f"Failed to evaluate {missing} sampling points on the mesh.")

    return global_vals.reshape(ny, nx)


def _make_problem(n, degree, beta_vec, eps_value):
    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_square(comm, n, n, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", degree))

    fdim = domain.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool))
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc = fem.dirichletbc(ScalarType(0.0), boundary_dofs, V)

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    x = ufl.SpatialCoordinate(domain)

    eps_c = fem.Constant(domain, ScalarType(eps_value))
    beta = fem.Constant(domain, np.array(beta_vec, dtype=np.float64))
    f_expr = ufl.sin(10.0 * ufl.pi * x[0]) * ufl.sin(8.0 * ufl.pi * x[1])

    h = ufl.CellDiameter(domain)
    bnorm = ufl.sqrt(ufl.dot(beta, beta) + ScalarType(1.0e-14))
    pe = bnorm * h / (2.0 * eps_c + ScalarType(1.0e-14))
    tau = h / (2.0 * bnorm) * (ufl.cosh(pe) / ufl.sinh(pe) - 1.0 / pe)

    Lu = -eps_c * ufl.div(ufl.grad(u)) + ufl.dot(beta, ufl.grad(u))
    stream_v = ufl.dot(beta, ufl.grad(v))

    a = (
        eps_c * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
        + ufl.dot(beta, ufl.grad(u)) * v * ufl.dx
        + tau * Lu * stream_v * ufl.dx
    )
    L = f_expr * v * ufl.dx + tau * f_expr * stream_v * ufl.dx

    return domain, V, bc, a, L, f_expr, beta, eps_c


def _solve_once(n, degree, beta_vec, eps_value, ksp_type="gmres", pc_type="ilu", rtol=1e-9):
    domain, V, bc, a, L, f_expr, beta, eps_c = _make_problem(n, degree, beta_vec, eps_value)

    a_form = fem.form(a)
    L_form = fem.form(L)
    A = petsc.assemble_matrix(a_form, bcs=[bc])
    A.assemble()
    b = petsc.create_vector(L_form.function_spaces)

    solver = PETSc.KSP().create(domain.comm)
    solver.setOperators(A)
    solver.setType(ksp_type)
    solver.getPC().setType(pc_type)
    solver.setTolerances(rtol=rtol, atol=1e-12, max_it=5000)

    uh = fem.Function(V)

    with b.localForm() as loc:
        loc.set(0.0)
    petsc.assemble_vector(b, L_form)
    petsc.apply_lifting(b, [a_form], bcs=[[bc]])
    b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    petsc.set_bc(b, [bc])

    try:
        solver.solve(b, uh.x.petsc_vec)
        uh.x.scatter_forward()
        reason = solver.getConvergedReason()
        if reason <= 0:
            raise RuntimeError(f"Iterative solver failed with reason {reason}")
        ksp_used = solver.getType()
        pc_used = solver.getPC().getType()
        its = int(solver.getIterationNumber())
    except Exception:
        solver = PETSc.KSP().create(domain.comm)
        solver.setOperators(A)
        solver.setType("preonly")
        solver.getPC().setType("lu")
        solver.setTolerances(rtol=1e-12, atol=1e-14, max_it=1)
        solver.solve(b, uh.x.petsc_vec)
        uh.x.scatter_forward()
        ksp_used = "preonly"
        pc_used = "lu"
        its = int(solver.getIterationNumber())

    residual_form = fem.form(
        (-eps_c * ufl.div(ufl.grad(uh)) + ufl.dot(beta, ufl.grad(uh)) - f_expr) ** 2 * ufl.dx
    )
    residual_sq = fem.assemble_scalar(residual_form)
    residual_sq = domain.comm.allreduce(residual_sq, op=MPI.SUM)
    residual_norm = float(np.sqrt(max(residual_sq, 0.0)))

    return {
        "domain": domain,
        "V": V,
        "uh": uh,
        "iterations": its,
        "ksp_type": str(ksp_used),
        "pc_type": str(pc_used),
        "rtol": float(rtol),
        "residual_norm": residual_norm,
    }


def solve(case_spec: dict) -> dict:
    start = time.perf_counter()

    pde = case_spec.get("pde", {})
    beta_vec = np.array(pde.get("beta", [15.0, 7.0]), dtype=np.float64)
    eps_value = float(pde.get("epsilon", 0.005))
    grid_spec = case_spec["output"]["grid"]

    degree = 2
    mesh_candidates = [48, 64, 80, 96]
    time_budget = 157.180
    reserve = 20.0

    prev_grid = None
    prev_n = None
    best_pack = None
    verification = {}

    for n in mesh_candidates:
        pack = _solve_once(n, degree, beta_vec, eps_value, ksp_type="gmres", pc_type="ilu", rtol=1e-9)
        grid = _sample_on_grid(pack["domain"], pack["uh"], grid_spec)

        rel_change = None
        if prev_grid is not None:
            rel_change = float(np.linalg.norm((grid - prev_grid).ravel()) / (np.linalg.norm(grid.ravel()) + 1.0e-14))

        elapsed = time.perf_counter() - start
        verification = {
            "residual_norm": float(pack["residual_norm"]),
            "grid_relative_change": rel_change,
            "mesh_convergence_pair": (prev_n, n) if prev_n is not None else None,
            "wall_time_so_far": float(elapsed),
        }

        best_pack = (n, pack, grid)
        prev_grid = grid
        prev_n = n

        if rel_change is not None and rel_change < 3.0e-3 and n >= 64:
            break
        if elapsed > time_budget - reserve:
            break

    n, pack, grid = best_pack

    return {
        "u": grid,
        "solver_info": {
            "mesh_resolution": int(n),
            "element_degree": int(degree),
            "ksp_type": str(pack["ksp_type"]),
            "pc_type": str(pack["pc_type"]),
            "rtol": float(pack["rtol"]),
            "iterations": int(pack["iterations"]),
            "verification": verification,
        },
    }


if __name__ == "__main__":
    case_spec = {
        "pde": {"epsilon": 0.005, "beta": [15.0, 7.0], "time": None},
        "output": {"grid": {"nx": 32, "ny": 24, "bbox": [0.0, 1.0, 0.0, 1.0]}},
    }
    out = solve(case_spec)
    if MPI.COMM_WORLD.rank == 0:
        print(out["u"].shape)
        print(np.min(out["u"]), np.max(out["u"]))
        print(out["solver_info"])
