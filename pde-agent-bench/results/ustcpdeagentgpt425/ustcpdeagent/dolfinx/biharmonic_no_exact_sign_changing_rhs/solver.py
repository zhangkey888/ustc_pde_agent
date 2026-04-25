import time
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc

from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl


ScalarType = PETSc.ScalarType


def _rhs_expr(x):
    return np.cos(4.0 * np.pi * x[0]) * np.sin(3.0 * np.pi * x[1])


def _boundary_all(x):
    return np.isclose(x[0], 0.0) | np.isclose(x[0], 1.0) | np.isclose(x[1], 0.0) | np.isclose(x[1], 1.0)


def _sample_function_on_grid(u_func, domain, grid_spec):
    nx = int(grid_spec["nx"])
    ny = int(grid_spec["ny"])
    bbox = grid_spec["bbox"]
    xs = np.linspace(float(bbox[0]), float(bbox[1]), nx)
    ys = np.linspace(float(bbox[2]), float(bbox[3]), ny)
    XX, YY = np.meshgrid(xs, ys, indexing="xy")
    pts = np.column_stack([XX.ravel(), YY.ravel(), np.zeros(nx * ny, dtype=np.float64)])

    tree = geometry.bb_tree(domain, domain.topology.dim)
    candidates = geometry.compute_collisions_points(tree, pts)
    colliding = geometry.compute_colliding_cells(domain, candidates, pts)

    local_point_ids = []
    local_points = []
    local_cells = []
    for i in range(pts.shape[0]):
        links = colliding.links(i)
        if len(links) > 0:
            local_point_ids.append(i)
            local_points.append(pts[i])
            local_cells.append(links[0])

    local_vals = np.full(pts.shape[0], np.nan, dtype=np.float64)
    if local_points:
        values = u_func.eval(np.array(local_points, dtype=np.float64), np.array(local_cells, dtype=np.int32))
        values = np.asarray(values).reshape(len(local_points), -1)[:, 0]
        local_vals[np.array(local_point_ids, dtype=np.int32)] = values

    comm = domain.comm
    gathered = comm.gather(local_vals, root=0)
    if comm.rank == 0:
        final = np.full(pts.shape[0], np.nan, dtype=np.float64)
        for arr in gathered:
            mask = np.isnan(final) & ~np.isnan(arr)
            final[mask] = arr[mask]
        final = np.nan_to_num(final, nan=0.0)
        return final.reshape(ny, nx)
    return None


def _solve_poisson(domain, V, rhs_function, bc, ksp_type, pc_type, rtol):
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    a = ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = ufl.inner(rhs_function, v) * ufl.dx

    a_form = fem.form(a)
    L_form = fem.form(L)

    A = petsc.assemble_matrix(a_form, bcs=[bc])
    A.assemble()
    b = petsc.create_vector(L_form.function_spaces)

    ksp = PETSc.KSP().create(domain.comm)
    ksp.setOperators(A)
    ksp.setType(ksp_type)
    pc = ksp.getPC()
    pc.setType(pc_type)
    if pc_type == "hypre":
        try:
            pc.setHYPREType("boomeramg")
        except Exception:
            pass
    ksp.setTolerances(rtol=rtol)

    uh = fem.Function(V)
    with b.localForm() as loc:
        loc.set(0.0)
    petsc.assemble_vector(b, L_form)
    petsc.apply_lifting(b, [a_form], bcs=[[bc]])
    b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    petsc.set_bc(b, [bc])
    ksp.solve(b, uh.x.petsc_vec)
    uh.x.scatter_forward()
    return uh, int(ksp.getIterationNumber())


def _solve_once(n, degree=2, ksp_type="cg", pc_type="hypre", rtol=1e-10):
    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_square(comm, n, n, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", degree))

    f = fem.Function(V)
    f.interpolate(_rhs_expr)

    fdim = domain.topology.dim - 1
    facets = mesh.locate_entities_boundary(domain, fdim, _boundary_all)
    dofs = fem.locate_dofs_topological(V, fdim, facets)
    zero = fem.Function(V)
    zero.x.array[:] = 0.0
    bc = fem.dirichletbc(zero, dofs)

    w, its1 = _solve_poisson(domain, V, f, bc, ksp_type, pc_type, rtol)
    u, its2 = _solve_poisson(domain, V, w, bc, ksp_type, pc_type, rtol)

    solver_info = {
        "mesh_resolution": int(n),
        "element_degree": int(degree),
        "ksp_type": str(ksp_type),
        "pc_type": str(pc_type),
        "rtol": float(rtol),
        "iterations": int(its1 + its2),
    }
    return domain, u, solver_info


def _l2_difference(u1, u2):
    V = u1.function_space
    diff = fem.Function(V)
    diff.x.array[:] = u1.x.array - u2.x.array
    val_local = fem.assemble_scalar(fem.form(ufl.inner(diff, diff) * ufl.dx))
    val = V.mesh.comm.allreduce(val_local, op=MPI.SUM)
    return float(np.sqrt(max(val, 0.0)))


def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD
    t0 = time.perf_counter()

    grid_spec = case_spec["output"]["grid"]
    time_limit = float(case_spec.get("time_limit", 13.597))
    budget = min(13.0, max(4.0, 0.9 * time_limit))

    degree = 2
    ksp_type = "cg"
    pc_type = "hypre"
    rtol = 1e-10

    candidate_ns = [48, 64, 80, 96, 112, 128]
    accepted = []

    prev_grid = None
    prev_info = None
    prev_n = None
    prev_elapsed = 0.0

    for n in candidate_ns:
        domain, uh, info = _solve_once(n=n, degree=degree, ksp_type=ksp_type, pc_type=pc_type, rtol=rtol)
        u_grid = _sample_function_on_grid(uh, domain, grid_spec) if comm.rank == 0 else None
        elapsed = time.perf_counter() - t0

        verify = {"mesh_convergence_l2_grid_diff_to_previous": None}
        if comm.rank == 0 and prev_grid is not None:
            verify["mesh_convergence_l2_grid_diff_to_previous"] = float(
                np.sqrt(np.mean((u_grid - prev_grid) ** 2))
            )

        accepted.append((n, uh, info, verify, u_grid, elapsed))
        prev_grid = u_grid
        prev_info = info
        prev_n = n
        prev_elapsed = elapsed

        if elapsed >= budget:
            break

    n, uh, info, verify, u_grid, elapsed = accepted[-1]
    if len(accepted) >= 2 and elapsed > budget:
        n, uh, info, verify, u_grid, elapsed = accepted[-2]

    if comm.rank == 0:
        info["accuracy_verification"] = verify
        return {"u": np.asarray(u_grid, dtype=np.float64), "solver_info": info}
    return {"u": None, "solver_info": info}


if __name__ == "__main__":
    case_spec = {
        "pde": {"time": None},
        "output": {"grid": {"nx": 64, "ny": 64, "bbox": [0.0, 1.0, 0.0, 1.0]}},
        "time_limit": 13.597,
    }
    out = solve(case_spec)
    if MPI.COMM_WORLD.rank == 0:
        print(out["u"].shape)
        print(out["solver_info"])
