import time
import math
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl

ScalarType = PETSc.ScalarType


def _build_problem(n, degree):
    domain = mesh.create_unit_square(MPI.COMM_WORLD, n, n, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", degree))

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    x = ufl.SpatialCoordinate(domain)

    kappa = 1.0 + 50.0 * ufl.exp(-200.0 * (x[0] - 0.5) ** 2)
    f = 1.0 + ufl.sin(2.0 * ufl.pi * x[0]) * ufl.cos(2.0 * ufl.pi * x[1])

    a = ufl.inner(kappa * ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = ufl.inner(f, v) * ufl.dx

    fdim = domain.topology.dim - 1
    facets = mesh.locate_entities_boundary(domain, fdim, lambda X: np.ones(X.shape[1], dtype=bool))
    dofs = fem.locate_dofs_topological(V, fdim, facets)
    bc = fem.dirichletbc(ScalarType(0.0), dofs, V)
    return domain, V, a, L, [bc]


def _solve_linear(domain, V, a, L, bcs):
    a_form = fem.form(a)
    L_form = fem.form(L)

    A = petsc.assemble_matrix(a_form, bcs=bcs)
    A.assemble()
    b = petsc.create_vector(L_form.function_spaces)
    with b.localForm() as loc:
        loc.set(0.0)
    petsc.assemble_vector(b, L_form)
    petsc.apply_lifting(b, [a_form], bcs=[bcs])
    b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    petsc.set_bc(b, bcs)

    uh = fem.Function(V)
    solver = PETSc.KSP().create(domain.comm)
    solver.setOperators(A)
    solver.setType("cg")
    pc = solver.getPC()
    pc.setType("hypre")
    try:
        pc.setHYPREType("boomeramg")
    except Exception:
        pass
    solver.setTolerances(rtol=1e-10, atol=1e-12, max_it=10000)

    try:
        solver.solve(b, uh.x.petsc_vec)
        if solver.getConvergedReason() <= 0:
            raise RuntimeError("CG failed")
    except Exception:
        solver.destroy()
        solver = PETSc.KSP().create(domain.comm)
        solver.setOperators(A)
        solver.setType("preonly")
        solver.getPC().setType("lu")
        solver.solve(b, uh.x.petsc_vec)

    uh.x.scatter_forward()
    return uh, {
        "iterations": int(solver.getIterationNumber()),
        "ksp_type": str(solver.getType()),
        "pc_type": str(solver.getPC().getType()),
        "rtol": float(1e-10 if str(solver.getType()) != "preonly" else 1e-12),
    }


def _sample_function(u_fun, bbox, nx, ny):
    domain = u_fun.function_space.mesh
    xs = np.linspace(bbox[0], bbox[1], nx)
    ys = np.linspace(bbox[2], bbox[3], ny)
    XX, YY = np.meshgrid(xs, ys, indexing="xy")
    pts = np.column_stack([XX.ravel(), YY.ravel(), np.zeros(nx * ny, dtype=np.float64)])

    tree = geometry.bb_tree(domain, domain.topology.dim)
    candidates = geometry.compute_collisions_points(tree, pts)
    colliding = geometry.compute_colliding_cells(domain, candidates, pts)

    local_vals = np.full(nx * ny, np.nan, dtype=np.float64)
    pts_local = []
    cells_local = []
    ids = []
    for i in range(pts.shape[0]):
        links = colliding.links(i)
        if len(links) > 0:
            pts_local.append(pts[i])
            cells_local.append(links[0])
            ids.append(i)

    if ids:
        vals = np.asarray(
            u_fun.eval(np.array(pts_local, dtype=np.float64), np.array(cells_local, dtype=np.int32))
        ).reshape(-1)
        local_vals[np.array(ids, dtype=np.int32)] = vals.real.astype(np.float64)

    gathered = domain.comm.gather(local_vals, root=0)
    if domain.comm.rank == 0:
        out = np.full(nx * ny, np.nan, dtype=np.float64)
        for arr in gathered:
            mask = np.isnan(out) & ~np.isnan(arr)
            out[mask] = arr[mask]
        out[np.isnan(out)] = 0.0
        return out.reshape((ny, nx))
    return None


def _relative_l2_difference(coarse_n, coarse_degree, fine_n, fine_degree):
    domain_c, Vc, ac, Lc, bcc = _build_problem(coarse_n, coarse_degree)
    uc, _ = _solve_linear(domain_c, Vc, ac, Lc, bcc)

    domain_f, Vf, af, Lf, bcf = _build_problem(fine_n, fine_degree)
    uf, sinfo = _solve_linear(domain_f, Vf, af, Lf, bcf)

    uc_on_f = fem.Function(Vf)
    uc_on_f.interpolate(uc)

    err_sq = domain_f.comm.allreduce(
        fem.assemble_scalar(fem.form((uf - uc_on_f) ** 2 * ufl.dx)), op=MPI.SUM
    )
    ref_sq = domain_f.comm.allreduce(
        fem.assemble_scalar(fem.form(uf ** 2 * ufl.dx)), op=MPI.SUM
    )
    rel = math.sqrt(err_sq / max(ref_sq, 1e-30))
    return rel, uf, sinfo


def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD
    t0 = time.perf_counter()

    grid = case_spec["output"]["grid"]
    nx = int(grid["nx"])
    ny = int(grid["ny"])
    bbox = grid["bbox"]

    candidates = [
        (40, 1, 80, 1),
        (48, 1, 96, 1),
        (40, 2, 80, 2),
        (48, 2, 96, 2),
        (64, 2, 128, 2),
    ]

    best = None
    best_u = None
    best_info = None
    trials = []
    time_budget = 4.2

    for coarse_n, coarse_deg, fine_n, fine_deg in candidates:
        if time.perf_counter() - t0 > time_budget:
            break
        try:
            rel, uf, sinfo = _relative_l2_difference(coarse_n, coarse_deg, fine_n, fine_deg)
            trials.append(
                {
                    "coarse_n": int(coarse_n),
                    "coarse_degree": int(coarse_deg),
                    "fine_n": int(fine_n),
                    "fine_degree": int(fine_deg),
                    "relative_l2_difference": float(rel),
                }
            )
            best = (fine_n, fine_deg, rel)
            best_u = uf
            best_info = sinfo
            if rel < 2e-3 and (time.perf_counter() - t0) > 0.6 * time_budget:
                break
        except Exception:
            continue

    if best_u is None:
        n, deg = 64, 1
        domain, V, a, L, bcs = _build_problem(n, deg)
        best_u, best_info = _solve_linear(domain, V, a, L, bcs)
        best = (n, deg, None)

    u_grid = _sample_function(best_u, bbox, nx, ny)

    if comm.rank != 0:
        return {"u": None, "solver_info": {}}

    mesh_resolution, element_degree, _ = best
    return {
        "u": u_grid,
        "solver_info": {
            "mesh_resolution": int(mesh_resolution),
            "element_degree": int(element_degree),
            "ksp_type": best_info["ksp_type"],
            "pc_type": best_info["pc_type"],
            "rtol": float(best_info["rtol"]),
            "iterations": int(best_info["iterations"]),
            "verification": {
                "type": "mesh_convergence",
                "trials": trials,
            },
        },
    }


if __name__ == "__main__":
    case_spec = {
        "output": {"grid": {"nx": 32, "ny": 32, "bbox": [0.0, 1.0, 0.0, 1.0]}}
    }
    result = solve(case_spec)
    if MPI.COMM_WORLD.rank == 0:
        print(result["u"].shape)
        print(result["solver_info"])
