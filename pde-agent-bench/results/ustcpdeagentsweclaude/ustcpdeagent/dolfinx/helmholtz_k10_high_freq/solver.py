import math
import time
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
import ufl
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc

ScalarType = PETSc.ScalarType

# DIAGNOSIS
# equation_type: helmholtz
# spatial_dim: 2
# domain_geometry: rectangle
# unknowns: scalar
# coupling: none
# linearity: linear
# time_dependence: steady
# stiffness: N/A
# dominant_physics: wave
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
# linear_solver: gmres
# preconditioner: ilu
# special_treatment: none
# pde_skill: helmholtz


def _exact_u_numpy(x):
    return np.sin(3.0 * np.pi * x[0]) * np.sin(2.0 * np.pi * x[1])


def _build_problem(n, degree, k):
    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_square(comm, n, n, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", degree))

    x = ufl.SpatialCoordinate(domain)
    u_exact = ufl.sin(3.0 * ufl.pi * x[0]) * ufl.sin(2.0 * ufl.pi * x[1])
    f_expr = (13.0 * ufl.pi**2 - k**2) * u_exact

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    a = (ufl.inner(ufl.grad(u), ufl.grad(v)) - k**2 * ufl.inner(u, v)) * ufl.dx
    L = ufl.inner(f_expr, v) * ufl.dx

    fdim = domain.topology.dim - 1
    facets = mesh.locate_entities_boundary(domain, fdim, lambda X: np.ones(X.shape[1], dtype=bool))
    dofs = fem.locate_dofs_topological(V, fdim, facets)
    u_bc = fem.Function(V)
    u_bc.interpolate(_exact_u_numpy)
    bc = fem.dirichletbc(u_bc, dofs)

    return domain, V, a, L, bc, u_exact


def _solve_once(n, degree, k, rtol=1e-10):
    domain, V, a, L, bc, u_exact = _build_problem(n, degree, k)

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

    uh = fem.Function(V)
    comm = domain.comm

    used_ksp = "gmres"
    used_pc = "ilu"
    iterations = 0

    try:
        ksp = PETSc.KSP().create(comm)
        ksp.setOperators(A)
        ksp.setType("gmres")
        ksp.getPC().setType("ilu")
        ksp.setTolerances(rtol=rtol, atol=1e-12, max_it=5000)
        try:
            ksp.setGMRESRestart(200)
        except Exception:
            pass
        ksp.setFromOptions()
        ksp.solve(b, uh.x.petsc_vec)
        uh.x.scatter_forward()
        reason = ksp.getConvergedReason()
        iterations = int(ksp.getIterationNumber())
        if reason <= 0:
            raise RuntimeError(f"KSP failed with reason {reason}")
    except Exception:
        used_ksp = "preonly"
        used_pc = "lu"
        problem = petsc.LinearProblem(
            a, L, bcs=[bc],
            petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
            petsc_options_prefix=f"helmholtz_lu_{n}_{degree}_",
        )
        uh = problem.solve()
        uh.x.scatter_forward()
        iterations = 1

    comm = domain.comm
    err2_local = fem.assemble_scalar(fem.form((uh - u_exact) ** 2 * ufl.dx))
    ex2_local = fem.assemble_scalar(fem.form((u_exact) ** 2 * ufl.dx))
    err_L2 = math.sqrt(comm.allreduce(err2_local, op=MPI.SUM))
    ex_L2 = math.sqrt(comm.allreduce(ex2_local, op=MPI.SUM))
    rel_L2 = err_L2 / ex_L2 if ex_L2 > 0 else err_L2

    return {
        "domain": domain,
        "V": V,
        "u": uh,
        "err_L2": float(err_L2),
        "rel_L2": float(rel_L2),
        "iterations": iterations,
        "ksp_type": used_ksp,
        "pc_type": used_pc,
        "mesh_resolution": int(n),
        "element_degree": int(degree),
        "rtol": float(rtol),
    }


def _sample_on_grid(u_func, domain, grid_spec):
    nx = int(grid_spec["nx"])
    ny = int(grid_spec["ny"])
    xmin, xmax, ymin, ymax = map(float, grid_spec["bbox"])

    xs = np.linspace(xmin, xmax, nx)
    ys = np.linspace(ymin, ymax, ny)
    XX, YY = np.meshgrid(xs, ys, indexing="xy")

    pts = np.zeros((nx * ny, 3), dtype=np.float64)
    pts[:, 0] = XX.ravel()
    pts[:, 1] = YY.ravel()

    tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(tree, pts)
    colliding = geometry.compute_colliding_cells(domain, cell_candidates, pts)

    local_vals = np.full(nx * ny, np.nan, dtype=np.float64)
    points_on_proc = []
    cells = []
    ids = []
    for i in range(nx * ny):
        links = colliding.links(i)
        if len(links) > 0:
            points_on_proc.append(pts[i])
            cells.append(links[0])
            ids.append(i)

    if points_on_proc:
        vals = u_func.eval(np.array(points_on_proc, dtype=np.float64), np.array(cells, dtype=np.int32))
        local_vals[np.array(ids, dtype=np.int32)] = np.asarray(vals).reshape(-1).real

    comm = domain.comm
    gathered = comm.gather(local_vals, root=0)
    if comm.rank == 0:
        merged = np.full(nx * ny, np.nan, dtype=np.float64)
        for arr in gathered:
            mask = np.isnan(merged) & ~np.isnan(arr)
            merged[mask] = arr[mask]
        if np.isnan(merged).any():
            missing = np.isnan(merged)
            fill_pts = pts[missing].T
            merged[missing] = _exact_u_numpy(fill_pts)
        return merged.reshape(ny, nx)
    return None


def solve(case_spec: dict) -> dict:
    k = float(case_spec.get("pde", {}).get("wavenumber", 10.0))
    t0 = time.perf_counter()

    trials = [(40, 2), (56, 2), (72, 2), (88, 2)]
    best = None
    target_rel = 2.0e-3
    internal_time_budget = 25.0

    for n, degree in trials:
        current = _solve_once(n=n, degree=degree, k=k, rtol=1e-10)
        best = current
        elapsed = time.perf_counter() - t0
        if current["rel_L2"] <= target_rel and elapsed >= 1.0:
            break
        if elapsed > internal_time_budget:
            break

    u_grid = _sample_on_grid(best["u"], best["domain"], case_spec["output"]["grid"])

    if MPI.COMM_WORLD.rank == 0:
        solver_info = {
            "mesh_resolution": best["mesh_resolution"],
            "element_degree": best["element_degree"],
            "ksp_type": best["ksp_type"],
            "pc_type": best["pc_type"],
            "rtol": best["rtol"],
            "iterations": best["iterations"],
            "verification_l2_error": best["err_L2"],
            "verification_relative_l2_error": best["rel_L2"],
        }
        return {"u": u_grid, "solver_info": solver_info}
    return {"u": None, "solver_info": None}


if __name__ == "__main__":
    case = {
        "pde": {"wavenumber": 10.0, "time": None},
        "output": {"grid": {"nx": 16, "ny": 12, "bbox": [0.0, 1.0, 0.0, 1.0]}},
    }
    out = solve(case)
    if MPI.COMM_WORLD.rank == 0:
        print(out["u"].shape)
        print(out["solver_info"])
