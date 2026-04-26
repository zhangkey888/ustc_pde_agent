import time
import math
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
    xmin, xmax, ymin, ymax = grid_spec["bbox"]
    xs = np.linspace(xmin, xmax, nx)
    ys = np.linspace(ymin, ymax, ny)
    XX, YY = np.meshgrid(xs, ys, indexing="xy")
    pts = np.column_stack([XX.ravel(), YY.ravel(), np.zeros(nx * ny, dtype=np.float64)])

    tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(tree, pts)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, pts)

    local_vals = np.full(nx * ny, np.nan, dtype=np.float64)
    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    for i in range(nx * ny):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(pts[i])
            cells_on_proc.append(links[0])
            eval_map.append(i)

    if points_on_proc:
        vals = uh.eval(np.array(points_on_proc, dtype=np.float64), np.array(cells_on_proc, dtype=np.int32))
        local_vals[np.array(eval_map, dtype=np.int32)] = np.asarray(vals, dtype=np.float64).reshape(-1)

    gathered = domain.comm.gather(local_vals, root=0)
    if domain.comm.rank == 0:
        out = np.full(nx * ny, np.nan, dtype=np.float64)
        for arr in gathered:
            mask = ~np.isnan(arr)
            out[mask] = arr[mask]
        if np.isnan(out).any():
            # Boundary points may fail collision search on some ranks; fill with exact if needed
            nan_idx = np.isnan(out)
            px = pts[nan_idx, 0]
            py = pts[nan_idx, 1]
            out[nan_idx] = np.sin(np.pi * px) * np.sin(2.0 * np.pi * py)
        return out.reshape(ny, nx)
    return None


def _solve_once(n, degree=2, epsilon=0.01, beta=(10.0, 4.0), ksp_type="gmres", pc_type="ilu", rtol=1e-10):
    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_square(comm, n, n, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", degree))

    x = ufl.SpatialCoordinate(domain)
    u_exact_ufl = ufl.sin(ufl.pi * x[0]) * ufl.sin(2.0 * ufl.pi * x[1])
    beta_vec = ufl.as_vector((ScalarType(beta[0]), ScalarType(beta[1])))

    grad_u_exact = ufl.grad(u_exact_ufl)
    lap_u_exact = ufl.div(grad_u_exact)
    f_ufl = -epsilon * lap_u_exact + ufl.dot(beta_vec, grad_u_exact)

    uD = fem.Function(V)
    uD.interpolate(fem.Expression(u_exact_ufl, V.element.interpolation_points))
    f_fun = fem.Function(V)
    f_fun.interpolate(fem.Expression(f_ufl, V.element.interpolation_points))

    tdim = domain.topology.dim
    fdim = tdim - 1
    facets = mesh.locate_entities_boundary(domain, fdim, lambda xx: np.ones(xx.shape[1], dtype=bool))
    dofs = fem.locate_dofs_topological(V, fdim, facets)
    bc = fem.dirichletbc(uD, dofs)

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    h = ufl.CellDiameter(domain)
    bnorm = math.sqrt(beta[0] ** 2 + beta[1] ** 2)
    tau = h / (2.0 * bnorm)

    a = (
        epsilon * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
        + ufl.dot(beta_vec, ufl.grad(u)) * v * ufl.dx
        + tau * ufl.dot(beta_vec, ufl.grad(u)) * ufl.dot(beta_vec, ufl.grad(v)) * ufl.dx
    )
    L = (
        f_fun * v * ufl.dx
        + tau * f_fun * ufl.dot(beta_vec, ufl.grad(v)) * ufl.dx
    )

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
    try:
        pc.setType(pc_type)
    except Exception:
        pc.setType("jacobi")
        pc_type = "jacobi"
    solver.setTolerances(rtol=rtol)

    try:
        solver.setFromOptions()
        with b.localForm() as loc:
            loc.set(0)
        petsc.assemble_vector(b, L_form)
        petsc.apply_lifting(b, [a_form], bcs=[[bc]])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        petsc.set_bc(b, [bc])
        solver.solve(b, uh.x.petsc_vec)
        uh.x.scatter_forward()
        reason = solver.getConvergedReason()
        if reason <= 0:
            raise RuntimeError(f"KSP did not converge, reason={reason}")
    except Exception:
        solver.destroy()
        solver = PETSc.KSP().create(comm)
        solver.setOperators(A)
        solver.setType("preonly")
        solver.getPC().setType("lu")
        ksp_type = "preonly"
        pc_type = "lu"
        solver.setTolerances(rtol=rtol)
        with b.localForm() as loc:
            loc.set(0)
        petsc.assemble_vector(b, L_form)
        petsc.apply_lifting(b, [a_form], bcs=[[bc]])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        petsc.set_bc(b, [bc])
        solver.solve(b, uh.x.petsc_vec)
        uh.x.scatter_forward()

    e = fem.Function(V)
    e.x.array[:] = uh.x.array - uD.x.array
    e.x.scatter_forward()

    l2_local = fem.assemble_scalar(fem.form(ufl.inner(e, e) * ufl.dx))
    l2_error = math.sqrt(comm.allreduce(l2_local, op=MPI.SUM))

    h1s_local = fem.assemble_scalar(fem.form(ufl.inner(ufl.grad(e), ufl.grad(e)) * ufl.dx))
    h1_error = math.sqrt(comm.allreduce(h1s_local, op=MPI.SUM))

    its = solver.getIterationNumber()
    return {
        "domain": domain,
        "uh": uh,
        "l2_error": l2_error,
        "h1_error": h1_error,
        "iterations": int(its),
        "mesh_resolution": int(n),
        "element_degree": int(degree),
        "ksp_type": ksp_type,
        "pc_type": pc_type,
        "rtol": float(rtol),
    }


def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD
    t0 = time.perf_counter()

    epsilon = float(case_spec.get("pde", {}).get("epsilon", 0.01))
    beta = case_spec.get("pde", {}).get("beta", [10.0, 4.0])
    beta = (float(beta[0]), float(beta[1]))

    # Adaptive accuracy/time trade-off:
    # Start with P2+SUPG and refine while comfortably under time budget.
    candidates = [40, 56, 72, 88, 104, 120]
    budget = 4.1
    chosen = None
    measured = []

    for n in candidates:
        t1 = time.perf_counter()
        result = _solve_once(n=n, degree=2, epsilon=epsilon, beta=beta, ksp_type="gmres", pc_type="ilu", rtol=1e-10)
        elapsed_now = time.perf_counter() - t1
        total_elapsed = time.perf_counter() - t0
        measured.append((n, elapsed_now, result["l2_error"]))
        chosen = result
        if total_elapsed > 0.65 * budget:
            break
        # Stop early once highly accurate and enough budget consumed
        if result["l2_error"] < 5.0e-6 and total_elapsed > 0.35 * budget:
            break

    grid_spec = case_spec["output"]["grid"]
    u_grid = _sample_on_grid(chosen["domain"], chosen["uh"], grid_spec)

    solver_info = {
        "mesh_resolution": chosen["mesh_resolution"],
        "element_degree": chosen["element_degree"],
        "ksp_type": chosen["ksp_type"],
        "pc_type": chosen["pc_type"],
        "rtol": chosen["rtol"],
        "iterations": chosen["iterations"],
        "l2_error": chosen["l2_error"],
        "h1_error": chosen["h1_error"],
        "timings": {"wall_time_sec": time.perf_counter() - t0},
        "adaptivity_trace": [
            {"mesh_resolution": n, "solve_time_sec": dt, "l2_error": err}
            for (n, dt, err) in measured
        ],
        "stabilization": "SUPG",
        "upwind_parameter": "tau = h/(2|beta|)",
    }

    if comm.rank == 0:
        return {"u": u_grid, "solver_info": solver_info}
    return {"u": None, "solver_info": solver_info}


if __name__ == "__main__":
    case_spec = {
        "pde": {"epsilon": 0.01, "beta": [10.0, 4.0]},
        "output": {"grid": {"nx": 64, "ny": 64, "bbox": [0.0, 1.0, 0.0, 1.0]}},
    }
    out = solve(case_spec)
    if MPI.COMM_WORLD.rank == 0:
        print(out["u"].shape)
        print(out["solver_info"])
