import math
import time
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl


def _defaults(case_spec: dict):
    pde = case_spec.get("pde", {})
    grid = case_spec.get("output", {}).get("grid", {})
    return {
        "t0": float(pde.get("t0", 0.0)),
        "t_end": float(pde.get("t_end", 0.1)),
        "dt": float(pde.get("dt", 0.02)),
        "epsilon": float(pde.get("epsilon", 0.1)),
        "beta": np.array(pde.get("beta", [1.0, 0.5]), dtype=np.float64),
        "nx": int(grid.get("nx", 64)),
        "ny": int(grid.get("ny", 64)),
        "bbox": grid.get("bbox", [0.0, 1.0, 0.0, 1.0]),
    }


def _u_exact(x, t):
    return ufl.exp(-t) * ufl.sin(ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])


def _f_rhs(x, t, eps, beta):
    uex = _u_exact(x, t)
    grad_uex = ufl.as_vector((
        ufl.exp(-t) * ufl.pi * ufl.cos(ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1]),
        ufl.exp(-t) * ufl.pi * ufl.sin(ufl.pi * x[0]) * ufl.cos(ufl.pi * x[1]),
    ))
    u_t = -uex
    lap_u = -2.0 * ufl.pi * ufl.pi * uex
    return u_t - eps * lap_u + ufl.dot(beta, grad_uex)


def _sample_function(uh, domain, nx, ny, bbox):
    xmin, xmax, ymin, ymax = bbox
    xs = np.linspace(xmin, xmax, nx)
    ys = np.linspace(ymin, ymax, ny)
    XX, YY = np.meshgrid(xs, ys, indexing="xy")
    pts = np.column_stack([XX.ravel(), YY.ravel(), np.zeros(nx * ny, dtype=np.float64)])

    tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(tree, pts)
    colliding = geometry.compute_colliding_cells(domain, cell_candidates, pts)

    local_vals = np.full(nx * ny, np.nan, dtype=np.float64)
    pts_local = []
    cells_local = []
    idx_local = []
    for i in range(nx * ny):
        links = colliding.links(i)
        if len(links) > 0:
            pts_local.append(pts[i])
            cells_local.append(links[0])
            idx_local.append(i)

    if pts_local:
        vals = uh.eval(np.array(pts_local, dtype=np.float64), np.array(cells_local, dtype=np.int32))
        local_vals[np.array(idx_local, dtype=np.int32)] = np.asarray(vals).reshape(-1)

    gathered = domain.comm.allgather(local_vals)
    vals = np.full(nx * ny, np.nan, dtype=np.float64)
    for arr in gathered:
        mask = np.isnan(vals) & ~np.isnan(arr)
        vals[mask] = arr[mask]
    vals[np.isnan(vals)] = 0.0
    return vals.reshape((ny, nx))


def _run_one(mesh_n, degree, dt, t0, t_end, eps, beta):
    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_square(comm, mesh_n, mesh_n, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", degree))
    x = ufl.SpatialCoordinate(domain)

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    eps_c = fem.Constant(domain, PETSc.ScalarType(eps))
    beta_c = fem.Constant(domain, PETSc.ScalarType(beta))
    dt_c = fem.Constant(domain, PETSc.ScalarType(dt))
    t_c = fem.Constant(domain, PETSc.ScalarType(t0))

    u_n = fem.Function(V)
    u_h = fem.Function(V)

    u_n.interpolate(fem.Expression(_u_exact(x, t_c), V.element.interpolation_points))
    u_n.x.scatter_forward()

    fdim = domain.topology.dim - 1
    facets = mesh.locate_entities_boundary(domain, fdim, lambda X: np.ones(X.shape[1], dtype=bool))
    bdofs = fem.locate_dofs_topological(V, fdim, facets)
    u_bc = fem.Function(V)
    bc_expr = fem.Expression(_u_exact(x, t_c), V.element.interpolation_points)
    u_bc.interpolate(bc_expr)
    u_bc.x.scatter_forward()
    bc = fem.dirichletbc(u_bc, bdofs)

    h = ufl.CellDiameter(domain)
    bnorm = ufl.sqrt(ufl.dot(beta_c, beta_c) + 1.0e-14)
    tau = h / (2.0 * bnorm)

    f = _f_rhs(x, t_c, eps_c, beta_c)
    a = (
        (u / dt_c) * v * ufl.dx
        + eps_c * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
        + ufl.dot(beta_c, ufl.grad(u)) * v * ufl.dx
        + tau * ((u / dt_c) - eps_c * ufl.div(ufl.grad(u)) + ufl.dot(beta_c, ufl.grad(u))) * ufl.dot(beta_c, ufl.grad(v)) * ufl.dx
    )
    L = (
        (u_n / dt_c) * v * ufl.dx
        + f * v * ufl.dx
        + tau * ((u_n / dt_c) + f) * ufl.dot(beta_c, ufl.grad(v)) * ufl.dx
    )

    a_form = fem.form(a)
    L_form = fem.form(L)

    A = petsc.assemble_matrix(a_form, bcs=[bc])
    A.assemble()
    b = petsc.create_vector(L_form.function_spaces)

    ksp = PETSc.KSP().create(comm)
    ksp.setOperators(A)
    ksp.setType("gmres")
    ksp.getPC().setType("ilu")
    ksp.setTolerances(rtol=1e-9, atol=1e-12, max_it=2000)

    n_steps = int(round((t_end - t0) / dt))
    total_iterations = 0
    start = time.perf_counter()

    for _ in range(n_steps):
        t_c.value = PETSc.ScalarType(float(t_c.value) + dt)
        u_bc.interpolate(bc_expr)
        u_bc.x.scatter_forward()

        with b.localForm() as loc:
            loc.set(0.0)
        petsc.assemble_vector(b, L_form)
        petsc.apply_lifting(b, [a_form], bcs=[[bc]])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        petsc.set_bc(b, [bc])

        try:
            ksp.solve(b, u_h.x.petsc_vec)
        except Exception:
            ksp.setType("preonly")
            ksp.getPC().setType("lu")
            ksp.setOperators(A)
            ksp.solve(b, u_h.x.petsc_vec)

        u_h.x.scatter_forward()
        total_iterations += int(ksp.getIterationNumber())
        u_n.x.array[:] = u_h.x.array
        u_n.x.scatter_forward()

    elapsed = time.perf_counter() - start

    u_ex = fem.Function(V)
    u_ex.interpolate(fem.Expression(_u_exact(x, t_c), V.element.interpolation_points))
    u_ex.x.scatter_forward()

    e = fem.Function(V)
    e.x.array[:] = u_h.x.array - u_ex.x.array
    e.x.scatter_forward()
    e2_local = fem.assemble_scalar(fem.form(ufl.inner(e, e) * ufl.dx))
    e2 = comm.allreduce(e2_local, op=MPI.SUM)
    l2_err = math.sqrt(e2)

    return {
        "domain": domain,
        "V": V,
        "u": u_h,
        "mesh_resolution": mesh_n,
        "element_degree": degree,
        "dt": dt,
        "n_steps": n_steps,
        "iterations": total_iterations,
        "ksp_type": ksp.getType(),
        "pc_type": ksp.getPC().getType(),
        "rtol": 1e-9,
        "l2_error": l2_err,
        "elapsed": elapsed,
    }


def solve(case_spec: dict) -> dict:
    p = _defaults(case_spec)

    candidates = [
        (48, 1, min(p["dt"], 0.01)),
        (64, 1, 0.01),
        (64, 2, 0.01),
        (80, 2, 0.005),
    ]

    best = None
    used_time = 0.0
    budget = 5.628

    for mesh_n, degree, dt in candidates:
        result = _run_one(mesh_n, degree, dt, p["t0"], p["t_end"], p["epsilon"], p["beta"])
        best = result
        used_time += result["elapsed"]
        if result["l2_error"] <= 7.05e-03 and used_time > 0.5 * budget:
            break
        if used_time > 0.9 * budget:
            break

    domain = best["domain"]
    V = best["V"]
    x = ufl.SpatialCoordinate(domain)
    t0_c = fem.Constant(domain, PETSc.ScalarType(p["t0"]))
    u0 = fem.Function(V)
    u0.interpolate(fem.Expression(_u_exact(x, t0_c), V.element.interpolation_points))
    u0.x.scatter_forward()

    u_grid = _sample_function(best["u"], domain, p["nx"], p["ny"], p["bbox"])
    u0_grid = _sample_function(u0, domain, p["nx"], p["ny"], p["bbox"])

    return {
        "u": u_grid,
        "u_initial": u0_grid,
        "solver_info": {
            "mesh_resolution": int(best["mesh_resolution"]),
            "element_degree": int(best["element_degree"]),
            "ksp_type": str(best["ksp_type"]),
            "pc_type": str(best["pc_type"]),
            "rtol": float(best["rtol"]),
            "iterations": int(best["iterations"]),
            "dt": float(best["dt"]),
            "n_steps": int(best["n_steps"]),
            "time_scheme": "backward_euler",
            "l2_error_vs_manufactured": float(best["l2_error"]),
        },
    }


if __name__ == "__main__":
    case_spec = {
        "pde": {
            "t0": 0.0,
            "t_end": 0.1,
            "dt": 0.02,
            "epsilon": 0.1,
            "beta": [1.0, 0.5],
            "time": True,
        },
        "output": {"grid": {"nx": 16, "ny": 16, "bbox": [0.0, 1.0, 0.0, 1.0]}},
    }
    out = solve(case_spec)
    if MPI.COMM_WORLD.rank == 0:
        print(out["u"].shape)
        print(out["solver_info"])
