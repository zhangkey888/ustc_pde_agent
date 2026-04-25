import time
import math
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl

ScalarType = PETSc.ScalarType

# DIAGNOSIS: transient 2D high-Pe convection-diffusion, scalar, linear, all-Dirichlet, manufactured solution
# METHOD: FEM with Lagrange elements, backward Euler, SUPG stabilization, GMRES/ILU with fallback options


def _get_case_time(case_spec):
    pde = case_spec.get("pde", {})
    time_spec = pde.get("time", case_spec.get("time", {}))
    t0 = float(time_spec.get("t0", 0.0))
    t_end = float(time_spec.get("t_end", 0.06))
    dt = float(time_spec.get("dt", 0.005))
    scheme = time_spec.get("scheme", "backward_euler")
    return t0, t_end, dt, scheme


def _sample_on_grid(domain, uh, grid_spec):
    nx = int(grid_spec["nx"])
    ny = int(grid_spec["ny"])
    xmin, xmax, ymin, ymax = map(float, grid_spec["bbox"])
    xs = np.linspace(xmin, xmax, nx)
    ys = np.linspace(ymin, ymax, ny)
    XX, YY = np.meshgrid(xs, ys, indexing="xy")
    pts = np.column_stack([XX.ravel(), YY.ravel(), np.zeros(nx * ny, dtype=np.float64)])

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

    local_vals = np.full(pts.shape[0], np.nan, dtype=np.float64)
    if points_on_proc:
        vals = uh.eval(np.array(points_on_proc, dtype=np.float64),
                       np.array(cells_on_proc, dtype=np.int32))
        local_vals[np.array(eval_map, dtype=np.int32)] = np.asarray(vals).reshape(-1)

    finite = np.isfinite(local_vals).astype(np.int32)
    owner = np.zeros_like(finite)
    domain.comm.Allreduce(finite, owner, op=MPI.MAX)

    send = np.nan_to_num(local_vals, nan=0.0)
    recv = np.zeros_like(send)
    domain.comm.Allreduce(send, recv, op=MPI.SUM)

    out = recv.reshape(ny, nx)
    return out


def _run_candidate(mesh_n, degree, dt, t0, t_end, beta_vec, epsilon, ksp_type="gmres", pc_type="ilu", rtol=1e-9):
    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_square(comm, mesh_n, mesh_n, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", degree))

    x = ufl.SpatialCoordinate(domain)
    eps_c = fem.Constant(domain, ScalarType(epsilon))
    beta_c = fem.Constant(domain, np.array(beta_vec, dtype=ScalarType))
    t_c = fem.Constant(domain, ScalarType(t0))
    dt_c = fem.Constant(domain, ScalarType(dt))

    u_exact_ufl = ufl.exp(-t_c) * ufl.sin(4.0 * ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])
    grad_u = ufl.as_vector((
        4.0 * ufl.pi * ufl.exp(-t_c) * ufl.cos(4.0 * ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1]),
        ufl.pi * ufl.exp(-t_c) * ufl.sin(4.0 * ufl.pi * x[0]) * ufl.cos(ufl.pi * x[1]),
    ))
    lap_u = -(16.0 * ufl.pi**2 + ufl.pi**2) * u_exact_ufl
    f_ufl = -u_exact_ufl - eps_c * lap_u + ufl.dot(beta_c, grad_u)

    u_n = fem.Function(V)
    uh = fem.Function(V)
    u_bc = fem.Function(V)
    u_exact_fun = fem.Function(V)
    u0_fun = fem.Function(V)

    def interpolate_exact(target, tval):
        t_c.value = ScalarType(tval)
        expr = fem.Expression(u_exact_ufl, V.element.interpolation_points)
        target.interpolate(expr)
        target.x.scatter_forward()

    interpolate_exact(u0_fun, t0)
    u_n.x.array[:] = u0_fun.x.array
    u_n.x.scatter_forward()

    fdim = domain.topology.dim - 1
    facets = mesh.locate_entities_boundary(domain, fdim, lambda X: np.ones(X.shape[1], dtype=bool))
    bdofs = fem.locate_dofs_topological(V, fdim, facets)
    interpolate_exact(u_bc, t0)
    bc = fem.dirichletbc(u_bc, bdofs)

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    h = ufl.CellDiameter(domain)
    beta_norm = ufl.sqrt(ufl.dot(beta_c, beta_c) + 1.0e-14)
    tau = 1.0 / ufl.sqrt((2.0 / dt_c)**2 + (2.0 * beta_norm / h)**2 + (36.0 * eps_c / h**2)**2)

    a = (
        (u / dt_c) * v * ufl.dx
        + eps_c * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
        + ufl.dot(beta_c, ufl.grad(u)) * v * ufl.dx
        + tau * ((u / dt_c) - eps_c * ufl.div(ufl.grad(u)) + ufl.dot(beta_c, ufl.grad(u)))
        * ufl.dot(beta_c, ufl.grad(v)) * ufl.dx
    )
    L = (
        (u_n / dt_c) * v * ufl.dx
        + f_ufl * v * ufl.dx
        + tau * (f_ufl + u_n / dt_c) * ufl.dot(beta_c, ufl.grad(v)) * ufl.dx
    )

    a_form = fem.form(a)
    L_form = fem.form(L)

    A = petsc.assemble_matrix(a_form, bcs=[bc])
    A.assemble()
    b = petsc.create_vector(L_form.function_spaces)

    solver = PETSc.KSP().create(comm)
    solver.setOperators(A)
    solver.setType(ksp_type)
    solver.getPC().setType(pc_type)
    solver.setTolerances(rtol=rtol, atol=1e-14, max_it=5000)
    solver.setFromOptions()

    n_steps = int(round((t_end - t0) / dt))
    total_iterations = 0
    for step in range(n_steps):
        t_now = t0 + (step + 1) * dt
        t_c.value = ScalarType(t_now)
        interpolate_exact(u_bc, t_now)

        with b.localForm() as loc:
            loc.set(0.0)
        petsc.assemble_vector(b, L_form)
        petsc.apply_lifting(b, [a_form], bcs=[[bc]])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        petsc.set_bc(b, [bc])

        try:
            solver.solve(b, uh.x.petsc_vec)
        except Exception:
            solver.setType("preonly")
            solver.getPC().setType("lu")
            solver.setOperators(A)
            solver.solve(b, uh.x.petsc_vec)
        uh.x.scatter_forward()
        try:
            total_iterations += int(solver.getIterationNumber())
        except Exception:
            pass
        u_n.x.array[:] = uh.x.array
        u_n.x.scatter_forward()

    interpolate_exact(u_exact_fun, t_end)
    err_fun = fem.Function(V)
    err_fun.x.array[:] = uh.x.array - u_exact_fun.x.array
    err_fun.x.scatter_forward()

    l2_sq = fem.assemble_scalar(fem.form(ufl.inner(err_fun, err_fun) * ufl.dx))
    ex_sq = fem.assemble_scalar(fem.form(ufl.inner(u_exact_fun, u_exact_fun) * ufl.dx))
    l2_sq = comm.allreduce(l2_sq, op=MPI.SUM)
    ex_sq = comm.allreduce(ex_sq, op=MPI.SUM)

    return {
        "domain": domain,
        "solution": uh,
        "u_initial": u0_fun,
        "l2_error": math.sqrt(max(l2_sq, 0.0)),
        "rel_l2_error": math.sqrt(max(l2_sq, 0.0)) / (math.sqrt(max(ex_sq, 0.0)) + 1e-16),
        "mesh_resolution": mesh_n,
        "element_degree": degree,
        "ksp_type": solver.getType(),
        "pc_type": solver.getPC().getType(),
        "rtol": rtol,
        "iterations": total_iterations,
        "dt": dt,
        "n_steps": n_steps,
    }


def solve(case_spec: dict) -> dict:
    t0, t_end, dt_suggested, scheme = _get_case_time(case_spec)
    pde = case_spec.get("pde", {})
    epsilon = float(pde.get("epsilon", 0.01))
    beta_vec = pde.get("beta", [12.0, 4.0])
    output_grid = case_spec["output"]["grid"]

    best = _run_candidate(56, 2, min(dt_suggested, 0.004), t0, t_end, beta_vec, epsilon)

    u_grid = _sample_on_grid(best["domain"], best["solution"], output_grid)
    u0_grid = _sample_on_grid(best["domain"], best["u_initial"], output_grid)

    solver_info = {
        "mesh_resolution": int(best["mesh_resolution"]),
        "element_degree": int(best["element_degree"]),
        "ksp_type": str(best["ksp_type"]),
        "pc_type": str(best["pc_type"]),
        "rtol": float(best["rtol"]),
        "iterations": int(best["iterations"]),
        "dt": float(best["dt"]),
        "n_steps": int(best["n_steps"]),
        "time_scheme": str(scheme),
        "l2_error_vs_manufactured": float(best["l2_error"]),
        "rel_l2_error_vs_manufactured": float(best["rel_l2_error"]),
    }

    return {"u": u_grid, "u_initial": u0_grid, "solver_info": solver_info}


if __name__ == "__main__":
    case_spec = {
        "pde": {
            "epsilon": 0.01,
            "beta": [12.0, 4.0],
            "time": {"t0": 0.0, "t_end": 0.06, "dt": 0.005, "scheme": "backward_euler"},
        },
        "output": {"grid": {"nx": 32, "ny": 32, "bbox": [0.0, 1.0, 0.0, 1.0]}},
    }
    result = solve(case_spec)
    if MPI.COMM_WORLD.rank == 0:
        print(result["u"].shape)
        print(result["solver_info"])
