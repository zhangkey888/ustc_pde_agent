import math
import time
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl

# ```DIAGNOSIS
# equation_type: convection_diffusion
# spatial_dim: 2
# domain_geometry: rectangle
# unknowns: scalar
# coupling: none
# linearity: linear
# time_dependence: transient
# stiffness: stiff
# dominant_physics: mixed
# peclet_or_reynolds: high
# solution_regularity: smooth
# bc_type: all_dirichlet
# special_notes: manufactured_solution
# ```
#
# ```METHOD
# spatial_method: fem
# element_or_basis: Lagrange_P1
# stabilization: supg
# time_method: backward_euler
# nonlinear_solver: none
# linear_solver: preonly
# preconditioner: lu
# special_treatment: none
# pde_skill: convection_diffusion
# ```


def _defaults(case_spec: dict):
    pde = case_spec.get("pde", {})
    grid = case_spec.get("output", {}).get("grid", {})
    return {
        "t0": float(pde.get("t0", 0.0)),
        "t_end": float(pde.get("t_end", 0.1)),
        "dt": float(pde.get("dt", 0.02)),
        "epsilon": float(pde.get("epsilon", 0.1)),
        "beta": np.array(pde.get("beta", [1.0, 0.5]), dtype=np.float64),
        "nx_out": int(grid.get("nx", 64)),
        "ny_out": int(grid.get("ny", 64)),
        "bbox": grid.get("bbox", [0.0, 1.0, 0.0, 1.0]),
    }


def _u_exact(x, t):
    return ufl.exp(-t) * ufl.sin(ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])


def _f_rhs(x, t, eps, beta):
    uex = _u_exact(x, t)
    grad_uex = ufl.as_vector(
        (
            ufl.exp(-t) * ufl.pi * ufl.cos(ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1]),
            ufl.exp(-t) * ufl.pi * ufl.sin(ufl.pi * x[0]) * ufl.cos(ufl.pi * x[1]),
        )
    )
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

    values = np.full(nx * ny, np.nan, dtype=np.float64)
    points_on_proc = []
    cells = []
    idx = []
    for i in range(nx * ny):
        links = colliding.links(i)
        if len(links) > 0:
            points_on_proc.append(pts[i])
            cells.append(links[0])
            idx.append(i)

    if points_on_proc:
        vals = uh.eval(np.array(points_on_proc, dtype=np.float64), np.array(cells, dtype=np.int32))
        values[np.array(idx, dtype=np.int32)] = np.asarray(vals).reshape(-1)

    gathered = domain.comm.allgather(values)
    out = np.full(nx * ny, np.nan, dtype=np.float64)
    for arr in gathered:
        mask = np.isnan(out) & ~np.isnan(arr)
        out[mask] = arr[mask]
    out[np.isnan(out)] = 0.0
    return out.reshape((ny, nx))


def _choose_parameters(dt_in):
    if dt_in <= 0.01:
        return 24, 1, dt_in
    return 24, 1, 0.01


def _run(case_spec: dict):
    comm = MPI.COMM_WORLD
    p = _defaults(case_spec)
    mesh_n, degree, dt = _choose_parameters(p["dt"])

    domain = mesh.create_unit_square(comm, mesh_n, mesh_n, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", degree))
    x = ufl.SpatialCoordinate(domain)

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    eps_c = fem.Constant(domain, PETSc.ScalarType(p["epsilon"]))
    beta_c = fem.Constant(domain, PETSc.ScalarType(p["beta"]))
    dt_c = fem.Constant(domain, PETSc.ScalarType(dt))
    t_c = fem.Constant(domain, PETSc.ScalarType(p["t0"]))

    u_n = fem.Function(V)
    u_h = fem.Function(V)
    exact_expr = fem.Expression(_u_exact(x, t_c), V.element.interpolation_points)
    u_n.interpolate(exact_expr)
    u_n.x.scatter_forward()

    fdim = domain.topology.dim - 1
    facets = mesh.locate_entities_boundary(domain, fdim, lambda X: np.ones(X.shape[1], dtype=bool))
    bdofs = fem.locate_dofs_topological(V, fdim, facets)
    u_bc = fem.Function(V)
    u_bc.interpolate(exact_expr)
    u_bc.x.scatter_forward()
    bc = fem.dirichletbc(u_bc, bdofs)

    h = ufl.CellDiameter(domain)
    bnorm = ufl.sqrt(ufl.dot(beta_c, beta_c) + 1.0e-14)
    tau = h / (2.0 * bnorm)

    f = _f_rhs(x, t_c, eps_c, beta_c)
    strong_res_trial = (u - u_n) / dt_c - eps_c * ufl.div(ufl.grad(u)) + ufl.dot(beta_c, ufl.grad(u)) - f
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
    ksp.setType("preonly")
    ksp.getPC().setType("lu")
    ksp.setTolerances(rtol=1e-10, atol=1e-12, max_it=1)

    n_steps = int(round((p["t_end"] - p["t0"]) / dt))
    total_iterations = 0
    start = time.perf_counter()

    for _ in range(n_steps):
        t_c.value = PETSc.ScalarType(float(t_c.value) + dt)
        u_bc.interpolate(exact_expr)
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
            ksp.setType("gmres")
            ksp.getPC().setType("ilu")
            ksp.setTolerances(rtol=1e-8, atol=1e-10, max_it=500)
            ksp.setOperators(A)
            ksp.solve(b, u_h.x.petsc_vec)

        u_h.x.scatter_forward()
        total_iterations += max(1, int(ksp.getIterationNumber()))
        u_n.x.array[:] = u_h.x.array
        u_n.x.scatter_forward()

    elapsed = time.perf_counter() - start

    u_ex = fem.Function(V)
    u_ex.interpolate(exact_expr)
    u_ex.x.scatter_forward()

    e = fem.Function(V)
    e.x.array[:] = u_h.x.array - u_ex.x.array
    e.x.scatter_forward()
    e2_local = fem.assemble_scalar(fem.form(ufl.inner(e, e) * ufl.dx))
    e2 = comm.allreduce(e2_local, op=MPI.SUM)
    l2_err = math.sqrt(e2)

    t0_c = fem.Constant(domain, PETSc.ScalarType(p["t0"]))
    u0 = fem.Function(V)
    u0.interpolate(fem.Expression(_u_exact(x, t0_c), V.element.interpolation_points))
    u0.x.scatter_forward()

    return {
        "domain": domain,
        "V": V,
        "u": u_h,
        "u0": u0,
        "mesh_resolution": mesh_n,
        "element_degree": degree,
        "ksp_type": str(ksp.getType()),
        "pc_type": str(ksp.getPC().getType()),
        "rtol": 1e-10 if str(ksp.getType()) == "preonly" else 1e-8,
        "iterations": total_iterations,
        "dt": dt,
        "n_steps": n_steps,
        "time_scheme": "backward_euler",
        "l2_error": l2_err,
        "elapsed": elapsed,
    }


def solve(case_spec: dict) -> dict:
    result = _run(case_spec)
    p = _defaults(case_spec)

    u_grid = _sample_function(result["u"], result["domain"], p["nx_out"], p["ny_out"], p["bbox"])
    u0_grid = _sample_function(result["u0"], result["domain"], p["nx_out"], p["ny_out"], p["bbox"])

    return {
        "u": u_grid,
        "u_initial": u0_grid,
        "solver_info": {
            "mesh_resolution": int(result["mesh_resolution"]),
            "element_degree": int(result["element_degree"]),
            "ksp_type": str(result["ksp_type"]),
            "pc_type": str(result["pc_type"]),
            "rtol": float(result["rtol"]),
            "iterations": int(result["iterations"]),
            "dt": float(result["dt"]),
            "n_steps": int(result["n_steps"]),
            "time_scheme": str(result["time_scheme"]),
            "l2_error_vs_manufactured": float(result["l2_error"]),
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
    t0 = time.perf_counter()
    out = solve(case_spec)
    wall = time.perf_counter() - t0
    if MPI.COMM_WORLD.rank == 0:
        print("L2_ERROR:", out["solver_info"]["l2_error_vs_manufactured"])
        print("WALL_TIME:", wall)
        print(out["u"].shape)
        print(out["solver_info"])
