import math
import time
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl

ScalarType = PETSc.ScalarType

# ```DIAGNOSIS
# equation_type: heat
# spatial_dim: 2
# domain_geometry: rectangle
# unknowns: scalar
# coupling: none
# linearity: linear
# time_dependence: transient
# stiffness: stiff
# dominant_physics: diffusion
# peclet_or_reynolds: N/A
# solution_regularity: smooth
# bc_type: all_dirichlet
# special_notes: manufactured_solution
# ```
#
# ```METHOD
# spatial_method: fem
# element_or_basis: Lagrange_P1
# stabilization: none
# time_method: backward_euler
# nonlinear_solver: none
# linear_solver: cg
# preconditioner: hypre
# special_treatment: none
# pde_skill: heat
# ```


def _probe_function(u_func, points):
    domain = u_func.function_space.mesh
    tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(tree, points)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points)

    points_on_proc = []
    cells = []
    ids = []
    for i in range(points.shape[0]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(points[i])
            cells.append(links[0])
            ids.append(i)

    values_local = np.full(points.shape[0], np.nan, dtype=np.float64)
    if len(points_on_proc) > 0:
        vals = u_func.eval(np.array(points_on_proc, dtype=np.float64),
                           np.array(cells, dtype=np.int32))
        values_local[np.array(ids, dtype=np.int32)] = np.array(vals).reshape(-1)

    comm = domain.comm
    gathered = comm.gather(values_local, root=0)
    if comm.rank == 0:
        out = np.full(points.shape[0], np.nan, dtype=np.float64)
        for arr in gathered:
            mask = np.isfinite(arr)
            out[mask] = arr[mask]
    else:
        out = None
    out = comm.bcast(out, root=0)
    return out


def _run_config(mesh_n, degree, dt, t_end):
    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_square(comm, mesh_n, mesh_n, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", degree))
    x = ufl.SpatialCoordinate(domain)

    kappa = 0.5
    fdim = domain.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda X: np.ones(X.shape[1], dtype=bool)
    )
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)

    t_bc = fem.Constant(domain, ScalarType(0.0))

    uD = fem.Function(V)
    uD.interpolate(lambda X: np.exp(-2.0 * float(t_bc.value)) * np.cos(np.pi * X[0]) * np.cos(np.pi * X[1]))
    bc = fem.dirichletbc(uD, boundary_dofs)

    u_n = fem.Function(V)
    u_n.interpolate(lambda X: np.cos(np.pi * X[0]) * np.cos(np.pi * X[1]))

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    dt_const = fem.Constant(domain, ScalarType(dt))

    rhs_factor = (-2.0 + kappa * 2.0 * np.pi**2)
    f_expr = ufl.exp(-2.0 * t_bc) * rhs_factor * ufl.cos(ufl.pi * x[0]) * ufl.cos(ufl.pi * x[1])

    a = (u * v + dt_const * kappa * ufl.inner(ufl.grad(u), ufl.grad(v))) * ufl.dx
    L = (u_n + dt_const * f_expr) * v * ufl.dx

    a_form = fem.form(a)
    L_form = fem.form(L)

    A = petsc.assemble_matrix(a_form, bcs=[bc])
    A.assemble()
    b = petsc.create_vector(L_form.function_spaces)

    solver = PETSc.KSP().create(comm)
    solver.setOperators(A)
    solver.setType("cg")
    solver.getPC().setType("hypre")
    solver.setTolerances(rtol=1e-10, atol=1e-12, max_it=5000)

    uh = fem.Function(V)
    n_steps = int(round(t_end / dt))
    ts = 0.0
    total_iterations = 0

    t_start = time.perf_counter()
    for _ in range(n_steps):
        ts += dt
        t_bc.value = ScalarType(ts)
        uD.interpolate(lambda X, tt=ts: np.exp(-2.0 * tt) * np.cos(np.pi * X[0]) * np.cos(np.pi * X[1]))

        with b.localForm() as loc:
            loc.set(0.0)
        petsc.assemble_vector(b, L_form)
        petsc.apply_lifting(b, [a_form], bcs=[[bc]])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        petsc.set_bc(b, [bc])

        solver.solve(b, uh.x.petsc_vec)
        uh.x.scatter_forward()
        total_iterations += solver.getIterationNumber()

        u_n.x.array[:] = uh.x.array
        u_n.x.scatter_forward()

    elapsed = time.perf_counter() - t_start

    u_exact = fem.Function(V)
    u_exact.interpolate(lambda X, tt=ts: np.exp(-2.0 * tt) * np.cos(np.pi * X[0]) * np.cos(np.pi * X[1]))
    err_form = fem.form((uh - u_exact) ** 2 * ufl.dx)
    err_l2 = math.sqrt(comm.allreduce(fem.assemble_scalar(err_form), op=MPI.SUM))

    return {
        "domain": domain,
        "uh": uh,
        "mesh_resolution": mesh_n,
        "element_degree": degree,
        "ksp_type": solver.getType(),
        "pc_type": solver.getPC().getType(),
        "rtol": 1e-10,
        "iterations": total_iterations,
        "dt": dt,
        "n_steps": n_steps,
        "time_scheme": "backward_euler",
        "l2_error": err_l2,
        "wall": elapsed,
    }


def solve(case_spec: dict) -> dict:
    pde = case_spec.get("pde", {})
    out_grid = case_spec.get("output", {}).get("grid", {})
    nx_out = int(out_grid.get("nx", 64))
    ny_out = int(out_grid.get("ny", 64))
    bbox = out_grid.get("bbox", [0.0, 1.0, 0.0, 1.0])

    t0 = float(pde.get("t0", 0.0))
    t_end = float(pde.get("t_end", 0.2))
    dt_suggest = float(pde.get("dt", 0.02))
    _ = t0  # fixed initial time

    candidates = [
        (32, 1, min(dt_suggest, 0.02)),
        (40, 1, 0.01),
        (48, 1, 0.01),
    ]

    best = None
    budget = 2.090
    controller_start = time.perf_counter()

    for mesh_n, degree, dt in candidates:
        if best is not None and (time.perf_counter() - controller_start) > 0.85 * budget:
            break
        try:
            result = _run_config(mesh_n, degree, dt, t_end)
        except Exception:
            continue
        if best is None:
            best = result
        else:
            if result["l2_error"] < best["l2_error"] and (time.perf_counter() - controller_start) < budget:
                best = result
            elif result["wall"] < 0.35 * budget and result["l2_error"] <= 1.05 * best["l2_error"]:
                best = result

    if best is None:
        best = _run_config(24, 1, min(dt_suggest, 0.02), t_end)

    xs = np.linspace(bbox[0], bbox[1], nx_out)
    ys = np.linspace(bbox[2], bbox[3], ny_out)
    XX, YY = np.meshgrid(xs, ys, indexing="xy")
    points = np.column_stack([XX.ravel(), YY.ravel(), np.zeros(nx_out * ny_out, dtype=np.float64)])

    u_vals = _probe_function(best["uh"], points).reshape(ny_out, nx_out)
    u_initial = np.cos(np.pi * XX) * np.cos(np.pi * YY)

    solver_info = {
        "mesh_resolution": int(best["mesh_resolution"]),
        "element_degree": int(best["element_degree"]),
        "ksp_type": str(best["ksp_type"]),
        "pc_type": str(best["pc_type"]),
        "rtol": float(best["rtol"]),
        "iterations": int(best["iterations"]),
        "dt": float(best["dt"]),
        "n_steps": int(best["n_steps"]),
        "time_scheme": str(best["time_scheme"]),
    }

    return {
        "u": u_vals,
        "solver_info": solver_info,
        "u_initial": u_initial,
    }


if __name__ == "__main__":
    case_spec = {
        "pde": {"t0": 0.0, "t_end": 0.2, "dt": 0.02, "time": True},
        "output": {"grid": {"nx": 32, "ny": 32, "bbox": [0.0, 1.0, 0.0, 1.0]}},
    }
    result = solve(case_spec)
    if MPI.COMM_WORLD.rank == 0:
        print(result["u"].shape)
        print(result["solver_info"])
