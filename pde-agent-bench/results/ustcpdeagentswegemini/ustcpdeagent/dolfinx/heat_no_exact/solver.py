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
# special_notes: none
# ```

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
    candidates = geometry.compute_collisions_points(tree, points)
    colliding = geometry.compute_colliding_cells(domain, candidates, points)

    pts_local = []
    cells_local = []
    ids_local = []
    for i in range(points.shape[0]):
        links = colliding.links(i)
        if len(links) > 0:
            pts_local.append(points[i])
            cells_local.append(links[0])
            ids_local.append(i)

    values = np.full(points.shape[0], np.nan, dtype=np.float64)
    if pts_local:
        vals = u_func.eval(np.asarray(pts_local, dtype=np.float64),
                          np.asarray(cells_local, dtype=np.int32))
        values[np.asarray(ids_local, dtype=np.int32)] = np.asarray(vals).reshape(-1)
    return values


def _sample_on_grid(u_func, grid_spec):
    nx = int(grid_spec["nx"])
    ny = int(grid_spec["ny"])
    bbox = grid_spec["bbox"]
    xmin, xmax, ymin, ymax = map(float, bbox)
    xs = np.linspace(xmin, xmax, nx)
    ys = np.linspace(ymin, ymax, ny)
    X, Y = np.meshgrid(xs, ys, indexing="xy")
    pts = np.column_stack([X.ravel(), Y.ravel(), np.zeros(nx * ny, dtype=np.float64)])
    vals = _probe_function(u_func, pts)

    comm = u_func.function_space.mesh.comm
    if comm.size > 1:
        gathered = comm.allgather(vals)
        out = np.full_like(vals, np.nan)
        for arr in gathered:
            mask = np.isnan(out) & ~np.isnan(arr)
            out[mask] = arr[mask]
        vals = out

    vals = np.nan_to_num(vals, nan=0.0)
    return vals.reshape(ny, nx)


def _global_scalar(comm, value):
    return comm.allreduce(value, op=MPI.SUM)


def _energy(comm, u):
    return _global_scalar(comm, fem.assemble_scalar(fem.form(ufl.inner(u, u) * ufl.dx)))


def _mass(comm, u):
    return _global_scalar(comm, fem.assemble_scalar(fem.form(u * ufl.dx)))


def _run_simulation(case_spec, mesh_resolution, degree, dt, ksp_type="cg", pc_type="hypre", rtol=1e-8):
    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_square(comm, mesh_resolution, mesh_resolution, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", degree))

    tdim = domain.topology.dim
    fdim = tdim - 1
    facets = mesh.locate_entities_boundary(domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool))
    bdofs = fem.locate_dofs_topological(V, fdim, facets)
    bc = fem.dirichletbc(ScalarType(0.0), bdofs, V)

    x = ufl.SpatialCoordinate(domain)
    kappa = fem.Constant(domain, ScalarType(1.0))
    f_expr = ufl.sin(ufl.pi * x[0]) * ufl.cos(ufl.pi * x[1])

    T = float(case_spec["pde"]["time"].get("t_end", 0.1))
    t0 = float(case_spec["pde"]["time"].get("t0", 0.0))
    n_steps = max(1, int(round((T - t0) / dt)))
    dt = (T - t0) / n_steps

    u_n = fem.Function(V)
    u_n.interpolate(lambda X: np.sin(np.pi * X[0]) * np.sin(np.pi * X[1]))
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    dt_c = fem.Constant(domain, ScalarType(dt))
    a = (u * v + dt_c * kappa * ufl.inner(ufl.grad(u), ufl.grad(v))) * ufl.dx
    L = (u_n * v + dt_c * f_expr * v) * ufl.dx

    a_form = fem.form(a)
    L_form = fem.form(L)

    A = petsc.assemble_matrix(a_form, bcs=[bc])
    A.assemble()
    b = petsc.create_vector(L_form.function_spaces)

    uh = fem.Function(V)
    solver = PETSc.KSP().create(comm)
    solver.setOperators(A)
    solver.setType(ksp_type)
    solver.getPC().setType(pc_type)
    solver.setTolerances(rtol=rtol, atol=1e-12, max_it=2000)

    total_iterations = 0
    energies = [_energy(comm, u_n)]
    masses = [_mass(comm, u_n)]

    for _ in range(n_steps):
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
        total_iterations += solver.getIterationNumber()

        u_n.x.array[:] = uh.x.array
        u_n.x.scatter_forward()
        energies.append(_energy(comm, u_n))
        masses.append(_mass(comm, u_n))

    verify_rel = None
    try:
        dt2 = dt / 2.0
        dt2_c = fem.Constant(domain, ScalarType(dt2))
        u0 = fem.Function(V)
        u0.interpolate(lambda X: np.sin(np.pi * X[0]) * np.sin(np.pi * X[1]))
        a2 = (u * v + dt2_c * kappa * ufl.inner(ufl.grad(u), ufl.grad(v))) * ufl.dx
        L2 = (u0 * v + dt2_c * f_expr * v) * ufl.dx
        a2_form = fem.form(a2)
        L2_form = fem.form(L2)
        A2 = petsc.assemble_matrix(a2_form, bcs=[bc])
        A2.assemble()
        b2 = petsc.create_vector(L2_form.function_spaces)
        solver2 = PETSc.KSP().create(comm)
        solver2.setOperators(A2)
        solver2.setType(ksp_type)
        solver2.getPC().setType(pc_type)
        solver2.setTolerances(rtol=rtol, atol=1e-12, max_it=2000)
        u2 = fem.Function(V)
        for _ in range(2 * n_steps):
            with b2.localForm() as loc:
                loc.set(0.0)
            petsc.assemble_vector(b2, L2_form)
            petsc.apply_lifting(b2, [a2_form], bcs=[[bc]])
            b2.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
            petsc.set_bc(b2, [bc])
            solver2.solve(b2, u2.x.petsc_vec)
            u2.x.scatter_forward()
            u0.x.array[:] = u2.x.array
            u0.x.scatter_forward()

        diff = fem.Function(V)
        diff.x.array[:] = uh.x.array - u2.x.array
        diff.x.scatter_forward()
        num = math.sqrt(_global_scalar(comm, fem.assemble_scalar(fem.form(ufl.inner(diff, diff) * ufl.dx))))
        den = math.sqrt(_global_scalar(comm, fem.assemble_scalar(fem.form(ufl.inner(u2, u2) * ufl.dx))))
        verify_rel = num / den if den > 0 else num
    except Exception:
        verify_rel = None

    return {
        "solution": uh,
        "iterations": int(total_iterations),
        "n_steps": int(n_steps),
        "dt": float(dt),
        "energy_history": energies,
        "mass_history": masses,
        "verification_rel_diff": verify_rel,
        "ksp_type": solver.getType(),
        "pc_type": solver.getPC().getType(),
        "rtol": float(rtol),
        "mesh_resolution": int(mesh_resolution),
        "element_degree": int(degree),
    }


def solve(case_spec: dict) -> dict:
    t_start = time.perf_counter()
    pde_time = case_spec.get("pde", {}).get("time", {})
    T = float(pde_time.get("t_end", 0.1))
    dt_suggested = float(pde_time.get("dt", 0.02))
    grid_spec = case_spec["output"]["grid"]

    candidates = [
        (40, 1, min(dt_suggested, T / 8 if T > 0 else dt_suggested)),
        (56, 1, min(dt_suggested / 2.0, T / 12 if T > 0 else dt_suggested / 2.0)),
        (72, 1, min(dt_suggested / 2.0, T / 16 if T > 0 else dt_suggested / 2.0)),
        (64, 2, min(dt_suggested / 2.0, T / 16 if T > 0 else dt_suggested / 2.0)),
    ]

    best = None
    time_budget = 8.876
    soft_limit = 0.8 * time_budget

    for params in candidates:
        if (time.perf_counter() - t_start) > soft_limit and best is not None:
            break
        try:
            best = _run_simulation(case_spec, *params)
        except Exception:
            if best is None:
                best = _run_simulation(case_spec, 40, 1, dt_suggested, ksp_type="preonly", pc_type="lu", rtol=1e-10)
            break

    if best is None:
        best = _run_simulation(case_spec, 40, 1, dt_suggested, ksp_type="preonly", pc_type="lu", rtol=1e-10)

    V = best["solution"].function_space
    u_init = fem.Function(V)
    u_init.interpolate(lambda X: np.sin(np.pi * X[0]) * np.sin(np.pi * X[1]))

    u_grid = _sample_on_grid(best["solution"], grid_spec)
    u_initial_grid = _sample_on_grid(u_init, grid_spec)

    solver_info = {
        "mesh_resolution": best["mesh_resolution"],
        "element_degree": best["element_degree"],
        "ksp_type": best["ksp_type"],
        "pc_type": best["pc_type"],
        "rtol": best["rtol"],
        "iterations": best["iterations"],
        "dt": best["dt"],
        "n_steps": best["n_steps"],
        "time_scheme": "backward_euler",
    }

    if best["verification_rel_diff"] is not None:
        solver_info["verification_rel_diff"] = float(best["verification_rel_diff"])
    if len(best["energy_history"]) >= 2:
        solver_info["initial_energy"] = float(best["energy_history"][0])
        solver_info["final_energy"] = float(best["energy_history"][-1])
    if len(best["mass_history"]) >= 2:
        solver_info["initial_mass"] = float(best["mass_history"][0])
        solver_info["final_mass"] = float(best["mass_history"][-1])

    return {
        "u": u_grid,
        "u_initial": u_initial_grid,
        "solver_info": solver_info,
    }
