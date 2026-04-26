import time
import math
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl

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

ScalarType = PETSc.ScalarType


def _probe_function(u_func, pts):
    domain = u_func.function_space.mesh
    tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(tree, pts)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, pts)

    values = np.full((pts.shape[0],), np.nan, dtype=np.float64)
    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    for i in range(pts.shape[0]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(pts[i])
            cells_on_proc.append(links[0])
            eval_map.append(i)

    if len(points_on_proc) > 0:
        vals = u_func.eval(np.array(points_on_proc, dtype=np.float64),
                           np.array(cells_on_proc, dtype=np.int32))
        values[np.array(eval_map, dtype=np.int32)] = np.asarray(vals).reshape(-1).real
    return values


def _sample_on_grid(u_func, grid_spec):
    nx = int(grid_spec["nx"])
    ny = int(grid_spec["ny"])
    bbox = grid_spec["bbox"]
    xs = np.linspace(float(bbox[0]), float(bbox[1]), nx)
    ys = np.linspace(float(bbox[2]), float(bbox[3]), ny)
    XX, YY = np.meshgrid(xs, ys, indexing="xy")
    pts = np.column_stack([XX.ravel(), YY.ravel(), np.zeros(nx * ny, dtype=np.float64)])
    vals = _probe_function(u_func, pts)
    return vals.reshape(ny, nx)


def _run_heat(mesh_resolution, degree, dt, t_end, kappa_value, ksp_type="cg", pc_type="hypre", rtol=1e-8):
    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_square(comm, mesh_resolution, mesh_resolution, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", degree))

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    tdim = domain.topology.dim
    fdim = tdim - 1
    boundary_facets = mesh.locate_entities_boundary(domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool))
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc = fem.dirichletbc(ScalarType(0.0), boundary_dofs, V)

    x = ufl.SpatialCoordinate(domain)
    f_expr = ufl.sin(6.0 * ufl.pi * x[0]) * ufl.sin(6.0 * ufl.pi * x[1])
    f = fem.Expression(f_expr, V.element.interpolation_points)

    u_n = fem.Function(V)
    u_n.x.array[:] = 0.0

    uh = fem.Function(V)
    forcing = fem.Function(V)
    forcing.interpolate(f)

    kappa = fem.Constant(domain, ScalarType(kappa_value))
    dt_c = fem.Constant(domain, ScalarType(dt))

    a = (ufl.inner(u, v) + dt_c * kappa * ufl.inner(ufl.grad(u), ufl.grad(v))) * ufl.dx
    L = (ufl.inner(u_n, v) + dt_c * ufl.inner(forcing, v)) * ufl.dx

    a_form = fem.form(a)
    L_form = fem.form(L)

    A = petsc.assemble_matrix(a_form, bcs=[bc])
    A.assemble()
    b = petsc.create_vector(L_form.function_spaces)

    solver = PETSc.KSP().create(comm)
    solver.setOperators(A)
    solver.setType(ksp_type)
    pc = solver.getPC()
    pc.setType(pc_type)
    solver.setTolerances(rtol=rtol)
    solver.setFromOptions()

    n_steps = int(round(t_end / dt))
    t_actual = n_steps * dt
    total_iterations = 0

    for _ in range(n_steps):
        with b.localForm() as loc:
            loc.set(0.0)
        petsc.assemble_vector(b, L_form)
        petsc.apply_lifting(b, [a_form], bcs=[[bc]])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        petsc.set_bc(b, [bc])

        uh.x.array[:] = 0.0
        solver.solve(b, uh.x.petsc_vec)
        uh.x.scatter_forward()
        its = solver.getIterationNumber()
        total_iterations += int(max(its, 1 if ksp_type != "preonly" else 0))
        u_n.x.array[:] = uh.x.array

    return {
        "domain": domain,
        "V": V,
        "u": uh,
        "u_initial": fem.Function(V),
        "iterations": total_iterations,
        "n_steps": n_steps,
        "t_actual": t_actual,
        "ksp_type": solver.getType(),
        "pc_type": solver.getPC().getType(),
        "rtol": float(rtol),
    }


def solve(case_spec: dict) -> dict:
    t_start = time.perf_counter()

    pde = case_spec.get("pde", {})
    coef = case_spec.get("coefficients", {})
    output = case_spec.get("output", {})
    grid = output.get("grid", {"nx": 64, "ny": 64, "bbox": [0.0, 1.0, 0.0, 1.0]})

    t_end = float(pde.get("t_end", case_spec.get("t_end", 0.12)))
    dt_suggested = float(pde.get("dt", case_spec.get("dt", 0.02)))
    if t_end <= 0:
        t_end = 0.12
    if dt_suggested <= 0:
        dt_suggested = 0.02

    kappa_value = float(coef.get("kappa", 0.8))
    degree = 2

    candidate_settings = [
        (40, min(dt_suggested, 0.01)),
        (56, min(dt_suggested, 0.008)),
        (72, min(dt_suggested, 0.006)),
        (88, min(dt_suggested, 0.005)),
        (104, min(dt_suggested, 0.004)),
    ]

    best = None
    verification = {}
    last_elapsed = 0.0

    for mesh_resolution, dt in candidate_settings:
        run_start = time.perf_counter()
        result = _run_heat(mesh_resolution, degree, dt, t_end, kappa_value)
        elapsed = time.perf_counter() - run_start
        last_elapsed = elapsed

        if best is not None:
            fine_grid = _sample_on_grid(result["u"], grid)
            coarse_grid = _sample_on_grid(best["u"], grid)
            diff = np.linalg.norm(fine_grid - coarse_grid) / max(np.linalg.norm(fine_grid), 1e-14)
            verification = {
                "consistency_rel_l2": float(diff),
                "previous_mesh_resolution": int(best["mesh_resolution"]),
                "current_mesh_resolution": int(mesh_resolution),
                "previous_dt": float(best["dt"]),
                "current_dt": float(dt),
            }
            best = {
                "mesh_resolution": mesh_resolution,
                "dt": dt,
                **result,
            }
            if elapsed > 10.0 or diff < 2e-3 or (time.perf_counter() - t_start) > 14.0:
                break
        else:
            best = {
                "mesh_resolution": mesh_resolution,
                "dt": dt,
                **result,
            }
            if elapsed > 10.0:
                break

    u_grid = _sample_on_grid(best["u"], grid)
    u_initial_grid = np.zeros_like(u_grid)

    solver_info = {
        "mesh_resolution": int(best["mesh_resolution"]),
        "element_degree": int(degree),
        "ksp_type": str(best["ksp_type"]),
        "pc_type": str(best["pc_type"]),
        "rtol": float(best["rtol"]),
        "iterations": int(best["iterations"]),
        "dt": float(best["dt"]),
        "n_steps": int(best["n_steps"]),
        "time_scheme": "backward_euler",
        "accuracy_verification": verification if verification else {
            "consistency_rel_l2": None,
            "note": "single run only"
        },
        "wall_time_sec_estimate": float(time.perf_counter() - t_start),
        "last_run_sec": float(last_elapsed),
    }

    return {
        "u": u_grid,
        "u_initial": u_initial_grid,
        "solver_info": solver_info,
    }
