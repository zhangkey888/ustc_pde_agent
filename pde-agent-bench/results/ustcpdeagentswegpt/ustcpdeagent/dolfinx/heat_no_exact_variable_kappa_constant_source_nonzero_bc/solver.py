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
# special_notes: variable_coeff
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


def _extract_time(case_spec):
    pde = case_spec.get("pde", {})
    t0 = float(pde.get("t0", case_spec.get("t0", 0.0)))
    t_end = float(pde.get("t_end", case_spec.get("t_end", 0.1)))
    dt = float(pde.get("dt", case_spec.get("dt", 0.02)))
    if dt <= 0.0:
        dt = 0.02
    return t0, t_end, dt


def _choose_parameters(case_spec):
    user_dt = _extract_time(case_spec)[2]
    mesh_resolution = 72
    degree = 1
    dt = min(user_dt, 0.01) if user_dt > 0.0 else 0.01
    return mesh_resolution, degree, dt


def _sample_function_on_grid(u_func, grid):
    nx = int(grid["nx"])
    ny = int(grid["ny"])
    xmin, xmax, ymin, ymax = map(float, grid["bbox"])

    xs = np.linspace(xmin, xmax, nx)
    ys = np.linspace(ymin, ymax, ny)
    XX, YY = np.meshgrid(xs, ys, indexing="xy")

    pts = np.zeros((nx * ny, 3), dtype=np.float64)
    pts[:, 0] = XX.ravel()
    pts[:, 1] = YY.ravel()

    domain = u_func.function_space.mesh
    tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(tree, pts)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, pts)

    local_values = np.full((pts.shape[0],), -np.inf, dtype=np.float64)
    points_on_proc = []
    cells_on_proc = []
    eval_map = []

    for i in range(pts.shape[0]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(pts[i])
            cells_on_proc.append(links[0])
            eval_map.append(i)

    if points_on_proc:
        vals = u_func.eval(np.array(points_on_proc, dtype=np.float64),
                           np.array(cells_on_proc, dtype=np.int32))
        local_values[np.array(eval_map, dtype=np.int32)] = np.asarray(vals).reshape(-1)

    global_values = np.empty_like(local_values)
    domain.comm.Allreduce(local_values, global_values, op=MPI.MAX)
    global_values[~np.isfinite(global_values)] = 0.0
    return global_values.reshape((ny, nx))


def _run_heat(mesh_resolution, degree, dt, t0, t_end, grid):
    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_square(comm, mesh_resolution, mesh_resolution, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", degree))

    x = ufl.SpatialCoordinate(domain)
    kappa = 1.0 + 0.5 * ufl.sin(2.0 * ufl.pi * x[0]) * ufl.sin(2.0 * ufl.pi * x[1])
    f = fem.Constant(domain, ScalarType(1.0))

    u_n = fem.Function(V)
    u_n.x.array[:] = 0.0
    u_n.x.scatter_forward()

    bc_fun = fem.Function(V)
    bc_fun.interpolate(lambda x: np.sin(np.pi * x[0]) + np.cos(np.pi * x[1]))

    fdim = domain.topology.dim - 1
    facets = mesh.locate_entities_boundary(domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool))
    dofs = fem.locate_dofs_topological(V, fdim, facets)
    bc = fem.dirichletbc(bc_fun, dofs)

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    a = (u * v + dt * kappa * ufl.inner(ufl.grad(u), ufl.grad(v))) * ufl.dx
    L = (u_n * v + dt * f * v) * ufl.dx

    a_form = fem.form(a)
    L_form = fem.form(L)

    A = petsc.assemble_matrix(a_form, bcs=[bc])
    A.assemble()
    b = petsc.create_vector(L_form.function_spaces)

    uh = fem.Function(V)

    solver = PETSc.KSP().create(comm)
    solver.setOperators(A)
    solver.setType("cg")
    solver.getPC().setType("hypre")
    solver.setTolerances(rtol=1e-10, atol=1e-12, max_it=5000)

    n_steps = int(round((t_end - t0) / dt))
    total_iterations = 0

    u_initial = _sample_function_on_grid(u_n, grid)

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
            solver.solve(b, uh.x.petsc_vec)

        uh.x.scatter_forward()
        its = solver.getIterationNumber()
        if its > 0:
            total_iterations += its
        u_n.x.array[:] = uh.x.array
        u_n.x.scatter_forward()

    u_grid = _sample_function_on_grid(uh, grid)

    return {
        "u": u_grid,
        "u_initial": u_initial,
        "solver_info": {
            "mesh_resolution": int(mesh_resolution),
            "element_degree": int(degree),
            "ksp_type": solver.getType(),
            "pc_type": solver.getPC().getType(),
            "rtol": float(1e-10),
            "iterations": int(total_iterations),
            "dt": float(dt),
            "n_steps": int(n_steps),
            "time_scheme": "backward_euler",
        },
    }


def solve(case_spec: dict) -> dict:
    grid = case_spec["output"]["grid"]
    t0, t_end, _ = _extract_time(case_spec)
    mesh_resolution, degree, dt = _choose_parameters(case_spec)

    start = time.perf_counter()
    out = _run_heat(mesh_resolution, degree, dt, t0, t_end, grid)
    elapsed = time.perf_counter() - start

    if elapsed < 8.0:
        ref = _run_heat(mesh_resolution, degree, dt / 2.0, t0, t_end, grid)
        diff = out["u"] - ref["u"]
        out["solver_info"]["accuracy_verification"] = {
            "temporal_self_convergence_l2_grid": float(np.sqrt(np.mean(diff**2))),
            "temporal_self_convergence_linf_grid": float(np.max(np.abs(diff))),
            "reference_dt": float(dt / 2.0),
        }
        if elapsed < 4.0 and out["solver_info"]["accuracy_verification"]["temporal_self_convergence_l2_grid"] > 2.5e-3:
            out = _run_heat(min(mesh_resolution + 16, 96), degree, dt / 2.0, t0, t_end, grid)
            ref2 = _run_heat(min(mesh_resolution + 16, 96), degree, dt / 4.0, t0, t_end, grid)
            diff2 = out["u"] - ref2["u"]
            out["solver_info"]["accuracy_verification"] = {
                "temporal_self_convergence_l2_grid": float(np.sqrt(np.mean(diff2**2))),
                "temporal_self_convergence_linf_grid": float(np.max(np.abs(diff2))),
                "reference_dt": float(dt / 4.0),
            }

    return out
