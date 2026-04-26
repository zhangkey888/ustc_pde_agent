# DIAGNOSIS
# equation_type: reaction_diffusion
# spatial_dim: 2
# domain_geometry: rectangle
# unknowns: scalar
# coupling: none
# linearity: nonlinear
# time_dependence: transient
# stiffness: stiff
# dominant_physics: mixed
# peclet_or_reynolds: N/A
# solution_regularity: smooth
# bc_type: all_dirichlet
# special_notes: manufactured_solution
#
# METHOD
# spatial_method: fem
# element_or_basis: Lagrange_P2
# stabilization: none
# time_method: backward_euler
# nonlinear_solver: newton
# linear_solver: gmres
# preconditioner: ilu
# special_treatment: none
# pde_skill: reaction_diffusion

from mpi4py import MPI
from petsc4py import PETSc
import numpy as np
import ufl
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc

ScalarType = PETSc.ScalarType


def _sample_on_grid(domain, u_func, grid_spec):
    nx = int(grid_spec["nx"])
    ny = int(grid_spec["ny"])
    bbox = grid_spec["bbox"]
    xmin, xmax, ymin, ymax = map(float, bbox)

    xs = np.linspace(xmin, xmax, nx, dtype=np.float64)
    ys = np.linspace(ymin, ymax, ny, dtype=np.float64)
    XX, YY = np.meshgrid(xs, ys, indexing="xy")

    pts = np.zeros((nx * ny, 3), dtype=np.float64)
    pts[:, 0] = XX.ravel()
    pts[:, 1] = YY.ravel()

    tree = geometry.bb_tree(domain, domain.topology.dim)
    candidates = geometry.compute_collisions_points(tree, pts)
    colliding = geometry.compute_colliding_cells(domain, candidates, pts)

    local_vals = np.full((pts.shape[0],), np.nan, dtype=np.float64)
    points_on_proc = []
    cells_on_proc = []
    eval_map = []

    for i in range(pts.shape[0]):
        links = colliding.links(i)
        if len(links) > 0:
            points_on_proc.append(pts[i])
            cells_on_proc.append(links[0])
            eval_map.append(i)

    if len(points_on_proc) > 0:
        vals = u_func.eval(
            np.array(points_on_proc, dtype=np.float64),
            np.array(cells_on_proc, dtype=np.int32),
        )
        local_vals[np.array(eval_map, dtype=np.int32)] = np.asarray(vals).reshape(-1)

    comm = domain.comm
    gathered = comm.gather(local_vals, root=0)
    if comm.rank == 0:
        merged = np.full_like(local_vals, np.nan)
        for arr in gathered:
            mask = ~np.isnan(arr)
            merged[mask] = arr[mask]
        if np.isnan(merged).any():
            merged[np.isnan(merged)] = 0.0
        out = merged.reshape(ny, nx)
    else:
        out = None
    out = comm.bcast(out, root=0)
    return out


def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD

    time_spec = case_spec.get("pde", {}).get("time", {})
    t0 = float(time_spec.get("t0", 0.0))
    t_end = float(time_spec.get("t_end", 0.15))
    dt_in = float(time_spec.get("dt", 0.005))
    time_scheme = str(time_spec.get("scheme", "backward_euler")).lower()
    if time_scheme != "backward_euler":
        time_scheme = "backward_euler"

    params = case_spec.get("params", {})
    mesh_resolution = int(params.get("mesh_resolution", 48))
    element_degree = int(params.get("element_degree", 2))
    epsilon = float(params.get("epsilon", 0.05))
    dt = float(params.get("dt", dt_in))
    newton_rtol = float(params.get("newton_rtol", 1.0e-10))

    n_steps = int(np.round((t_end - t0) / dt))
    n_steps = max(n_steps, 1)
    dt = (t_end - t0) / n_steps

    domain = mesh.create_unit_square(
        comm, nx=mesh_resolution, ny=mesh_resolution, cell_type=mesh.CellType.triangle
    )
    V = fem.functionspace(domain, ("Lagrange", element_degree))

    x = ufl.SpatialCoordinate(domain)
    t_const = fem.Constant(domain, ScalarType(t0))

    def u_exact(tt):
        return ufl.exp(-tt) * (ScalarType(0.3) * ufl.sin(ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1]))

    u_ex = u_exact(t_const)
    f_expr = -u_ex - epsilon * ufl.div(ufl.grad(u_ex)) + (u_ex**3 - u_ex)

    u_n = fem.Function(V)
    ic_expr = fem.Expression(u_exact(t_const), V.element.interpolation_points)
    u_n.interpolate(ic_expr)
    u_n.x.scatter_forward()

    fdim = domain.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda X: np.ones(X.shape[1], dtype=bool)
    )
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)

    u_bc = fem.Function(V)
    u_bc.interpolate(ic_expr)
    u_bc.x.scatter_forward()
    bc = fem.dirichletbc(u_bc, boundary_dofs)

    uh = fem.Function(V)
    uh.x.array[:] = u_n.x.array
    uh.x.scatter_forward()

    v = ufl.TestFunction(V)
    du = ufl.TrialFunction(V)

    F = (
        ((uh - u_n) / dt) * v * ufl.dx
        + epsilon * ufl.inner(ufl.grad(uh), ufl.grad(v)) * ufl.dx
        + (uh**3 - uh) * v * ufl.dx
        - f_expr * v * ufl.dx
    )
    J = ufl.derivative(F, uh, du)

    problem = petsc.NonlinearProblem(
        F,
        uh,
        bcs=[bc],
        J=J,
        petsc_options_prefix="rd_",
        petsc_options={
            "snes_type": "newtonls",
            "snes_linesearch_type": "bt",
            "snes_rtol": newton_rtol,
            "snes_atol": 1.0e-12,
            "snes_max_it": 25,
            "ksp_type": "gmres",
            "ksp_rtol": 1.0e-9,
            "pc_type": "ilu",
        },
    )

    u_initial = _sample_on_grid(domain, u_n, case_spec["output"]["grid"])

    nonlinear_iterations = []
    total_linear_iterations = 0

    for step in range(1, n_steps + 1):
        t_const.value = ScalarType(t0 + step * dt)

        bc_expr = fem.Expression(u_exact(t_const), V.element.interpolation_points)
        u_bc.interpolate(bc_expr)
        u_bc.x.scatter_forward()

        uh.x.array[:] = u_n.x.array
        uh.x.scatter_forward()

        problem.solve()
        uh.x.scatter_forward()

        try:
            snes = problem.solver
            nonlinear_iterations.append(int(snes.getIterationNumber()))
            total_linear_iterations += int(snes.getLinearSolveIterations())
        except Exception:
            nonlinear_iterations.append(0)

        u_n.x.array[:] = uh.x.array
        u_n.x.scatter_forward()

    t_const.value = ScalarType(t_end)
    u_exact_h = fem.Function(V)
    u_exact_h.interpolate(fem.Expression(u_exact(t_const), V.element.interpolation_points))
    u_exact_h.x.scatter_forward()

    err2_local = fem.assemble_scalar(fem.form((uh - u_exact_h) ** 2 * ufl.dx))
    ex2_local = fem.assemble_scalar(fem.form((u_exact_h) ** 2 * ufl.dx))
    err2 = comm.allreduce(err2_local, op=MPI.SUM)
    ex2 = comm.allreduce(ex2_local, op=MPI.SUM)
    l2_error = float(np.sqrt(err2))
    relative_l2_error = float(np.sqrt(err2 / max(ex2, 1.0e-30)))

    u_grid = _sample_on_grid(domain, uh, case_spec["output"]["grid"])

    solver_info = {
        "mesh_resolution": mesh_resolution,
        "element_degree": element_degree,
        "ksp_type": "gmres",
        "pc_type": "ilu",
        "rtol": 1.0e-9,
        "iterations": int(total_linear_iterations),
        "dt": float(dt),
        "n_steps": int(n_steps),
        "time_scheme": time_scheme,
        "nonlinear_iterations": nonlinear_iterations,
        "l2_error": l2_error,
        "relative_l2_error": relative_l2_error,
    }

    return {
        "u": u_grid,
        "u_initial": u_initial,
        "solver_info": solver_info,
    }
