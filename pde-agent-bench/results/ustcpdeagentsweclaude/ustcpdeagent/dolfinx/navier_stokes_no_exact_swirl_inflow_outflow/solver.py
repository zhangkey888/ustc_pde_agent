import numpy as np
from mpi4py import MPI
from petsc4py import PETSc

from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from basix.ufl import element as basix_element, mixed_element as basix_mixed_element


ScalarType = PETSc.ScalarType


def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD

    # Defaults / selectable parameters with conservative accuracy-oriented choices
    nu = float(case_spec.get("pde", {}).get("nu", case_spec.get("nu", 0.22)))
    if "physics" in case_spec and isinstance(case_spec["physics"], dict):
        nu = float(case_spec["physics"].get("nu", nu))
    if "coefficients" in case_spec and isinstance(case_spec["coefficients"], dict):
        nu = float(case_spec["coefficients"].get("nu", nu))
    nu = 0.22 if nu is None else nu

    # Use a moderately fine mesh and Taylor-Hood P2/P1 for robustness/accuracy.
    mesh_resolution = int(case_spec.get("agent_params", {}).get("mesh_resolution", 56))
    degree_u = int(case_spec.get("agent_params", {}).get("degree_u", 2))
    degree_p = int(case_spec.get("agent_params", {}).get("degree_p", 1))
    newton_rtol = float(case_spec.get("agent_params", {}).get("newton_rtol", 1.0e-8))
    newton_max_it = int(case_spec.get("agent_params", {}).get("newton_max_it", 40))
    picard_max_it = int(case_spec.get("agent_params", {}).get("picard_max_it", 12))
    picard_tol = float(case_spec.get("agent_params", {}).get("picard_tol", 5.0e-9))

    # Mesh
    msh = mesh.create_unit_square(
        comm, mesh_resolution, mesh_resolution, cell_type=mesh.CellType.triangle
    )
    gdim = msh.geometry.dim
    cell_name = msh.topology.cell_name()

    # Mixed Taylor-Hood space
    vel_el = basix_element("Lagrange", cell_name, degree_u, shape=(gdim,))
    pres_el = basix_element("Lagrange", cell_name, degree_p)
    W = fem.functionspace(msh, basix_mixed_element([vel_el, pres_el]))
    V, _ = W.sub(0).collapse()
    Q, _ = W.sub(1).collapse()

    # Boundary conditions
    fdim = msh.topology.dim - 1

    def left(x):
        return np.isclose(x[0], 0.0)

    def bottom(x):
        return np.isclose(x[1], 0.0)

    def top(x):
        return np.isclose(x[1], 1.0)

    left_facets = mesh.locate_entities_boundary(msh, fdim, left)
    bottom_facets = mesh.locate_entities_boundary(msh, fdim, bottom)
    top_facets = mesh.locate_entities_boundary(msh, fdim, top)

    u_in = fem.Function(V)
    u_in.interpolate(
        lambda x: np.vstack(
            (
                np.sin(np.pi * x[1]),
                0.2 * np.sin(2.0 * np.pi * x[1]),
            )
        )
    )

    u_zero = fem.Function(V)
    u_zero.x.array[:] = 0.0

    left_dofs = fem.locate_dofs_topological((W.sub(0), V), fdim, left_facets)
    bottom_dofs = fem.locate_dofs_topological((W.sub(0), V), fdim, bottom_facets)
    top_dofs = fem.locate_dofs_topological((W.sub(0), V), fdim, top_facets)

    bc_left = fem.dirichletbc(u_in, left_dofs, W.sub(0))
    bc_bottom = fem.dirichletbc(u_zero, bottom_dofs, W.sub(0))
    bc_top = fem.dirichletbc(u_zero, top_dofs, W.sub(0))

    # Pressure pinning for uniqueness at corner (0,0)
    p_corner_dofs = fem.locate_dofs_geometrical(
        (W.sub(1), Q),
        lambda x: np.isclose(x[0], 0.0) & np.isclose(x[1], 0.0),
    )
    p_zero = fem.Function(Q)
    p_zero.x.array[:] = 0.0
    bcs = [bc_left, bc_bottom, bc_top]
    if len(p_corner_dofs) > 0:
        bc_p = fem.dirichletbc(p_zero, p_corner_dofs, W.sub(1))
        bcs.append(bc_p)

    # Unknowns/forms
    w = fem.Function(W)
    w_prev = fem.Function(W)

    u, p = ufl.split(w)
    v, q = ufl.TestFunctions(W)

    x = ufl.SpatialCoordinate(msh)
    f = fem.Constant(msh, np.array((0.0, 0.0), dtype=ScalarType))

    def eps(a):
        return ufl.sym(ufl.grad(a))

    # Initialize with boundary values
    w.x.array[:] = 0.0
    w_prev.x.array[:] = 0.0

    # Picard initialization: linearize convection with previous velocity
    nonlinear_iterations = []
    total_linear_iterations = 0

    u_prev, p_prev = ufl.split(w_prev)
    du, dp = ufl.TrialFunctions(W)

    a_picard = (
        2.0 * nu * ufl.inner(eps(du), eps(v)) * ufl.dx
        + ufl.inner(ufl.grad(du) * u_prev, v) * ufl.dx
        - dp * ufl.div(v) * ufl.dx
        + ufl.div(du) * q * ufl.dx
    )
    L_picard = ufl.inner(f, v) * ufl.dx

    picard_problem = petsc.LinearProblem(
        a_picard,
        L_picard,
        bcs=bcs,
        petsc_options_prefix="ns_picard_",
        petsc_options={
            "ksp_type": "gmres",
            "pc_type": "lu",
            "ksp_rtol": 1.0e-10,
        },
    )

    for k in range(picard_max_it):
        w.x.array[:] = w_prev.x.array
        w = picard_problem.solve()
        w.x.scatter_forward()

        dy = w.x.array - w_prev.x.array
        loc_num = np.dot(dy, dy)
        loc_den = np.dot(w.x.array, w.x.array)
        num = comm.allreduce(loc_num, op=MPI.SUM)
        den = comm.allreduce(loc_den, op=MPI.SUM)
        rel = np.sqrt(num / max(den, 1.0e-30))
        nonlinear_iterations.append(k + 1)
        # Cannot reliably extract iterations from LinearProblem; keep aggregate 0 if unavailable.
        if rel < picard_tol:
            break
        w_prev.x.array[:] = w.x.array

    # Newton refinement
    F = (
        2.0 * nu * ufl.inner(eps(u), eps(v)) * ufl.dx
        + ufl.inner(ufl.grad(u) * u, v) * ufl.dx
        - p * ufl.div(v) * ufl.dx
        + ufl.div(u) * q * ufl.dx
        - ufl.inner(f, v) * ufl.dx
    )
    J = ufl.derivative(F, w)

    problem = petsc.NonlinearProblem(
        F,
        w,
        bcs=bcs,
        J=J,
        petsc_options_prefix="ns_newton_",
        petsc_options={
            "snes_type": "newtonls",
            "snes_linesearch_type": "bt",
            "snes_rtol": newton_rtol,
            "snes_atol": 1.0e-10,
            "snes_max_it": newton_max_it,
            "ksp_type": "gmres",
            "pc_type": "lu",
            "ksp_rtol": 1.0e-10,
        },
    )
    w = problem.solve()
    w.x.scatter_forward()

    # Record nonlinear iterations if SNES is accessible; otherwise keep Picard count summary
    try:
        snes_it = int(problem.solver.getIterationNumber())  # type: ignore[attr-defined]
        nonlinear_iterations = [max(snes_it, 1)]
        try:
            total_linear_iterations = int(problem.solver.getLinearSolveIterations())  # type: ignore[attr-defined]
        except Exception:
            pass
    except Exception:
        if len(nonlinear_iterations) == 0:
            nonlinear_iterations = [1]

    # Collapse velocity for output and diagnostics
    u_h = w.sub(0).collapse()
    p_h = w.sub(1).collapse()

    # Accuracy verification module:
    # 1) compute L2 norm of div(u)
    # 2) compute kinetic energy
    div_u_form = fem.form(ufl.inner(ufl.div(u_h), ufl.div(u_h)) * ufl.dx)
    ke_form = fem.form(0.5 * ufl.inner(u_h, u_h) * ufl.dx)
    div_u_l2 = np.sqrt(comm.allreduce(fem.assemble_scalar(div_u_form), op=MPI.SUM))
    kinetic_energy = comm.allreduce(fem.assemble_scalar(ke_form), op=MPI.SUM)

    # Sample on requested uniform grid
    grid = case_spec["output"]["grid"]
    nx = int(grid["nx"])
    ny = int(grid["ny"])
    xmin, xmax, ymin, ymax = [float(v) for v in grid["bbox"]]

    xs = np.linspace(xmin, xmax, nx)
    ys = np.linspace(ymin, ymax, ny)
    XX, YY = np.meshgrid(xs, ys, indexing="xy")
    pts = np.column_stack(
        [XX.ravel(), YY.ravel(), np.zeros(nx * ny, dtype=np.float64)]
    )

    tree = geometry.bb_tree(msh, msh.topology.dim)
    cell_candidates = geometry.compute_collisions_points(tree, pts)
    colliding_cells = geometry.compute_colliding_cells(msh, cell_candidates, pts)

    local_vals = np.full((nx * ny, gdim), np.nan, dtype=np.float64)
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
        vals = u_h.eval(np.array(points_on_proc, dtype=np.float64), np.array(cells_on_proc, dtype=np.int32))
        vals = np.asarray(vals, dtype=np.float64).reshape(len(points_on_proc), gdim)
        local_vals[np.array(eval_map, dtype=np.int32), :] = vals

    gathered = comm.gather(local_vals, root=0)

    if comm.rank == 0:
        merged = np.full((nx * ny, gdim), np.nan, dtype=np.float64)
        for arr in gathered:
            mask = np.isfinite(arr[:, 0])
            merged[mask, :] = arr[mask, :]
        # Fill any remaining NaNs robustly (should not happen for boundary points in unit square)
        nan_mask = ~np.isfinite(merged[:, 0])
        if np.any(nan_mask):
            merged[nan_mask, :] = 0.0
        mag = np.linalg.norm(merged, axis=1).reshape(ny, nx)
        solver_info = {
            "mesh_resolution": mesh_resolution,
            "element_degree": degree_u,
            "ksp_type": "gmres",
            "pc_type": "lu",
            "rtol": 1.0e-10,
            "iterations": int(total_linear_iterations),
            "nonlinear_iterations": nonlinear_iterations,
            "verification": {
                "div_u_l2": float(div_u_l2),
                "kinetic_energy": float(kinetic_energy),
                "pressure_l2": float(
                    np.sqrt(comm.allreduce(fem.assemble_scalar(fem.form(ufl.inner(p_h, p_h) * ufl.dx)), op=MPI.SUM))
                ),
            },
        }
        return {"u": mag, "solver_info": solver_info}
    else:
        return {"u": np.empty((ny, nx), dtype=np.float64), "solver_info": {
            "mesh_resolution": mesh_resolution,
            "element_degree": degree_u,
            "ksp_type": "gmres",
            "pc_type": "lu",
            "rtol": 1.0e-10,
            "iterations": int(total_linear_iterations),
            "nonlinear_iterations": nonlinear_iterations,
            "verification": {
                "div_u_l2": float(div_u_l2),
                "kinetic_energy": float(kinetic_energy),
            },
        }}


if __name__ == "__main__":
    case_spec = {
        "pde": {"nu": 0.22, "time": None},
        "output": {
            "grid": {
                "nx": 64,
                "ny": 64,
                "bbox": [0.0, 1.0, 0.0, 1.0],
            }
        },
    }
    out = solve(case_spec)
    if MPI.COMM_WORLD.rank == 0:
        print(out["u"].shape)
        print(out["solver_info"])
