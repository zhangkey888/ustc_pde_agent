import math
import time
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
import ufl
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc

ScalarType = PETSc.ScalarType


def _default_case_spec(case_spec: dict) -> dict:
    spec = dict(case_spec) if case_spec is not None else {}
    pde = dict(spec.get("pde", {}))
    time_spec = dict(pde.get("time", {}))
    output = dict(spec.get("output", {}))
    grid = dict(output.get("grid", {}))

    time_spec.setdefault("t0", 0.0)
    time_spec.setdefault("t_end", 0.1)
    time_spec.setdefault("dt", 0.02)
    time_spec.setdefault("scheme", "backward_euler")
    pde["time"] = time_spec

    grid.setdefault("nx", 64)
    grid.setdefault("ny", 64)
    grid.setdefault("bbox", [0.0, 1.0, 0.0, 1.0])
    output["grid"] = grid
    spec["output"] = output
    spec["pde"] = pde
    return spec


def _probe_function(u_func: fem.Function, pts_xyz: np.ndarray) -> np.ndarray:
    domain = u_func.function_space.mesh
    tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(tree, pts_xyz)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, pts_xyz)

    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    for i in range(pts_xyz.shape[0]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(pts_xyz[i])
            cells_on_proc.append(links[0])
            eval_map.append(i)

    local_vals = np.full((pts_xyz.shape[0],), np.nan, dtype=np.float64)
    if len(points_on_proc) > 0:
        vals = u_func.eval(np.array(points_on_proc, dtype=np.float64),
                           np.array(cells_on_proc, dtype=np.int32))
        local_vals[np.array(eval_map, dtype=np.int32)] = np.asarray(vals).reshape(-1).real

    comm = domain.comm
    gathered = comm.gather(local_vals, root=0)
    if comm.rank == 0:
        out = np.full((pts_xyz.shape[0],), np.nan, dtype=np.float64)
        for arr in gathered:
            mask = ~np.isnan(arr)
            out[mask] = arr[mask]
        return out
    return None


def solve(case_spec: dict) -> dict:
    case_spec = _default_case_spec(case_spec)

    comm = MPI.COMM_WORLD
    rank = comm.rank

    eps = 0.1
    beta_vec = np.array([1.0, 0.5], dtype=np.float64)
    beta_norm = float(np.linalg.norm(beta_vec))

    time_spec = case_spec["pde"]["time"]
    t0 = float(time_spec.get("t0", 0.0))
    t_end = float(time_spec.get("t_end", 0.1))
    dt_suggested = float(time_spec.get("dt", 0.02))
    time_scheme = str(time_spec.get("scheme", "backward_euler")).lower()

    mesh_resolution = 40
    element_degree = 1
    dt = min(dt_suggested, 0.01)
    n_steps = max(1, int(round((t_end - t0) / dt)))
    dt = (t_end - t0) / n_steps

    domain = mesh.create_unit_square(comm, mesh_resolution, mesh_resolution, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", element_degree))

    x = ufl.SpatialCoordinate(domain)
    t_c = fem.Constant(domain, ScalarType(t0))
    eps_c = fem.Constant(domain, ScalarType(eps))
    dt_c = fem.Constant(domain, ScalarType(dt))
    beta_c = fem.Constant(domain, np.array(beta_vec, dtype=ScalarType))

    u_exact = ufl.exp(-t_c) * ufl.sin(ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])
    f_expr = ufl.diff(u_exact, t_c) - eps_c * ufl.div(ufl.grad(u_exact)) + ufl.dot(beta_c, ufl.grad(u_exact))

    u_n = fem.Function(V)
    u_n.interpolate(fem.Expression(
        ufl.exp(-ScalarType(t0)) * ufl.sin(ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1]),
        V.element.interpolation_points
    ))

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    h = ufl.CellDiameter(domain)
    tau = 1.0 / ufl.sqrt((2.0 / dt_c) ** 2 + (2.0 * beta_norm / h) ** 2 + (4.0 * eps / (h * h)) ** 2)

    a = (
        (u / dt_c) * v * ufl.dx
        + eps_c * ufl.dot(ufl.grad(u), ufl.grad(v)) * ufl.dx
        + ufl.dot(beta_c, ufl.grad(u)) * v * ufl.dx
        + tau * ((u / dt_c) - eps_c * ufl.div(ufl.grad(u)) + ufl.dot(beta_c, ufl.grad(u))) * ufl.dot(beta_c, ufl.grad(v)) * ufl.dx
    )
    L = (
        (u_n / dt_c) * v * ufl.dx
        + f_expr * v * ufl.dx
        + tau * ((u_n / dt_c) + f_expr) * ufl.dot(beta_c, ufl.grad(v)) * ufl.dx
    )

    fdim = domain.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(domain, fdim, lambda X: np.ones(X.shape[1], dtype=bool))
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    u_bc = fem.Function(V)
    bc = fem.dirichletbc(u_bc, boundary_dofs)

    a_form = fem.form(a)
    L_form = fem.form(L)
    A = petsc.assemble_matrix(a_form, bcs=[bc])
    A.assemble()
    b = petsc.create_vector(L_form.function_spaces)

    uh = fem.Function(V)

    solver = PETSc.KSP().create(comm)
    solver.setOperators(A)
    solver.setType("gmres")
    solver.getPC().setType("ilu")
    solver.setTolerances(rtol=1e-8, atol=1e-12, max_it=1000)
    solver.setFromOptions()

    grid = case_spec["output"]["grid"]
    nx = int(grid["nx"])
    ny = int(grid["ny"])
    xmin, xmax, ymin, ymax = map(float, grid["bbox"])
    xs = np.linspace(xmin, xmax, nx)
    ys = np.linspace(ymin, ymax, ny)
    XX, YY = np.meshgrid(xs, ys, indexing="xy")
    pts = np.column_stack([XX.ravel(), YY.ravel(), np.zeros(nx * ny, dtype=np.float64)])

    u_initial_flat = _probe_function(u_n, pts)

    total_iterations = 0
    wall0 = time.perf_counter()

    for step in range(1, n_steps + 1):
        t_c.value = ScalarType(t0 + step * dt)

        u_bc.interpolate(fem.Expression(
            ufl.exp(-t_c) * ufl.sin(ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1]),
            V.element.interpolation_points
        ))

        A.zeroEntries()
        petsc.assemble_matrix(A, a_form, bcs=[bc])
        A.assemble()

        with b.localForm() as loc:
            loc.set(0.0)
        petsc.assemble_vector(b, L_form)
        petsc.apply_lifting(b, [a_form], bcs=[[bc]])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        petsc.set_bc(b, [bc])

        solver.solve(b, uh.x.petsc_vec)
        uh.x.scatter_forward()
        total_iterations += int(max(0, solver.getIterationNumber()))

        u_n.x.array[:] = uh.x.array
        u_n.x.scatter_forward()

    wall_time = time.perf_counter() - wall0

    u_ex_final = fem.Function(V)
    u_ex_final.interpolate(fem.Expression(
        ufl.exp(-ScalarType(t_end)) * ufl.sin(ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1]),
        V.element.interpolation_points
    ))
    err = fem.Function(V)
    err.x.array[:] = uh.x.array - u_ex_final.x.array
    err.x.scatter_forward()

    l2_sq = fem.assemble_scalar(fem.form(err * err * ufl.dx))
    l2_ref_sq = fem.assemble_scalar(fem.form(u_ex_final * u_ex_final * ufl.dx))
    l2_error = math.sqrt(comm.allreduce(l2_sq, op=MPI.SUM))
    l2_ref = math.sqrt(comm.allreduce(l2_ref_sq, op=MPI.SUM))
    rel_l2_error = l2_error / l2_ref if l2_ref > 0 else l2_error

    u_grid_flat = _probe_function(uh, pts)

    if rank == 0:
        return {
            "u": u_grid_flat.reshape(ny, nx),
            "u_initial": u_initial_flat.reshape(ny, nx),
            "solver_info": {
                "mesh_resolution": mesh_resolution,
                "element_degree": element_degree,
                "ksp_type": solver.getType(),
                "pc_type": solver.getPC().getType(),
                "rtol": 1e-8,
                "iterations": int(total_iterations),
                "dt": float(dt),
                "n_steps": int(n_steps),
                "time_scheme": time_scheme,
                "l2_error": float(l2_error),
                "relative_l2_error": float(rel_l2_error),
                "wall_time_sec": float(wall_time),
                "stabilization": "supg"
            }
        }
    return {
        "u": None,
        "u_initial": None,
        "solver_info": {
            "mesh_resolution": mesh_resolution,
            "element_degree": element_degree,
            "ksp_type": solver.getType(),
            "pc_type": solver.getPC().getType(),
            "rtol": 1e-8,
            "iterations": int(total_iterations),
            "dt": float(dt),
            "n_steps": int(n_steps),
            "time_scheme": time_scheme
        }
    }


if __name__ == "__main__":
    case = {
        "pde": {"time": {"t0": 0.0, "t_end": 0.1, "dt": 0.02, "scheme": "backward_euler"}},
        "output": {"grid": {"nx": 32, "ny": 32, "bbox": [0.0, 1.0, 0.0, 1.0]}}
    }
    result = solve(case)
    if MPI.COMM_WORLD.rank == 0:
        print(result["u"].shape)
        print(result["solver_info"])
