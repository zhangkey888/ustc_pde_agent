import math
import time
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl

ScalarType = PETSc.ScalarType

DIAGNOSIS = "```DIAGNOSIS\n" \
    "equation_type: heat\n" \
    "spatial_dim: 2\n" \
    "domain_geometry: rectangle\n" \
    "unknowns: scalar\n" \
    "coupling: none\n" \
    "linearity: linear\n" \
    "time_dependence: transient\n" \
    "stiffness: stiff\n" \
    "dominant_physics: diffusion\n" \
    "peclet_or_reynolds: N/A\n" \
    "solution_regularity: smooth\n" \
    "bc_type: all_dirichlet\n" \
    "special_notes: manufactured_solution\n" \
    "```"

METHOD = "```METHOD\n" \
    "spatial_method: fem\n" \
    "element_or_basis: Lagrange_P2\n" \
    "stabilization: none\n" \
    "time_method: backward_euler\n" \
    "nonlinear_solver: none\n" \
    "linear_solver: cg\n" \
    "preconditioner: hypre\n" \
    "special_treatment: none\n" \
    "pde_skill: heat\n" \
    "```"


def _exact_numpy(x, t):
    return np.exp(-0.5 * t) * np.sin(2.0 * np.pi * x[0]) * np.sin(np.pi * x[1])


def _with_defaults(case_spec):
    case = {} if case_spec is None else dict(case_spec)
    case.setdefault("pde", {})
    case["pde"].setdefault("time", {})
    case["pde"]["time"].setdefault("t0", 0.0)
    case["pde"]["time"].setdefault("t_end", 0.2)
    case["pde"]["time"].setdefault("dt", 0.02)
    case.setdefault("output", {})
    case["output"].setdefault("grid", {})
    case["output"]["grid"].setdefault("nx", 64)
    case["output"]["grid"].setdefault("ny", 64)
    case["output"]["grid"].setdefault("bbox", [0.0, 1.0, 0.0, 1.0])
    return case


def _sample_scalar_function_on_grid(domain, uh, grid):
    nx = int(grid["nx"])
    ny = int(grid["ny"])
    xmin, xmax, ymin, ymax = grid["bbox"]
    xs = np.linspace(xmin, xmax, nx)
    ys = np.linspace(ymin, ymax, ny)
    X, Y = np.meshgrid(xs, ys, indexing="xy")
    pts = np.column_stack([X.ravel(), Y.ravel(), np.zeros(nx * ny, dtype=np.float64)])

    tree = geometry.bb_tree(domain, domain.topology.dim)
    candidates = geometry.compute_collisions_points(tree, pts)
    colliding = geometry.compute_colliding_cells(domain, candidates, pts)

    local_vals = np.full(nx * ny, np.nan, dtype=np.float64)
    points_on_proc = []
    cells_on_proc = []
    ids = []
    for i in range(nx * ny):
        links = colliding.links(i)
        if len(links) > 0:
            points_on_proc.append(pts[i])
            cells_on_proc.append(links[0])
            ids.append(i)

    if points_on_proc:
        vals = uh.eval(np.array(points_on_proc, dtype=np.float64),
                       np.array(cells_on_proc, dtype=np.int32))
        local_vals[np.array(ids, dtype=np.int32)] = np.asarray(vals).reshape(-1)

    gathered = domain.comm.gather(local_vals, root=0)
    if domain.comm.rank == 0:
        out = np.full(nx * ny, np.nan, dtype=np.float64)
        for arr in gathered:
            mask = np.isnan(out) & ~np.isnan(arr)
            out[mask] = arr[mask]
        if np.isnan(out).any():
            raise RuntimeError("Failed to sample all grid points.")
        return out.reshape(ny, nx)
    return None


def solve(case_spec: dict) -> dict:
    case_spec = _with_defaults(case_spec)
    comm = MPI.COMM_WORLD
    rank = comm.rank

    kappa = 0.1
    t0 = float(case_spec["pde"]["time"].get("t0", 0.0))
    t_end = float(case_spec["pde"]["time"].get("t_end", 0.2))
    dt_input = float(case_spec["pde"]["time"].get("dt", 0.02))

    mesh_resolution = 152
    element_degree = 2
    dt = min(dt_input, 0.0025)
    n_steps = max(1, int(round((t_end - t0) / dt)))
    dt = (t_end - t0) / n_steps

    domain = mesh.create_unit_square(comm, mesh_resolution, mesh_resolution, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", element_degree))

    x = ufl.SpatialCoordinate(domain)
    t_c = fem.Constant(domain, ScalarType(t0))
    dt_c = fem.Constant(domain, ScalarType(dt))
    kappa_c = fem.Constant(domain, ScalarType(kappa))

    u_exact = ufl.exp(-0.5 * t_c) * ufl.sin(2.0 * ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])
    f_expr = (-0.5 + 5.0 * (ufl.pi ** 2) * kappa_c) * u_exact

    u_n = fem.Function(V)
    u_n.interpolate(lambda X: _exact_numpy(X, t0))

    u_bc = fem.Function(V)
    u_bc.interpolate(lambda X: _exact_numpy(X, t0))

    fdim = domain.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(domain, fdim, lambda X: np.ones(X.shape[1], dtype=bool))
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc = fem.dirichletbc(u_bc, boundary_dofs)

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    a = (u * v + dt_c * kappa_c * ufl.inner(ufl.grad(u), ufl.grad(v))) * ufl.dx
    L = (u_n * v + dt_c * f_expr * v) * ufl.dx

    a_form = fem.form(a)
    L_form = fem.form(L)

    A = petsc.assemble_matrix(a_form, bcs=[bc])
    A.assemble()
    b = petsc.create_vector(L_form.function_spaces)

    solver = PETSc.KSP().create(comm)
    solver.setOperators(A)
    solver.setType("cg")
    solver.getPC().setType("hypre")
    solver.setTolerances(rtol=1.0e-10, atol=1.0e-12, max_it=5000)
    solver.setFromOptions()

    uh = fem.Function(V)
    iterations = 0

    grid = case_spec["output"]["grid"]
    nx = int(grid["nx"])
    ny = int(grid["ny"])
    xmin, xmax, ymin, ymax = grid["bbox"]
    xs = np.linspace(xmin, xmax, nx)
    ys = np.linspace(ymin, ymax, ny)
    XX, YY = np.meshgrid(xs, ys, indexing="xy")
    u_initial_grid = np.exp(-0.5 * t0) * np.sin(2.0 * np.pi * XX) * np.sin(np.pi * YY) if rank == 0 else None

    for step in range(1, n_steps + 1):
        t_now = t0 + step * dt
        t_c.value = ScalarType(t_now)
        u_bc.interpolate(lambda X, tt=t_now: _exact_numpy(X, tt))

        with b.localForm() as loc:
            loc.set(0.0)
        petsc.assemble_vector(b, L_form)
        petsc.apply_lifting(b, [a_form], bcs=[[bc]])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        petsc.set_bc(b, [bc])

        try:
            solver.solve(b, uh.x.petsc_vec)
        except RuntimeError:
            solver.setType("preonly")
            solver.getPC().setType("lu")
            solver.setOperators(A)
            solver.solve(b, uh.x.petsc_vec)

        uh.x.scatter_forward()
        its = solver.getIterationNumber()
        iterations += max(its, 1 if solver.getType() == "preonly" else its)
        u_n.x.array[:] = uh.x.array
        u_n.x.scatter_forward()

    u_grid = _sample_scalar_function_on_grid(domain, uh, grid)

    err_form = fem.form((uh - u_exact) ** 2 * ufl.dx)
    l2_sq_local = fem.assemble_scalar(err_form)
    l2_error = math.sqrt(comm.allreduce(l2_sq_local, op=MPI.SUM))

    solver_info = {
        "mesh_resolution": int(mesh_resolution),
        "element_degree": int(element_degree),
        "ksp_type": solver.getType(),
        "pc_type": solver.getPC().getType(),
        "rtol": 1.0e-10,
        "iterations": int(iterations),
        "dt": float(dt),
        "n_steps": int(n_steps),
        "time_scheme": "backward_euler",
        "l2_error": float(l2_error),
        "diagnosis": DIAGNOSIS,
        "method": METHOD,
    }

    if rank == 0:
        return {"u": u_grid, "u_initial": u_initial_grid, "solver_info": solver_info}
    return {"u": None, "u_initial": None, "solver_info": solver_info}


if __name__ == "__main__":
    case = {
        "pde": {"time": {"t0": 0.0, "t_end": 0.2, "dt": 0.02}},
        "output": {"grid": {"nx": 64, "ny": 64, "bbox": [0.0, 1.0, 0.0, 1.0]}},
    }
    tstart = time.time()
    result = solve(case)
    wall = time.time() - tstart
    if MPI.COMM_WORLD.rank == 0:
        print(f"L2_ERROR: {result['solver_info']['l2_error']:.12e}")
        print(f"WALL_TIME: {wall:.6f}")
        print(f"U_SHAPE: {result['u'].shape}")
