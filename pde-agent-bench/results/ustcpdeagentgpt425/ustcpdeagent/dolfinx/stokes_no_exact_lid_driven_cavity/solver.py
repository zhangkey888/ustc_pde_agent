import time
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
import ufl
from basix.ufl import element as basix_element, mixed_element as basix_mixed_element
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc


ScalarType = PETSc.ScalarType


def _build_markers():
    return {
        "x0": lambda x: np.isclose(x[0], 0.0),
        "x1": lambda x: np.isclose(x[0], 1.0),
        "y0": lambda x: np.isclose(x[1], 0.0),
        "y1": lambda x: np.isclose(x[1], 1.0),
    }


def _make_velocity_bc_func(V, value):
    gdim = V.mesh.geometry.dim
    f = fem.Function(V)
    val = np.array(value, dtype=np.float64)

    def bc_expr(x):
        return np.vstack([np.full(x.shape[1], val[i], dtype=np.float64) for i in range(gdim)])

    f.interpolate(bc_expr)
    return f


def _sample_vector_function(func, pts3):
    msh = func.function_space.mesh
    gdim = msh.geometry.dim
    Vs = fem.functionspace(msh, ("Lagrange", 1, (gdim,)))
    us = fem.Function(Vs)
    us.interpolate(func)

    tree = geometry.bb_tree(msh, msh.topology.dim)
    pts3 = np.ascontiguousarray(pts3, dtype=np.float64)
    candidates = geometry.compute_collisions_points(tree, pts3)
    colliding = geometry.compute_colliding_cells(msh, candidates, pts3)

    npts = pts3.shape[0]
    values = np.full((npts, gdim), np.nan, dtype=np.float64)
    points_on_proc = []
    cells_on_proc = []
    eval_ids = []
    for i in range(npts):
        links = colliding.links(i)
        if len(links) > 0:
            points_on_proc.append(pts3[i])
            cells_on_proc.append(links[0])
            eval_ids.append(i)

    if points_on_proc:
        vals = us.eval(np.ascontiguousarray(points_on_proc, dtype=np.float64),
                       np.asarray(cells_on_proc, dtype=np.int32))
        values[np.asarray(eval_ids, dtype=np.int32)] = np.asarray(vals, dtype=np.float64)

    return values


def _sample_magnitude(func, nx, ny, bbox):
    eps = 1.0e-10
    xs = np.linspace(bbox[0] + eps, bbox[1] - eps, nx, dtype=np.float64)
    ys = np.linspace(bbox[2] + eps, bbox[3] - eps, ny, dtype=np.float64)
    XX, YY = np.meshgrid(xs, ys, indexing="xy")
    pts3 = np.column_stack([XX.ravel(), YY.ravel(), np.zeros(nx * ny, dtype=np.float64)])
    vals = _sample_vector_function(func, pts3)
    mag = np.linalg.norm(vals, axis=1)
    return mag.reshape(ny, nx)


def _solve_stokes(case_spec, n):
    comm = MPI.COMM_WORLD
    msh = mesh.create_unit_square(comm, n, n, cell_type=mesh.CellType.triangle)
    gdim = msh.geometry.dim

    vel_el = basix_element("Lagrange", msh.topology.cell_name(), 2, shape=(gdim,))
    pres_el = basix_element("Lagrange", msh.topology.cell_name(), 1)
    W = fem.functionspace(msh, basix_mixed_element([vel_el, pres_el]))

    V, _ = W.sub(0).collapse()
    Q, _ = W.sub(1).collapse()

    nu = float(case_spec.get("pde", {}).get("nu", case_spec.get("pde", {}).get("viscosity", 0.2)))
    fvec = case_spec.get("pde", {}).get("source", ['0.0', '0.0'])
    f_vals = tuple(ScalarType(float(v)) for v in fvec)
    f_expr = fem.Constant(msh, f_vals)

    (u, p) = ufl.TrialFunctions(W)
    (v, q) = ufl.TestFunctions(W)

    a = (
        ScalarType(nu) * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
        - ufl.inner(p, ufl.div(v)) * ufl.dx
        + ufl.inner(ufl.div(u), q) * ufl.dx
    )
    L = ufl.inner(f_expr, v) * ufl.dx

    bcs = []
    markers = _build_markers()
    fdim = msh.topology.dim - 1

    bc_specs = case_spec.get("boundary_conditions", {}).get("dirichlet", [])
    if not bc_specs:
        bc_specs = [
            {"boundary": "y1", "value": [1.0, 0.0]},
            {"boundary": "y0", "value": [0.0, 0.0]},
            {"boundary": "x0", "value": [0.0, 0.0]},
            {"boundary": "x1", "value": [0.0, 0.0]},
        ]

    for bc_spec in bc_specs:
        name = bc_spec["boundary"]
        val = bc_spec["value"]
        facets = mesh.locate_entities_boundary(msh, fdim, markers[name])
        dofs = fem.locate_dofs_topological((W.sub(0), V), fdim, facets)
        gfun = _make_velocity_bc_func(V, val)
        bcs.append(fem.dirichletbc(gfun, dofs, W.sub(0)))

    p_dofs = fem.locate_dofs_geometrical((W.sub(1), Q), lambda x: np.isclose(x[0], 0.0) & np.isclose(x[1], 0.0))
    if len(p_dofs) > 0:
        p0 = fem.Function(Q)
        p0.x.array[:] = 0.0
        bcs.append(fem.dirichletbc(p0, p_dofs, W.sub(1)))

    petsc_options = {
        "ksp_type": "preonly",
        "pc_type": "lu",
        "pc_factor_mat_solver_type": "mumps",
    }

    t0 = time.perf_counter()
    iterations = -1
    problem = petsc.LinearProblem(
        a, L, bcs=bcs, petsc_options_prefix=f"stokes_{n}_", petsc_options=petsc_options
    )
    wh = problem.solve()
    try:
        iterations = int(problem.solver.getIterationNumber())
    except Exception:
        iterations = 1
    solve_time = time.perf_counter() - t0

    uh = wh.sub(0).collapse()
    ph = wh.sub(1).collapse()
    return {
        "mesh": msh,
        "W": W,
        "V": V,
        "Q": Q,
        "u": uh,
        "p": ph,
        "solve_time": solve_time,
        "iterations": iterations,
        "ksp_type": "preonly",
        "pc_type": "lu",
        "rtol": 1e-9,
        "mesh_resolution": n,
        "element_degree": 2,
    }


def solve(case_spec: dict) -> dict:
    output = case_spec.get("output", {}).get("grid", {})
    nx = int(output.get("nx", 64))
    ny = int(output.get("ny", 64))
    bbox = output.get("bbox", [0.0, 1.0, 0.0, 1.0])

    time_budget = float(case_spec.get("time_limit", case_spec.get("wall_time_sec", 60.0)))
    candidates = [24, 32, 40, 48, 56, 64]
    if time_budget > 300:
        candidates += [72, 80]

    result = None
    verification = {}
    coarse_grid = None

    for n in candidates:
        current = _solve_stokes(case_spec, n)
        grid = _sample_magnitude(current["u"], nx, ny, bbox)

        if result is None:
            result = current
            coarse_grid = grid
            if current["solve_time"] > 0.35 * time_budget:
                break
            continue

        diff = float(np.sqrt(np.mean((grid - coarse_grid) ** 2)))
        verification = {
            "mesh_compare_l2": diff,
            "coarse_n": int(result["mesh_resolution"]),
            "fine_n": int(current["mesh_resolution"]),
        }
        result = current
        coarse_grid = grid

        if current["solve_time"] > 0.35 * time_budget:
            break
        if diff < 2.0e-3 and current["solve_time"] > 0.05 * time_budget:
            break

    u_grid = np.nan_to_num(coarse_grid, nan=0.0, posinf=0.0, neginf=0.0)
    solver_info = {
        "mesh_resolution": int(result["mesh_resolution"]),
        "element_degree": int(result["element_degree"]),
        "ksp_type": result["ksp_type"],
        "pc_type": result["pc_type"],
        "rtol": float(result["rtol"]),
        "iterations": int(max(result["iterations"], 0)),
        "verification": verification,
    }
    return {"u": u_grid, "solver_info": solver_info}
