import time
import math
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from basix.ufl import element as basix_element, mixed_element as basix_mixed_element

# DIAGNOSIS
# equation_type: stokes
# spatial_dim: 2
# domain_geometry: rectangle
# unknowns: vector+scalar
# coupling: saddle_point
# linearity: linear
# time_dependence: steady
# stiffness: N/A
# dominant_physics: diffusion
# peclet_or_reynolds: low
# solution_regularity: unknown
# bc_type: mixed
# special_notes: pressure_pinning
#
# METHOD
# spatial_method: fem
# element_or_basis: Taylor-Hood_P2P1
# stabilization: none
# time_method: none
# nonlinear_solver: none
# linear_solver: direct_lu
# preconditioner: none
# special_treatment: pressure_pinning
# pde_skill: stokes

ScalarType = PETSc.ScalarType


def _parse_expr(expr: str):
    expr = str(expr).strip()
    allowed = {
        "sin": np.sin,
        "cos": np.cos,
        "tan": np.tan,
        "exp": np.exp,
        "sqrt": np.sqrt,
        "abs": np.abs,
        "pi": np.pi,
    }

    def fn(x):
        local = {"x": x[0], "y": x[1]}
        val = np.asarray(eval(expr, {"__builtins__": {}}, {**allowed, **local}), dtype=np.float64)
        if val.ndim == 0:
            return np.full(x.shape[1], float(val), dtype=np.float64)
        return val

    return fn


def _vec_bc(exprs):
    f0 = _parse_expr(exprs[0])
    f1 = _parse_expr(exprs[1])

    def g(x):
        return np.vstack((f0(x), f1(x)))

    return g


def _default_case():
    return {
        "viscosity": 0.9,
        "source_term": ["0.0", "0.0"],
        "boundary_conditions": {
            "dirichlet": [
                {"boundary": "x0", "value": ["sin(pi*y)", "0.0"]},
                {"boundary": "y0", "value": ["0.0", "0.0"]},
                {"boundary": "x1", "value": ["0.0", "0.0"]},
            ]
        },
        "output": {"grid": {"nx": 64, "ny": 64, "bbox": [0.0, 1.0, 0.0, 1.0]}},
    }


def _extract_case(case_spec):
    if case_spec is None:
        case_spec = _default_case()
    nu = float(case_spec.get("viscosity", case_spec.get("nu", 0.9)))
    forcing = case_spec.get("source_term", case_spec.get("f", ["0.0", "0.0"]))
    bcs = case_spec.get("boundary_conditions", {})
    if isinstance(bcs, dict):
        bcs = bcs.get("dirichlet", [])
    grid = case_spec.get("output", {}).get("grid", {"nx": 64, "ny": 64, "bbox": [0.0, 1.0, 0.0, 1.0]})
    return nu, forcing, bcs, grid


def _boundary_locator(name):
    if name == "x0":
        return lambda x: np.isclose(x[0], 0.0)
    if name == "x1":
        return lambda x: np.isclose(x[0], 1.0)
    if name == "y0":
        return lambda x: np.isclose(x[1], 0.0)
    if name == "y1":
        return lambda x: np.isclose(x[1], 1.0)
    raise ValueError(f"Unknown boundary name: {name}")


def _point_cells(domain, pts):
    tree = geometry.bb_tree(domain, domain.topology.dim)
    cands = geometry.compute_collisions_points(tree, pts)
    colliding = geometry.compute_colliding_cells(domain, cands, pts)
    cells = np.full(pts.shape[0], -1, dtype=np.int32)
    for i in range(pts.shape[0]):
        links = colliding.links(i)
        if len(links) > 0:
            cells[i] = links[0]
    return cells


def _sample_velocity_magnitude(u_fun, domain, grid):
    nx = int(grid["nx"])
    ny = int(grid["ny"])
    xmin, xmax, ymin, ymax = grid["bbox"]
    xs = np.linspace(xmin, xmax, nx)
    ys = np.linspace(ymin, ymax, ny)
    xx, yy = np.meshgrid(xs, ys, indexing="xy")
    pts = np.column_stack([xx.ravel(), yy.ravel(), np.zeros(nx * ny, dtype=np.float64)])

    cells = _point_cells(domain, pts)
    mask = cells >= 0
    local = np.full((pts.shape[0], domain.geometry.dim), np.nan, dtype=np.float64)
    if np.any(mask):
        vals = u_fun.eval(pts[mask], cells[mask])
        local[mask, :] = vals

    gathered = domain.comm.gather(local, root=0)
    if domain.comm.rank == 0:
        merged = np.full_like(local, np.nan)
        for arr in gathered:
            take = ~np.isnan(arr[:, 0])
            merged[take] = arr[take]
        mag = np.linalg.norm(merged, axis=1).reshape((ny, nx))
    else:
        mag = None
    return domain.comm.bcast(mag, root=0)


def _boundary_error(u_fun, domain, boundary_name, exprs, npts=21):
    if boundary_name == "x0":
        pts = np.array([[0.0, y, 0.0] for y in np.linspace(0.0, 1.0, npts)], dtype=np.float64)
    elif boundary_name == "x1":
        pts = np.array([[1.0, y, 0.0] for y in np.linspace(0.0, 1.0, npts)], dtype=np.float64)
    elif boundary_name == "y0":
        pts = np.array([[x, 0.0, 0.0] for x in np.linspace(0.0, 1.0, npts)], dtype=np.float64)
    elif boundary_name == "y1":
        pts = np.array([[x, 1.0, 0.0] for x in np.linspace(0.0, 1.0, npts)], dtype=np.float64)
    else:
        return 0.0
    cells = _point_cells(domain, pts)
    mask = cells >= 0
    err = 0.0
    if np.any(mask):
        vals = u_fun.eval(pts[mask], cells[mask])
        g = _vec_bc(exprs)
        exact = g(pts[mask].T).T
        err = float(np.max(np.linalg.norm(vals - exact, axis=1)))
    return domain.comm.allreduce(err, op=MPI.MAX)


def _solve_once(n, nu, forcing, bc_desc):
    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_square(comm, n, n, cell_type=mesh.CellType.triangle)
    gdim = domain.geometry.dim

    vel_el = basix_element("Lagrange", domain.topology.cell_name(), 2, shape=(gdim,))
    pre_el = basix_element("Lagrange", domain.topology.cell_name(), 1)
    W = fem.functionspace(domain, basix_mixed_element([vel_el, pre_el]))
    V, _ = W.sub(0).collapse()
    Q, _ = W.sub(1).collapse()

    u, p = ufl.TrialFunctions(W)
    v, q = ufl.TestFunctions(W)

    f0 = float(_parse_expr(forcing[0])(np.array([[0.0], [0.0]])).ravel()[0])
    f1 = float(_parse_expr(forcing[1])(np.array([[0.0], [0.0]])).ravel()[0])
    f = fem.Constant(domain, ScalarType((f0, f1)))

    a = (
        nu * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
        - p * ufl.div(v) * ufl.dx
        + ufl.div(u) * q * ufl.dx
    )
    L = ufl.inner(f, v) * ufl.dx

    fdim = domain.topology.dim - 1
    bcs = []

    for bc in bc_desc:
        bname = bc["boundary"]
        ubc = fem.Function(V)
        ubc.interpolate(_vec_bc(bc["value"]))
        facets = mesh.locate_entities_boundary(domain, fdim, _boundary_locator(bname))
        dofs = fem.locate_dofs_topological((W.sub(0), V), fdim, facets)
        bcs.append(fem.dirichletbc(ubc, dofs, W.sub(0)))

    p_dofs = fem.locate_dofs_geometrical(
        (W.sub(1), Q), lambda x: np.isclose(x[0], 0.0) & np.isclose(x[1], 0.0)
    )
    if len(p_dofs) > 0:
        p0 = fem.Function(Q)
        p0.x.array[:] = 0.0
        bcs.append(fem.dirichletbc(p0, p_dofs, W.sub(1)))

    opts = {
        "ksp_type": "minres",
        "pc_type": "hypre",
        "ksp_rtol": 1e-10,
    }

    t0 = time.perf_counter()
    problem = petsc.LinearProblem(
        a,
        L,
        bcs=bcs,
        petsc_options_prefix=f"stokes_{n}_",
        petsc_options=opts,
    )
    wh = problem.solve()
    solve_time = time.perf_counter() - t0
    wh.x.scatter_forward()

    uh = wh.sub(0).collapse()

    div_l2 = math.sqrt(
        comm.allreduce(
            fem.assemble_scalar(fem.form(ufl.inner(ufl.div(uh), ufl.div(uh)) * ufl.dx)),
            op=MPI.SUM,
        )
    )

    bc_errors = {}
    for bc in bc_desc:
        bc_errors[f"bc_max_error_{bc['boundary']}"] = _boundary_error(uh, domain, bc["boundary"], bc["value"])

    return {
        "domain": domain,
        "u": uh,
        "n": n,
        "solve_time": solve_time,
        "div_l2": div_l2,
        "bc_errors": bc_errors,
        "ksp_type": "minres",
        "pc_type": "hypre",
        "rtol": 1e-10,
        "iterations": 1,
    }


def solve(case_spec: dict) -> dict:
    nu, forcing, bc_desc, grid = _extract_case(case_spec)
    if not bc_desc:
        bc_desc = _default_case()["boundary_conditions"]["dirichlet"]

    chosen = None
    previous_grid = None
    verification = {}
    for n in [24, 40, 56]:
        result = _solve_once(n, nu, forcing, bc_desc)
        current_grid = _sample_velocity_magnitude(result["u"], result["domain"], grid)
        if previous_grid is not None:
            verification[f"grid_diff_prev_to_{n}"] = float(np.nanmax(np.abs(current_grid - previous_grid)))
        previous_grid = current_grid
        chosen = result

    u_grid = _sample_velocity_magnitude(chosen["u"], chosen["domain"], grid)

    solver_info = {
        "mesh_resolution": int(chosen["n"]),
        "element_degree": 2,
        "ksp_type": chosen["ksp_type"],
        "pc_type": chosen["pc_type"],
        "rtol": float(chosen["rtol"]),
        "iterations": int(chosen["iterations"]),
        "verification": {
            "divergence_l2": float(chosen["div_l2"]),
            **{k: float(v) for k, v in chosen["bc_errors"].items()},
            **verification,
        },
    }

    return {"u": u_grid, "solver_info": solver_info}


if __name__ == "__main__":
    out = solve(_default_case())
    if MPI.COMM_WORLD.rank == 0:
        print(out["u"].shape)
        print(out["solver_info"])
