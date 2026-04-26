import time
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc

from dolfinx import mesh, fem, geometry
from dolfinx.fem.petsc import LinearProblem
import ufl
from basix.ufl import element as basix_element, mixed_element as basix_mixed_element


ScalarType = PETSc.ScalarType


def _build_mesh(comm, n):
    return mesh.create_unit_square(comm, n, n, cell_type=mesh.CellType.triangle)


def _make_spaces(domain, degree_u=2, degree_p=1):
    cell = domain.topology.cell_name()
    gdim = domain.geometry.dim
    vel_el = basix_element("Lagrange", cell, degree_u, shape=(gdim,))
    pre_el = basix_element("Lagrange", cell, degree_p)
    W = fem.functionspace(domain, mixed_element([vel_el, pre_el]))
    V, _ = W.sub(0).collapse()
    Q, _ = W.sub(1).collapse()
    return W, V, Q


def _locate_bcs(domain, W, V, Q):
    fdim = domain.topology.dim - 1

    def on_left(x):
        return np.isclose(x[0], 0.0)

    def on_bottom(x):
        return np.isclose(x[1], 0.0)

    def on_top(x):
        return np.isclose(x[1], 1.0)

    left_facets = mesh.locate_entities_boundary(domain, fdim, on_left)
    bottom_facets = mesh.locate_entities_boundary(domain, fdim, on_bottom)
    top_facets = mesh.locate_entities_boundary(domain, fdim, on_top)

    u_left = fem.Function(V)
    u_left.interpolate(lambda x: np.vstack((4.0 * x[1] * (1.0 - x[1]), np.zeros(x.shape[1], dtype=ScalarType))))

    u_zero = fem.Function(V)
    u_zero.x.array[:] = 0.0

    dofs_left = fem.locate_dofs_topological((W.sub(0), V), fdim, left_facets)
    dofs_bottom = fem.locate_dofs_topological((W.sub(0), V), fdim, bottom_facets)
    dofs_top = fem.locate_dofs_topological((W.sub(0), V), fdim, top_facets)

    bc_left = fem.dirichletbc(u_left, dofs_left, W.sub(0))
    bc_bottom = fem.dirichletbc(u_zero, dofs_bottom, W.sub(0))
    bc_top = fem.dirichletbc(u_zero, dofs_top, W.sub(0))

    # Pressure pinning for unique solvability
    p_dofs = fem.locate_dofs_geometrical(
        (W.sub(1), Q),
        lambda x: np.isclose(x[0], 0.0) & np.isclose(x[1], 0.0),
    )
    p_zero = fem.Function(Q)
    p_zero.x.array[:] = 0.0
    bc_p = fem.dirichletbc(p_zero, p_dofs, W.sub(1))

    return [bc_left, bc_bottom, bc_top, bc_p]


def _solve_stokes(case_spec, mesh_resolution, degree_u=2, degree_p=1, nu=0.05,
                  ksp_type="minres", pc_type="lu", rtol=1e-10):
    comm = MPI.COMM_WORLD
    domain = _build_mesh(comm, mesh_resolution)
    W, V, Q = _make_spaces(domain, degree_u=degree_u, degree_p=degree_p)
    bcs = _locate_bcs(domain, W, V, Q)

    (u, p) = ufl.TrialFunctions(W)
    (v, q) = ufl.TestFunctions(W)

    f = fem.Constant(domain, np.zeros(domain.geometry.dim, dtype=ScalarType))

    a = (
        2.0 * nu * ufl.inner(ufl.sym(ufl.grad(u)), ufl.sym(ufl.grad(v))) * ufl.dx
        - p * ufl.div(v) * ufl.dx
        + ufl.div(u) * q * ufl.dx
    )
    L = ufl.inner(f, v) * ufl.dx

    opts = {
        "ksp_type": ksp_type,
        "pc_type": pc_type,
        "ksp_rtol": rtol,
    }
    if pc_type == "lu":
        opts["ksp_type"] = "preonly"

    problem = LinearProblem(
        a, L, bcs=bcs,
        petsc_options=opts,
        petsc_options_prefix="stokes_",
    )

    t0 = time.perf_counter()
    wh = problem.solve()
    solve_time = time.perf_counter() - t0
    wh.x.scatter_forward()

    uh = wh.sub(0).collapse()
    ph = wh.sub(1).collapse()

    # Try to recover iteration count if available
    iterations = 0
    try:
        solver = problem.solver
        iterations = int(solver.getIterationNumber())
    except Exception:
        iterations = 0

    return domain, uh, ph, solve_time, iterations


def _sample_vector_function_magnitude(u_func, bbox, nx, ny):
    domain = u_func.function_space.mesh

    xs = np.linspace(bbox[0], bbox[1], nx)
    ys = np.linspace(bbox[2], bbox[3], ny)
    XX, YY = np.meshgrid(xs, ys, indexing="xy")
    pts = np.column_stack([XX.ravel(), YY.ravel(), np.zeros(nx * ny, dtype=np.float64)])

    tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(tree, pts)
    colliding = geometry.compute_colliding_cells(domain, cell_candidates, pts)

    local_vals = np.full((pts.shape[0], domain.geometry.dim), np.nan, dtype=np.float64)
    points_on_proc = []
    cells_on_proc = []
    map_ids = []
    for i in range(pts.shape[0]):
        links = colliding.links(i)
        if len(links) > 0:
            points_on_proc.append(pts[i])
            cells_on_proc.append(links[0])
            map_ids.append(i)

    if len(points_on_proc) > 0:
        vals = u_func.eval(np.array(points_on_proc, dtype=np.float64),
                           np.array(cells_on_proc, dtype=np.int32))
        local_vals[np.array(map_ids, dtype=np.int32)] = vals

    comm = domain.comm
    gathered = comm.gather(local_vals, root=0)

    if comm.rank == 0:
        global_vals = np.full_like(local_vals, np.nan)
        for arr in gathered:
            mask = np.isnan(global_vals[:, 0]) & ~np.isnan(arr[:, 0])
            global_vals[mask] = arr[mask]

        if np.isnan(global_vals).any():
            global_vals = np.nan_to_num(global_vals, nan=0.0)

        mag = np.linalg.norm(global_vals, axis=1).reshape(ny, nx)
        return mag
    return None


def _compute_poiseuille_verification(uh):
    domain = uh.function_space.mesh
    comm = domain.comm

    x = ufl.SpatialCoordinate(domain)
    u_exact = ufl.as_vector((4.0 * x[1] * (1.0 - x[1]), 0.0))
    err_form = fem.form(ufl.inner(uh - u_exact, uh - u_exact) * ufl.dx)
    ref_form = fem.form(ufl.inner(u_exact, u_exact) * ufl.dx)

    err_sq = fem.assemble_scalar(err_form)
    ref_sq = fem.assemble_scalar(ref_form)

    err_sq = comm.allreduce(err_sq, op=MPI.SUM)
    ref_sq = comm.allreduce(ref_sq, op=MPI.SUM)

    rel_l2 = np.sqrt(err_sq / ref_sq) if ref_sq > 0 else np.sqrt(err_sq)
    return float(rel_l2)


def solve(case_spec: dict) -> dict:
    pde = case_spec.get("pde", {})
    nu = float(pde.get("nu", case_spec.get("physics", {}).get("viscosity", 0.05)))
    if nu <= 0:
        nu = 0.05

    # Use available time budget for accuracy while staying robust.
    # Since limit is very large, pick a fairly fine mesh directly.
    mesh_resolution = int(case_spec.get("solver", {}).get("mesh_resolution", 96))
    degree_u = 2
    degree_p = 1
    ksp_type = "preonly"
    pc_type = "lu"
    rtol = 1e-10

    domain, uh, ph, wall_time, iterations = _solve_stokes(
        case_spec,
        mesh_resolution=mesh_resolution,
        degree_u=degree_u,
        degree_p=degree_p,
        nu=nu,
        ksp_type=ksp_type,
        pc_type=pc_type,
        rtol=rtol,
    )

    rel_l2 = _compute_poiseuille_verification(uh)

    # Opportunistic refinement if unexpectedly cheap and verification is not as tight as desired.
    if wall_time < 5.0 and rel_l2 > 1.0e-3 and mesh_resolution < 160:
        mesh_resolution = 144
        domain, uh, ph, wall_time2, iterations2 = _solve_stokes(
            case_spec,
            mesh_resolution=mesh_resolution,
            degree_u=degree_u,
            degree_p=degree_p,
            nu=nu,
            ksp_type=ksp_type,
            pc_type=pc_type,
            rtol=rtol,
        )
        wall_time += wall_time2
        iterations += iterations2
        rel_l2 = _compute_poiseuille_verification(uh)

    out = case_spec["output"]["grid"]
    nx = int(out["nx"])
    ny = int(out["ny"])
    bbox = out["bbox"]

    u_grid = _sample_vector_function_magnitude(uh, bbox, nx, ny)

    if domain.comm.rank == 0:
        solver_info = {
            "mesh_resolution": int(mesh_resolution),
            "element_degree": int(degree_u),
            "ksp_type": str(ksp_type),
            "pc_type": str(pc_type),
            "rtol": float(rtol),
            "iterations": int(iterations),
            "verification_relative_l2_poiseuille": float(rel_l2),
        }
        return {"u": u_grid, "solver_info": solver_info}
    else:
        return {"u": None, "solver_info": None}


if __name__ == "__main__":
    case_spec = {
        "pde": {"nu": 0.05, "time": None},
        "output": {
            "grid": {
                "nx": 64,
                "ny": 64,
                "bbox": [0.0, 1.0, 0.0, 1.0],
            }
        },
    }
    result = solve(case_spec)
    if MPI.COMM_WORLD.rank == 0:
        print(result["u"].shape)
        print(result["solver_info"])
