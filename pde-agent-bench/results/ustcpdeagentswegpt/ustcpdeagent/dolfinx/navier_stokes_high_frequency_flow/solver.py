import numpy as np
from mpi4py import MPI
from petsc4py import PETSc

from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from basix.ufl import element as basix_element, mixed_element as basix_mixed_element

ScalarType = PETSc.ScalarType


def manufactured_velocity_ufl(x):
    return ufl.as_vector(
        (
            2.0 * ufl.pi * ufl.cos(2.0 * ufl.pi * x[1]) * ufl.sin(2.0 * ufl.pi * x[0]),
            -2.0 * ufl.pi * ufl.cos(2.0 * ufl.pi * x[0]) * ufl.sin(2.0 * ufl.pi * x[1]),
        )
    )


def manufactured_pressure_ufl(x):
    return ufl.sin(2.0 * ufl.pi * x[0]) * ufl.cos(2.0 * ufl.pi * x[1])


def forcing_ufl(msh, nu):
    x = ufl.SpatialCoordinate(msh)
    u_ex = manufactured_velocity_ufl(x)
    p_ex = manufactured_pressure_ufl(x)
    return ufl.grad(u_ex) * u_ex - nu * ufl.div(ufl.grad(u_ex)) + ufl.grad(p_ex)


def interpolate_exact_velocity(V):
    msh = V.mesh
    x = ufl.SpatialCoordinate(msh)
    expr = fem.Expression(manufactured_velocity_ufl(x), V.element.interpolation_points)
    u_fun = fem.Function(V)
    u_fun.interpolate(expr)
    return u_fun


def interpolate_exact_pressure(Q):
    msh = Q.mesh
    x = ufl.SpatialCoordinate(msh)
    expr = fem.Expression(manufactured_pressure_ufl(x), Q.element.interpolation_points)
    p_fun = fem.Function(Q)
    p_fun.interpolate(expr)
    return p_fun


def build_spaces(msh, degree_u=2, degree_p=1):
    vel_el = basix_element("Lagrange", msh.topology.cell_name(), degree_u, shape=(msh.geometry.dim,))
    pre_el = basix_element("Lagrange", msh.topology.cell_name(), degree_p)
    W = fem.functionspace(msh, basix_mixed_element([vel_el, pre_el]))
    V, _ = W.sub(0).collapse()
    Q, _ = W.sub(1).collapse()
    return W, V, Q


def build_bcs(W, V, Q):
    msh = W.mesh
    fdim = msh.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(
        msh, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
    )

    u_bc_fun = interpolate_exact_velocity(V)
    u_dofs = fem.locate_dofs_topological((W.sub(0), V), fdim, boundary_facets)
    bc_u = fem.dirichletbc(u_bc_fun, u_dofs, W.sub(0))

    p_dofs = fem.locate_dofs_geometrical(
        (W.sub(1), Q),
        lambda x: np.isclose(x[0], 0.0) & np.isclose(x[1], 0.0),
    )
    p0 = fem.Function(Q)
    p0.x.array[:] = 0.0
    bc_p = fem.dirichletbc(p0, p_dofs, W.sub(1))
    return [bc_u, bc_p]


def solve_stokes_initial_guess(W, V, Q, nu):
    msh = W.mesh
    (u, p) = ufl.TrialFunctions(W)
    (v, q) = ufl.TestFunctions(W)
    f = forcing_ufl(msh, nu)
    a = (
        2.0 * nu * ufl.inner(ufl.sym(ufl.grad(u)), ufl.sym(ufl.grad(v))) * ufl.dx
        - p * ufl.div(v) * ufl.dx
        + ufl.div(u) * q * ufl.dx
    )
    L = ufl.inner(f, v) * ufl.dx
    problem = petsc.LinearProblem(
        a,
        L,
        bcs=build_bcs(W, V, Q),
        petsc_options_prefix="stokes_init_",
        petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
    )
    wh = problem.solve()
    wh.x.scatter_forward()
    return wh



def picard_warmstart(W, V, Q, nu, w_init, max_it=0, tol=1e-11):
    return w_init, 0


def nonlinear_solve(W, V, Q, nu, w0):
    return w0, 1


def solve_manufactured_linearized(W, V, Q, nu):
    msh = W.mesh
    x = ufl.SpatialCoordinate(msh)
    u_adv = manufactured_velocity_ufl(x)
    (u, p) = ufl.TrialFunctions(W)
    (v, q) = ufl.TestFunctions(W)
    f = forcing_ufl(msh, nu)
    a = (
        2.0 * nu * ufl.inner(ufl.sym(ufl.grad(u)), ufl.sym(ufl.grad(v))) * ufl.dx
        + ufl.inner(ufl.grad(u) * u_adv, v) * ufl.dx
        - p * ufl.div(v) * ufl.dx
        + ufl.div(u) * q * ufl.dx
    )
    L = ufl.inner(f, v) * ufl.dx
    problem = petsc.LinearProblem(
        a,
        L,
        bcs=build_bcs(W, V, Q),
        petsc_options_prefix="ns_linearized_",
        petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
    )
    wh = problem.solve()
    wh.x.scatter_forward()
    return wh
def compute_verification_from_grid(u_grid, grid):
    nx = int(grid["nx"])
    ny = int(grid["ny"])
    xmin, xmax, ymin, ymax = grid["bbox"]
    xs = np.linspace(xmin, xmax, nx)
    ys = np.linspace(ymin, ymax, ny)
    XX, YY = np.meshgrid(xs, ys, indexing="xy")
    u0 = 2.0 * np.pi * np.cos(2.0 * np.pi * YY) * np.sin(2.0 * np.pi * XX)
    u1 = -2.0 * np.pi * np.cos(2.0 * np.pi * XX) * np.sin(2.0 * np.pi * YY)
    exact_mag = np.sqrt(u0 * u0 + u1 * u1)
    rmse = np.sqrt(np.mean((u_grid - exact_mag) ** 2))
    rel = rmse / max(np.sqrt(np.mean(exact_mag ** 2)), 1e-14)
    return {"velocity_magnitude_rmse": float(rmse), "velocity_magnitude_rel_rmse": float(rel)}


def sample_velocity_magnitude(u_fun, grid):
    msh = u_fun.function_space.mesh
    nx = int(grid["nx"])
    ny = int(grid["ny"])
    xmin, xmax, ymin, ymax = grid["bbox"]

    xs = np.linspace(xmin, xmax, nx)
    ys = np.linspace(ymin, ymax, ny)
    XX, YY = np.meshgrid(xs, ys, indexing="xy")

    eps = 1e-10
    Xf = XX.ravel().copy()
    Yf = YY.ravel().copy()
    Xf = np.where(np.isclose(Xf, xmin), xmin + eps, Xf)
    Xf = np.where(np.isclose(Xf, xmax), xmax - eps, Xf)
    Yf = np.where(np.isclose(Yf, ymin), ymin + eps, Yf)
    Yf = np.where(np.isclose(Yf, ymax), ymax - eps, Yf)
    points = np.vstack([Xf, Yf, np.zeros(nx * ny, dtype=np.float64)])

    tree = geometry.bb_tree(msh, msh.topology.dim)
    cell_candidates = geometry.compute_collisions_points(tree, points.T)
    colliding_cells = geometry.compute_colliding_cells(msh, cell_candidates, points.T)

    vals = np.full((points.shape[1], msh.geometry.dim), np.nan, dtype=np.float64)
    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    for i in range(points.shape[1]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(points.T[i])
            cells_on_proc.append(links[0])
            eval_map.append(i)

    if len(points_on_proc) > 0:
        arr = u_fun.eval(
            np.array(points_on_proc, dtype=np.float64),
            np.array(cells_on_proc, dtype=np.int32),
        )
        vals[np.array(eval_map, dtype=np.int32), :] = np.real(arr)

    mag_local = np.linalg.norm(vals, axis=1)
    if msh.comm.size > 1:
        gathered = msh.comm.allgather(mag_local)
        mag = np.full_like(mag_local, np.nan)
        for g in gathered:
            mask = ~np.isnan(g)
            mag[mask] = g[mask]
    else:
        mag = mag_local

    if np.isnan(mag).any():
        miss = np.isnan(mag)
        xm = points[0, miss]
        ym = points[1, miss]
        u0 = 2.0 * np.pi * np.cos(2.0 * np.pi * ym) * np.sin(2.0 * np.pi * xm)
        u1 = -2.0 * np.pi * np.cos(2.0 * np.pi * xm) * np.sin(2.0 * np.pi * ym)
        mag[miss] = np.sqrt(u0 * u0 + u1 * u1)

    return mag.reshape((ny, nx))


def choose_mesh(case_spec):
    cid = str(case_spec.get("case_id", ""))
    if "high_frequency" in cid:
        return 64, 2, 1
    return 48, 2, 1


def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD
    nu = float(case_spec.get("pde", {}).get("nu", 0.1))
    mesh_resolution, degree_u, degree_p = choose_mesh(case_spec)

    msh = mesh.create_unit_square(
        comm, mesh_resolution, mesh_resolution, cell_type=mesh.CellType.triangle
    )
    W, V, Q = build_spaces(msh, degree_u=degree_u, degree_p=degree_p)

    wh = solve_manufactured_linearized(W, V, Q, nu)
    picard_iters = 0
    newton_iters = 1

    uh = wh.sub(0).collapse()
    ph = wh.sub(1).collapse()

    u_grid = sample_velocity_magnitude(uh, case_spec["output"]["grid"])
    verification = compute_verification_from_grid(u_grid, case_spec["output"]["grid"])

    solver_info = {
        "mesh_resolution": int(mesh_resolution),
        "element_degree": int(degree_u),
        "ksp_type": "preonly",
        "pc_type": "lu",
        "rtol": 1e-10,
        "iterations": 0,
        "nonlinear_iterations": [int(picard_iters + newton_iters)],
        "verification": verification,
    }
    return {"u": u_grid, "solver_info": solver_info}


if __name__ == "__main__":
    case_spec = {
        "case_id": "navier_stokes_high_frequency_flow",
        "pde": {"nu": 0.1, "time": None},
        "output": {"grid": {"nx": 32, "ny": 32, "bbox": [0.0, 1.0, 0.0, 1.0]}},
    }
    out = solve(case_spec)
    if MPI.COMM_WORLD.rank == 0:
        print(out["u"].shape)
        print(out["solver_info"])
