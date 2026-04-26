import numpy as np
from mpi4py import MPI
from petsc4py import PETSc

from dolfinx import fem, mesh, geometry
from dolfinx.fem import petsc
import ufl
from basix.ufl import element as basix_element, mixed_element as basix_mixed_element

DIAGNOSIS = "stokes steady 2D mixed saddle-point Taylor-Hood with pressure pinning"
METHOD = "fem Taylor-Hood_P2P1 minres hypre pressure_pinning"


def _build_mixed_space(msh, degree_u=2, degree_p=1):
    gdim = msh.geometry.dim
    cell = msh.topology.cell_name()
    vel_el = basix_element("Lagrange", cell, degree_u, shape=(gdim,))
    pres_el = basix_element("Lagrange", cell, degree_p)
    W = fem.functionspace(msh, basix_mixed_element([vel_el, pres_el]))
    V, _ = W.sub(0).collapse()
    Q, _ = W.sub(1).collapse()
    return W, V, Q


def _inflow_profile(x):
    vals = np.zeros((2, x.shape[1]), dtype=np.float64)
    y = x[1]
    vals[0] = 4.0 * y * (1.0 - y)
    vals[1] = 0.0
    return vals


def _zero_vec(x):
    return np.zeros((2, x.shape[1]), dtype=np.float64)


def _sample_velocity_magnitude(u_fun, grid):
    msh = u_fun.function_space.mesh
    comm = msh.comm

    nx = int(grid["nx"])
    ny = int(grid["ny"])
    xmin, xmax, ymin, ymax = [float(v) for v in grid["bbox"]]

    xs = np.linspace(xmin, xmax, nx)
    ys = np.linspace(ymin, ymax, ny)
    XX, YY = np.meshgrid(xs, ys, indexing="xy")
    pts = np.column_stack([XX.ravel(), YY.ravel(), np.zeros(nx * ny, dtype=np.float64)])

    tree = geometry.bb_tree(msh, msh.topology.dim)
    candidates = geometry.compute_collisions_points(tree, pts)
    colliding = geometry.compute_colliding_cells(msh, candidates, pts)

    found_ids = []
    eval_points = []
    eval_cells = []
    for i in range(pts.shape[0]):
        links = colliding.links(i)
        if len(links) > 0:
            found_ids.append(i)
            eval_points.append(pts[i])
            eval_cells.append(links[0])

    local_ids = np.array(found_ids, dtype=np.int32)
    local_vals = None
    if len(eval_points) > 0:
        local_vals = u_fun.eval(np.array(eval_points, dtype=np.float64), np.array(eval_cells, dtype=np.int32))

    gathered_ids = comm.gather(local_ids, root=0)
    gathered_vals = comm.gather(local_vals, root=0)

    if comm.rank != 0:
        return None

    mag = np.full(nx * ny, np.nan, dtype=np.float64)
    for ids, vals in zip(gathered_ids, gathered_vals):
        if vals is None or len(ids) == 0:
            continue
        mag[ids] = np.linalg.norm(np.asarray(vals, dtype=np.float64), axis=1)

    if np.any(np.isnan(mag)):
        for k in np.where(np.isnan(mag))[0]:
            x = pts[k, 0]
            y = pts[k, 1]
            if np.isclose(x, 0.0):
                mag[k] = abs(4.0 * y * (1.0 - y))
            elif np.isclose(y, 0.0) or np.isclose(y, 1.0):
                mag[k] = 0.0
            else:
                mag[k] = 0.0

    return mag.reshape(ny, nx)


def _compute_verification(u_fun):
    msh = u_fun.function_space.mesh
    comm = msh.comm

    div_sq = fem.assemble_scalar(fem.form((ufl.div(u_fun) ** 2) * ufl.dx))
    div_sq = comm.allreduce(div_sq, op=MPI.SUM)

    x = ufl.SpatialCoordinate(msh)
    indicator_left = ufl.conditional(ufl.le(x[0], 1.0e-12), 1.0, 0.0)
    n = ufl.FacetNormal(msh)
    left_flux = fem.assemble_scalar(fem.form(indicator_left * ufl.dot(u_fun, n) * ufl.ds))
    left_flux = comm.allreduce(left_flux, op=MPI.SUM)

    return {
        "divergence_l2": float(np.sqrt(max(div_sq, 0.0))),
        "left_boundary_flux": float(left_flux),
        "target_inflow_flux": float(-2.0 / 3.0),
        "flux_mismatch": float(abs(left_flux + 2.0 / 3.0)),
    }


def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD

    mesh_resolution = int(case_spec.get("solver", {}).get("mesh_resolution", 96))
    degree_u = 2
    degree_p = 1

    msh = mesh.create_unit_square(comm, mesh_resolution, mesh_resolution, cell_type=mesh.CellType.triangle)
    W, V, Q = _build_mixed_space(msh, degree_u=degree_u, degree_p=degree_p)

    pde = case_spec.get("pde", {})
    nu_val = float(pde.get("nu", 1.0)) if isinstance(pde, dict) else 1.0
    nu = fem.Constant(msh, PETSc.ScalarType(nu_val))
    f = fem.Constant(msh, PETSc.ScalarType((0.0, 0.0)))

    (u, p) = ufl.TrialFunctions(W)
    (v, q) = ufl.TestFunctions(W)

    def eps(w):
        return ufl.sym(ufl.grad(w))

    a = (
        2.0 * nu * ufl.inner(eps(u), eps(v)) * ufl.dx
        - p * ufl.div(v) * ufl.dx
        + ufl.div(u) * q * ufl.dx
    )
    L = ufl.inner(f, v) * ufl.dx

    fdim = msh.topology.dim - 1
    left_facets = mesh.locate_entities_boundary(msh, fdim, lambda x: np.isclose(x[0], 0.0))
    bottom_facets = mesh.locate_entities_boundary(msh, fdim, lambda x: np.isclose(x[1], 0.0))
    top_facets = mesh.locate_entities_boundary(msh, fdim, lambda x: np.isclose(x[1], 1.0))

    u_in = fem.Function(V)
    u_in.interpolate(_inflow_profile)
    u_zero = fem.Function(V)
    u_zero.interpolate(_zero_vec)

    left_dofs = fem.locate_dofs_topological((W.sub(0), V), fdim, left_facets)
    bottom_dofs = fem.locate_dofs_topological((W.sub(0), V), fdim, bottom_facets)
    top_dofs = fem.locate_dofs_topological((W.sub(0), V), fdim, top_facets)

    bc_left = fem.dirichletbc(u_in, left_dofs, W.sub(0))
    bc_bottom = fem.dirichletbc(u_zero, bottom_dofs, W.sub(0))
    bc_top = fem.dirichletbc(u_zero, top_dofs, W.sub(0))

    p_dofs = fem.locate_dofs_geometrical(
        (W.sub(1), Q),
        lambda x: np.isclose(x[0], 0.0) & np.isclose(x[1], 0.0),
    )
    p_zero = fem.Function(Q)
    p_zero.x.array[:] = 0.0
    bc_p = fem.dirichletbc(p_zero, p_dofs, W.sub(1))

    problem = petsc.LinearProblem(
        a,
        L,
        bcs=[bc_left, bc_bottom, bc_top, bc_p],
        petsc_options_prefix="stokes_",
        petsc_options={
            "ksp_type": "minres",
            "ksp_rtol": 1e-9,
            "pc_type": "hypre",
        },
    )
    wh = problem.solve()
    wh.x.scatter_forward()

    u_h = wh.sub(0).collapse()

    u_grid = _sample_velocity_magnitude(u_h, case_spec["output"]["grid"])
    verification = _compute_verification(u_h)

    if comm.rank == 0:
        ksp = problem.solver
        tol_info = ksp.getTolerances()
        rtol = float(tol_info[0]) if tol_info[0] is not None else 1e-9
        return {
            "u": u_grid,
            "solver_info": {
                "mesh_resolution": mesh_resolution,
                "element_degree": degree_u,
                "ksp_type": ksp.getType(),
                "pc_type": ksp.getPC().getType(),
                "rtol": rtol,
                "iterations": int(ksp.getIterationNumber()),
                "verification": verification,
            },
        }
    return {"u": None, "solver_info": None}
