from mpi4py import MPI
import numpy as np
from petsc4py import PETSc
import ufl

from dolfinx import fem, mesh, geometry
from dolfinx.fem import petsc as fem_petsc
from basix.ufl import element as basix_element, mixed_element as basix_mixed_element

ScalarType = PETSc.ScalarType


def _find_value(d, keys, default=None):
    if not isinstance(d, dict):
        return default
    for k in keys:
        if k in d:
            return d[k]
    return default


def _parse_force(case_spec):
    candidates = [
        _find_value(case_spec.get("pde", {}), ["source_term", "source", "f"]),
        _find_value(case_spec, ["source_term", "Source Term", "f"]),
    ]
    for force in candidates:
        if force is not None:
            return float(force[0]), float(force[1])
    return 1.0, 0.0


def _parse_viscosity(case_spec):
    pde = case_spec.get("pde", {})
    for key in ["nu", "viscosity", "mu"]:
        if key in pde:
            return float(pde[key])
    for key in ["viscosity", "nu", "Viscosity"]:
        if key in case_spec:
            return float(case_spec[key])
    return 0.4


def _get_output_grid(case_spec):
    grid = case_spec["output"]["grid"]
    nx = int(grid["nx"])
    ny = int(grid["ny"])
    bbox = grid["bbox"]
    return nx, ny, bbox


def _pick_mesh_resolution(nx_out, ny_out):
    target = max(nx_out, ny_out)
    if target <= 40:
        return 48
    if target <= 64:
        return 64
    if target <= 96:
        return 80
    if target <= 128:
        return 96
    if target <= 192:
        return 128
    return 160


def _build_spaces(domain):
    gdim = domain.geometry.dim
    cell_name = domain.topology.cell_name()
    vel_el = basix_element("Lagrange", cell_name, 2, shape=(gdim,))
    pres_el = basix_element("Lagrange", cell_name, 1)
    W = fem.functionspace(domain, basix_mixed_element([vel_el, pres_el]))
    V, _ = W.sub(0).collapse()
    Q, _ = W.sub(1).collapse()
    return W, V, Q


def _velocity_bcs(domain, W, V):
    fdim = domain.topology.dim - 1
    zero_u = fem.Function(V)
    zero_u.x.array[:] = 0.0
    bcs = []

    def add_bc(marker):
        facets = mesh.locate_entities_boundary(domain, fdim, marker)
        if len(facets) > 0:
            dofs = fem.locate_dofs_topological((W.sub(0), V), fdim, facets)
            bcs.append(fem.dirichletbc(zero_u, dofs, W.sub(0)))

    add_bc(lambda x: np.isclose(x[0], 0.0))
    add_bc(lambda x: np.isclose(x[1], 0.0))
    add_bc(lambda x: np.isclose(x[1], 1.0))
    return bcs


def _pressure_pin(W, Q, bcs):
    p_dofs = fem.locate_dofs_geometrical(
        (W.sub(1), Q), lambda x: np.isclose(x[0], 1.0) & np.isclose(x[1], 0.0)
    )
    if len(p_dofs) == 0:
        p_dofs = fem.locate_dofs_geometrical(
            (W.sub(1), Q), lambda x: np.isclose(x[0], 0.0) & np.isclose(x[1], 0.0)
        )
    if len(p_dofs) > 0:
        p0 = fem.Function(Q)
        p0.x.array[:] = 0.0
        bcs.append(fem.dirichletbc(p0, p_dofs, W.sub(1)))
    return bcs


def _solve_once(n, nu, fx, fy):
    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_square(comm, n, n, cell_type=mesh.CellType.triangle)
    W, V, Q = _build_spaces(domain)

    bcs = _velocity_bcs(domain, W, V)
    bcs = _pressure_pin(W, Q, bcs)

    u, p = ufl.TrialFunctions(W)
    v, q = ufl.TestFunctions(W)

    f = fem.Constant(domain, np.array([fx, fy], dtype=ScalarType))
    nu_c = fem.Constant(domain, ScalarType(nu))

    a = (
        2.0 * nu_c * ufl.inner(ufl.sym(ufl.grad(u)), ufl.sym(ufl.grad(v))) * ufl.dx
        - p * ufl.div(v) * ufl.dx
        + q * ufl.div(u) * ufl.dx
    )
    L = ufl.inner(f, v) * ufl.dx

    ksp_type = "minres"
    pc_type = "lu"
    rtol = 1.0e-10

    try:
        problem = fem_petsc.LinearProblem(
            a,
            L,
            bcs=bcs,
            petsc_options={
                "ksp_type": ksp_type,
                "pc_type": pc_type,
                "ksp_rtol": rtol,
            },
            petsc_options_prefix="stokes_",
        )
        wh = problem.solve()
        wh.x.scatter_forward()
        solver = problem.solver
    except Exception:
        ksp_type = "preonly"
        pc_type = "lu"
        rtol = 1.0e-12
        problem = fem_petsc.LinearProblem(
            a,
            L,
            bcs=bcs,
            petsc_options={
                "ksp_type": ksp_type,
                "pc_type": pc_type,
                "ksp_rtol": rtol,
            },
            petsc_options_prefix="stokes_fallback_",
        )
        wh = problem.solve()
        wh.x.scatter_forward()
        solver = problem.solver

    uh, _ = wh.sub(0).collapse()
    ph, _ = wh.sub(1).collapse()

    div_l2 = _divergence_l2(domain, uh)
    energy = _kinetic_energy(domain, uh)

    return uh, ph, {
        "mesh_resolution": int(n),
        "element_degree": 2,
        "ksp_type": str(solver.getType()),
        "pc_type": str(solver.getPC().getType()),
        "rtol": float(solver.getTolerances()[0]),
        "iterations": int(solver.getIterationNumber()),
        "divergence_l2": float(div_l2),
        "kinetic_energy": float(energy),
    }


def _divergence_l2(domain, uh):
    val = fem.assemble_scalar(fem.form((ufl.div(uh) ** 2) * ufl.dx))
    val = domain.comm.allreduce(val, op=MPI.SUM)
    return float(np.sqrt(max(val, 0.0)))


def _kinetic_energy(domain, uh):
    val = fem.assemble_scalar(fem.form(ufl.inner(uh, uh) * ufl.dx))
    val = domain.comm.allreduce(val, op=MPI.SUM)
    return float(np.sqrt(max(val, 0.0)))


def _sample_velocity_magnitude(uh, bbox, nx, ny):
    domain = uh.function_space.mesh
    xs = np.linspace(bbox[0], bbox[1], nx)
    ys = np.linspace(bbox[2], bbox[3], ny)
    X, Y = np.meshgrid(xs, ys, indexing="xy")
    pts = np.column_stack(
        [X.ravel(), Y.ravel(), np.zeros(nx * ny, dtype=np.float64)]
    )

    tree = geometry.bb_tree(domain, domain.topology.dim)
    candidates = geometry.compute_collisions_points(tree, pts)
    colliding = geometry.compute_colliding_cells(domain, candidates, pts)

    local_ids = []
    local_points = []
    local_cells = []
    for i in range(pts.shape[0]):
        links = colliding.links(i)
        if len(links) > 0:
            local_ids.append(i)
            local_points.append(pts[i])
            local_cells.append(links[0])

    mags_local = np.zeros(pts.shape[0], dtype=np.float64)
    owns_local = np.zeros(pts.shape[0], dtype=np.int32)

    if local_points:
        vals = uh.eval(
            np.asarray(local_points, dtype=np.float64),
            np.asarray(local_cells, dtype=np.int32),
        )
        vals = np.asarray(vals)
        if vals.ndim == 1:
            vals = vals.reshape(-1, domain.geometry.dim)
        mags_local[np.asarray(local_ids, dtype=np.int32)] = np.linalg.norm(vals, axis=1)
        owns_local[np.asarray(local_ids, dtype=np.int32)] = 1

    mags = np.zeros_like(mags_local)
    owns = np.zeros_like(owns_local)
    domain.comm.Allreduce(mags_local, mags, op=MPI.SUM)
    domain.comm.Allreduce(owns_local, owns, op=MPI.SUM)

    if np.any(owns == 0):
        mags[owns == 0] = 0.0

    return mags.reshape(ny, nx)


def solve(case_spec: dict) -> dict:
    nx_out, ny_out, bbox = _get_output_grid(case_spec)
    fx, fy = _parse_force(case_spec)
    nu = _parse_viscosity(case_spec)

    n = _pick_mesh_resolution(nx_out, ny_out)
    uh, ph, solver_info = _solve_once(n, nu, fx, fy)

    # Accuracy verification and adaptive refinement:
    # for incompressible Stokes without exact solution, monitor ||div u||_L2 and refine if needed.
    if solver_info["divergence_l2"] > 1.0e-8 and n < 192:
        n_refined = min(2 * n, 192)
        uh2, ph2, solver_info2 = _solve_once(n_refined, nu, fx, fy)
        if solver_info2["divergence_l2"] <= solver_info["divergence_l2"]:
            uh, ph, solver_info = uh2, ph2, solver_info2

    u_grid = _sample_velocity_magnitude(uh, bbox, nx_out, ny_out)

    return {
        "u": u_grid,
        "solver_info": solver_info,
    }
