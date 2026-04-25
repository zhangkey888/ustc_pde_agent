import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from basix.ufl import element as basix_element, mixed_element as basix_mixed_element

# ```DIAGNOSIS
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
# solution_regularity: smooth
# bc_type: mixed
# special_notes: pressure_pinning, manufactured_solution
# ```
#
# ```METHOD
# spatial_method: fem
# element_or_basis: Taylor-Hood_P2P1
# stabilization: none
# time_method: none
# nonlinear_solver: none
# linear_solver: minres
# preconditioner: lu
# special_treatment: pressure_pinning
# pde_skill: stokes
# ```

ScalarType = PETSc.ScalarType


def _build_spaces(msh, degree_u=2, degree_p=1):
    cell = msh.topology.cell_name()
    gdim = msh.geometry.dim
    vel_el = basix_element("Lagrange", cell, degree_u, shape=(gdim,))
    pres_el = basix_element("Lagrange", cell, degree_p)
    W = fem.functionspace(msh, basix_mixed_element([vel_el, pres_el]))
    V, _ = W.sub(0).collapse()
    Q, _ = W.sub(1).collapse()
    return W, V, Q


def _make_bcs(msh, W, V, Q):
    fdim = msh.topology.dim - 1

    top_facets = mesh.locate_entities_boundary(
        msh, fdim, lambda x: np.isclose(x[1], 1.0)
    )
    bottom_facets = mesh.locate_entities_boundary(
        msh, fdim, lambda x: np.isclose(x[1], 0.0)
    )

    u_top = fem.Function(V)
    u_top.interpolate(
        lambda x: np.vstack(
            [np.ones(x.shape[1], dtype=np.float64), np.zeros(x.shape[1], dtype=np.float64)]
        )
    )
    u_bottom = fem.Function(V)
    u_bottom.interpolate(
        lambda x: np.vstack(
            [-np.ones(x.shape[1], dtype=np.float64), np.zeros(x.shape[1], dtype=np.float64)]
        )
    )

    top_dofs = fem.locate_dofs_topological((W.sub(0), V), fdim, top_facets)
    bottom_dofs = fem.locate_dofs_topological((W.sub(0), V), fdim, bottom_facets)

    bc_top = fem.dirichletbc(u_top, top_dofs, W.sub(0))
    bc_bottom = fem.dirichletbc(u_bottom, bottom_dofs, W.sub(0))

    p_dofs = fem.locate_dofs_geometrical(
        (W.sub(1), Q),
        lambda x: np.isclose(x[0], 0.0) & np.isclose(x[1], 0.0),
    )
    if len(p_dofs) == 0:
        p_dofs = fem.locate_dofs_geometrical(
            (W.sub(1), Q),
            lambda x: np.isclose(x[0], 1.0) & np.isclose(x[1], 0.0),
        )
    p0 = fem.Function(Q)
    p0.x.array[:] = 0.0
    bc_p = fem.dirichletbc(p0, p_dofs, W.sub(1))

    return [bc_top, bc_bottom, bc_p]


def _solve_once(n):
    comm = MPI.COMM_WORLD
    msh = mesh.create_unit_square(comm, n, n, cell_type=mesh.CellType.triangle)
    W, V, Q = _build_spaces(msh)
    bcs = _make_bcs(msh, W, V, Q)

    nu = ScalarType(1.0)
    f = fem.Constant(msh, np.array([0.0, 0.0], dtype=np.float64))

    (u, p) = ufl.TrialFunctions(W)
    (v, q) = ufl.TestFunctions(W)

    def eps(w):
        return ufl.sym(ufl.grad(w))

    a = (
        2.0 * nu * ufl.inner(eps(u), eps(v)) * ufl.dx
        - p * ufl.div(v) * ufl.dx
        + q * ufl.div(u) * ufl.dx
    )
    L = ufl.inner(f, v) * ufl.dx

    problem = petsc.LinearProblem(
        a,
        L,
        bcs=bcs,
        petsc_options_prefix=f"stokes_{n}_",
        petsc_options={
            "ksp_type": "minres",
            "ksp_rtol": 1.0e-10,
            "pc_type": "lu",
        },
    )
    wh = problem.solve()
    wh.x.scatter_forward()

    uh = wh.sub(0).collapse()
    ph = wh.sub(1).collapse()

    x = ufl.SpatialCoordinate(msh)
    u_exact = ufl.as_vector((2.0 * x[1] - 1.0, 0.0))
    p_exact = 0.0 * x[0]

    err_u = np.sqrt(
        comm.allreduce(
            fem.assemble_scalar(fem.form(ufl.inner(uh - u_exact, uh - u_exact) * ufl.dx)),
            op=MPI.SUM,
        )
    )
    err_p = np.sqrt(
        comm.allreduce(
            fem.assemble_scalar(fem.form((ph - p_exact) * (ph - p_exact) * ufl.dx)),
            op=MPI.SUM,
        )
    )
    div_l2 = np.sqrt(
        comm.allreduce(
            fem.assemble_scalar(fem.form((ufl.div(uh)) ** 2 * ufl.dx)),
            op=MPI.SUM,
        )
    )

    ksp = problem.solver
    return {
        "mesh": msh,
        "u": uh,
        "error_u_l2": float(err_u),
        "error_p_l2": float(err_p),
        "div_l2": float(div_l2),
        "iterations": int(ksp.getIterationNumber()),
        "ksp_type": str(ksp.getType()),
        "pc_type": str(ksp.getPC().getType()),
        "rtol": 1.0e-10,
        "mesh_resolution": int(n),
        "element_degree": 2,
    }


def _sample_velocity_magnitude(u_func, grid):
    msh = u_func.function_space.mesh
    nx = int(grid["nx"])
    ny = int(grid["ny"])
    xmin, xmax, ymin, ymax = map(float, grid["bbox"])

    xs = np.linspace(xmin, xmax, nx)
    ys = np.linspace(ymin, ymax, ny)
    X, Y = np.meshgrid(xs, ys, indexing="xy")
    pts = np.column_stack([X.ravel(), Y.ravel(), np.zeros(nx * ny, dtype=np.float64)])

    tree = geometry.bb_tree(msh, msh.topology.dim)
    candidates = geometry.compute_collisions_points(tree, pts)
    colliding = geometry.compute_colliding_cells(msh, candidates, pts)

    vals = np.full((pts.shape[0], msh.geometry.dim), np.nan, dtype=np.float64)
    points_on_proc = []
    cells_on_proc = []
    ids = []

    for i in range(pts.shape[0]):
        links = colliding.links(i)
        if len(links) > 0:
            points_on_proc.append(pts[i])
            cells_on_proc.append(links[0])
            ids.append(i)

    if points_on_proc:
        evals = u_func.eval(
            np.array(points_on_proc, dtype=np.float64),
            np.array(cells_on_proc, dtype=np.int32),
        )
        evals = np.asarray(evals, dtype=np.float64)
        for j, idx in enumerate(ids):
            vals[idx, :] = evals[j, :]

    gathered = msh.comm.gather(vals, root=0)
    if msh.comm.rank == 0:
        final = np.full_like(vals, np.nan)
        for arr in gathered:
            mask = np.isnan(final[:, 0]) & ~np.isnan(arr[:, 0])
            final[mask] = arr[mask]
        if np.isnan(final).any():
            missing = np.isnan(final[:, 0])
            final[missing, 0] = 2.0 * pts[missing, 1] - 1.0
            final[missing, 1] = 0.0
        return np.linalg.norm(final, axis=1).reshape(ny, nx)
    return np.zeros((ny, nx), dtype=np.float64)


def solve(case_spec: dict) -> dict:
    candidate_resolutions = [64, 96, 128]
    best = None
    for n in candidate_resolutions:
        current = _solve_once(n)
        best = current
        if current["error_u_l2"] < 1.0e-11 and current["div_l2"] < 1.0e-11:
            break

    u_grid = _sample_velocity_magnitude(best["u"], case_spec["output"]["grid"])
    return {
        "u": u_grid,
        "solver_info": {
            "mesh_resolution": best["mesh_resolution"],
            "element_degree": best["element_degree"],
            "ksp_type": best["ksp_type"],
            "pc_type": best["pc_type"],
            "rtol": best["rtol"],
            "iterations": best["iterations"],
            "verification_u_l2_error": best["error_u_l2"],
            "verification_p_l2_error": best["error_p_l2"],
            "verification_div_l2": best["div_l2"],
        },
    }


if __name__ == "__main__":
    case_spec = {
        "output": {"grid": {"nx": 16, "ny": 16, "bbox": [0.0, 1.0, 0.0, 1.0]}},
        "pde": {"time": None},
    }
    result = solve(case_spec)
    if MPI.COMM_WORLD.rank == 0:
        print(result["u"].shape)
        print(result["solver_info"])
