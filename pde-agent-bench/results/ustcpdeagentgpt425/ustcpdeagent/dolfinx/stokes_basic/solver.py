import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
import ufl
from basix.ufl import element as basix_element, mixed_element as basix_mixed_element
from dolfinx import mesh, fem, geometry
from dolfinx.fem.petsc import LinearProblem

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
# bc_type: all_dirichlet
# special_notes: pressure_pinning / manufactured_solution
# ```
#
# ```METHOD
# spatial_method: fem
# element_or_basis: Taylor-Hood_P2P1
# stabilization: none
# time_method: none
# nonlinear_solver: none
# linear_solver: direct_lu
# preconditioner: none
# special_treatment: pressure_pinning
# pde_skill: stokes
# ```


def _u_exact_ufl(x):
    return ufl.as_vector(
        [
            ufl.pi * ufl.cos(ufl.pi * x[1]) * ufl.sin(ufl.pi * x[0]),
            -ufl.pi * ufl.cos(ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1]),
        ]
    )


def _p_exact_ufl(x):
    return ufl.cos(ufl.pi * x[0]) * ufl.cos(ufl.pi * x[1])


def _forcing_ufl(msh, nu):
    x = ufl.SpatialCoordinate(msh)
    u_ex = _u_exact_ufl(x)
    p_ex = _p_exact_ufl(x)
    return -nu * ufl.div(ufl.grad(u_ex)) + ufl.grad(p_ex)


def _build_spaces(msh):
    gdim = msh.geometry.dim
    vel_el = basix_element("Lagrange", msh.topology.cell_name(), 2, shape=(gdim,))
    pre_el = basix_element("Lagrange", msh.topology.cell_name(), 1)
    W = fem.functionspace(msh, basix_mixed_element([vel_el, pre_el]))
    V, _ = W.sub(0).collapse()
    Q, _ = W.sub(1).collapse()
    return W, V, Q


def _make_bcs(msh, W, V, Q):
    x = ufl.SpatialCoordinate(msh)
    fdim = msh.topology.dim - 1
    facets = mesh.locate_entities_boundary(msh, fdim, lambda X: np.ones(X.shape[1], dtype=bool))

    u_bc = fem.Function(V)
    u_bc.interpolate(fem.Expression(_u_exact_ufl(x), V.element.interpolation_points))
    u_dofs = fem.locate_dofs_topological((W.sub(0), V), fdim, facets)
    bc_u = fem.dirichletbc(u_bc, u_dofs, W.sub(0))

    p_dofs = fem.locate_dofs_geometrical(
        (W.sub(1), Q), lambda X: np.isclose(X[0], 0.0) & np.isclose(X[1], 0.0)
    )
    p0 = fem.Function(Q)
    p0.x.array[:] = 0.0
    bc_p = fem.dirichletbc(p0, p_dofs, W.sub(1))

    return [bc_u, bc_p]


def _solve_once(mesh_resolution):
    comm = MPI.COMM_WORLD
    msh = mesh.create_unit_square(comm, mesh_resolution, mesh_resolution, cell_type=mesh.CellType.triangle)
    W, V, Q = _build_spaces(msh)
    bcs = _make_bcs(msh, W, V, Q)

    nu = PETSc.ScalarType(1.0)
    f = _forcing_ufl(msh, nu)

    (u, p) = ufl.TrialFunctions(W)
    (v, q) = ufl.TestFunctions(W)

    a = (
        nu * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
        - ufl.inner(p, ufl.div(v)) * ufl.dx
        + ufl.inner(ufl.div(u), q) * ufl.dx
    )
    L = ufl.inner(f, v) * ufl.dx

    problem = LinearProblem(
        a,
        L,
        bcs=bcs,
        petsc_options_prefix=f"stokes_{mesh_resolution}_",
        petsc_options={
            "ksp_type": "preonly",
            "pc_type": "lu",
            "pc_factor_mat_solver_type": "mumps",
        },
    )
    wh = problem.solve()
    wh.x.scatter_forward()

    uh = wh.sub(0).collapse()
    ph = wh.sub(1).collapse()

    x = ufl.SpatialCoordinate(msh)
    u_ex = fem.Function(V)
    u_ex.interpolate(fem.Expression(_u_exact_ufl(x), V.element.interpolation_points))
    p_ex = fem.Function(Q)
    p_ex.interpolate(fem.Expression(_p_exact_ufl(x), Q.element.interpolation_points))

    err_u_l2 = np.sqrt(comm.allreduce(fem.assemble_scalar(fem.form(ufl.inner(uh - u_ex, uh - u_ex) * ufl.dx)), op=MPI.SUM))
    err_p_l2 = np.sqrt(comm.allreduce(fem.assemble_scalar(fem.form((ph - p_ex) * (ph - p_ex) * ufl.dx)), op=MPI.SUM))

    return msh, uh, {
        "mesh_resolution": int(mesh_resolution),
        "element_degree": 2,
        "ksp_type": "preonly",
        "pc_type": "lu",
        "rtol": 0.0,
        "iterations": 1,
        "u_l2_error": float(err_u_l2),
        "p_l2_error": float(err_p_l2),
    }


def _sample_velocity_magnitude(u_fun, grid_spec):
    msh = u_fun.function_space.mesh
    comm = msh.comm

    nx = int(grid_spec["nx"])
    ny = int(grid_spec["ny"])
    xmin, xmax, ymin, ymax = grid_spec["bbox"]

    xs = np.linspace(xmin, xmax, nx)
    ys = np.linspace(ymin, ymax, ny)
    XX, YY = np.meshgrid(xs, ys, indexing="xy")
    pts = np.column_stack([XX.ravel(), YY.ravel(), np.zeros(nx * ny, dtype=np.float64)])

    tree = geometry.bb_tree(msh, msh.topology.dim)
    candidates = geometry.compute_collisions_points(tree, pts)
    colliding = geometry.compute_colliding_cells(msh, candidates, pts)

    local_vals = np.full((pts.shape[0], msh.geometry.dim), np.nan, dtype=np.float64)
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
        vals = u_fun.eval(np.array(points_on_proc, dtype=np.float64), np.array(cells_on_proc, dtype=np.int32))
        local_vals[np.array(ids, dtype=np.int32)] = np.asarray(vals, dtype=np.float64)

    gathered = comm.gather(local_vals, root=0)
    if comm.rank == 0:
        merged = np.full_like(local_vals, np.nan)
        for arr in gathered:
            mask = np.isnan(merged[:, 0]) & ~np.isnan(arr[:, 0])
            merged[mask] = arr[mask]
        if np.isnan(merged).any():
            raise RuntimeError("Failed to evaluate FEM solution at some output grid points.")
        mag = np.linalg.norm(merged, axis=1).reshape(ny, nx)
    else:
        mag = None

    mag = comm.bcast(mag, root=0)
    return mag


def solve(case_spec: dict) -> dict:
    mesh_resolution = 96
    _, uh, info = _solve_once(mesh_resolution)
    u_grid = _sample_velocity_magnitude(uh, case_spec["output"]["grid"])

    solver_info = {
        "mesh_resolution": info["mesh_resolution"],
        "element_degree": info["element_degree"],
        "ksp_type": info["ksp_type"],
        "pc_type": info["pc_type"],
        "rtol": info["rtol"],
        "iterations": info["iterations"],
        "u_l2_error": info["u_l2_error"],
        "p_l2_error": info["p_l2_error"],
    }
    return {"u": u_grid, "solver_info": solver_info}
