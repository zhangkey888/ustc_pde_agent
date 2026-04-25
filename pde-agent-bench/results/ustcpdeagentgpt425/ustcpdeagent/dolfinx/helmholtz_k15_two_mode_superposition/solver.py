import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl

# ```DIAGNOSIS
# equation_type:        helmholtz
# spatial_dim:          2
# domain_geometry:      rectangle
# unknowns:             scalar
# coupling:             none
# linearity:            linear
# time_dependence:      steady
# stiffness:            N/A
# dominant_physics:     wave
# peclet_or_reynolds:   N/A
# solution_regularity:  smooth
# bc_type:              all_dirichlet
# special_notes:        manufactured_solution
# ```
#
# ```METHOD
# spatial_method:       fem
# element_or_basis:     Lagrange_P2
# stabilization:        none
# time_method:          none
# nonlinear_solver:     none
# linear_solver:        direct_lu
# preconditioner:       none
# special_treatment:    none
# pde_skill:            helmholtz
# ```

ScalarType = PETSc.ScalarType


def _exact_numpy(x, y):
    return np.sin(2.0 * np.pi * x) * np.sin(np.pi * y) + np.sin(np.pi * x) * np.sin(3.0 * np.pi * y)


def _make_sampling_grid(grid_spec):
    nx = int(grid_spec["nx"])
    ny = int(grid_spec["ny"])
    xmin, xmax, ymin, ymax = grid_spec["bbox"]
    xs = np.linspace(xmin, xmax, nx)
    ys = np.linspace(ymin, ymax, ny)
    X, Y = np.meshgrid(xs, ys, indexing="xy")
    pts = np.zeros((nx * ny, 3), dtype=np.float64)
    pts[:, 0] = X.ravel()
    pts[:, 1] = Y.ravel()
    return pts, nx, ny


def _sample_function_on_points(domain, uh, points):
    tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(tree, points)
    colliding = geometry.compute_colliding_cells(domain, cell_candidates, points)

    local_values = np.full(points.shape[0], np.nan, dtype=np.float64)
    points_on_proc = []
    cells_on_proc = []
    idx_map = []
    for i in range(points.shape[0]):
        links = colliding.links(i)
        if len(links) > 0:
            points_on_proc.append(points[i])
            cells_on_proc.append(links[0])
            idx_map.append(i)

    if points_on_proc:
        vals = uh.eval(np.asarray(points_on_proc, dtype=np.float64), np.asarray(cells_on_proc, dtype=np.int32))
        vals = np.real(np.asarray(vals).reshape(-1))
        local_values[np.asarray(idx_map, dtype=np.int32)] = vals

    comm = domain.comm
    if comm.size == 1:
        return local_values

    gathered = comm.gather(local_values, root=0)
    out = None
    if comm.rank == 0:
        stacked = np.vstack(gathered)
        out = stacked[0].copy()
        nanmask = np.isnan(out)
        for r in range(1, stacked.shape[0]):
            take = nanmask & (~np.isnan(stacked[r]))
            out[take] = stacked[r, take]
            nanmask = np.isnan(out)
        out[np.isnan(out)] = 0.0
    out = comm.bcast(out, root=0)
    return out


def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD
    pde = case_spec.get("pde", {})
    k = float(pde.get("k", case_spec.get("wavenumber", 15.0)))

    mesh_resolution = int(case_spec.get("solver", {}).get("mesh_resolution", 96))
    element_degree = int(case_spec.get("solver", {}).get("element_degree", 2))
    rtol = float(case_spec.get("solver", {}).get("rtol", 1e-10))

    domain = mesh.create_unit_square(comm, nx=mesh_resolution, ny=mesh_resolution, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", element_degree))

    x = ufl.SpatialCoordinate(domain)
    u_exact_ufl = ufl.sin(2.0 * ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1]) + ufl.sin(ufl.pi * x[0]) * ufl.sin(3.0 * ufl.pi * x[1])
    f_ufl = -ufl.div(ufl.grad(u_exact_ufl)) - (k ** 2) * u_exact_ufl

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    a = (ufl.inner(ufl.grad(u), ufl.grad(v)) - (k ** 2) * ufl.inner(u, v)) * ufl.dx
    L = ufl.inner(f_ufl, v) * ufl.dx

    fdim = domain.topology.dim - 1
    facets = mesh.locate_entities_boundary(domain, fdim, lambda X: np.ones(X.shape[1], dtype=bool))
    dofs = fem.locate_dofs_topological(V, fdim, facets)

    u_bc = fem.Function(V)
    u_bc.interpolate(fem.Expression(u_exact_ufl, V.element.interpolation_points))
    bc = fem.dirichletbc(u_bc, dofs)

    problem = petsc.LinearProblem(
        a,
        L,
        bcs=[bc],
        petsc_options_prefix="helmholtz_",
        petsc_options={"ksp_type": "preonly", "pc_type": "lu", "ksp_rtol": rtol},
    )
    uh = problem.solve()
    uh.x.scatter_forward()

    u_ex_h = fem.Function(V)
    u_ex_h.interpolate(fem.Expression(u_exact_ufl, V.element.interpolation_points))
    err_form = fem.form((uh - u_ex_h) ** 2 * ufl.dx)
    ex_form = fem.form(u_ex_h ** 2 * ufl.dx)
    l2_err = np.sqrt(comm.allreduce(fem.assemble_scalar(err_form), op=MPI.SUM))
    l2_norm_ex = np.sqrt(comm.allreduce(fem.assemble_scalar(ex_form), op=MPI.SUM))
    rel_l2_err = l2_err / max(l2_norm_ex, 1e-30)

    grid_spec = case_spec["output"]["grid"]
    points, nx, ny = _make_sampling_grid(grid_spec)
    vals = _sample_function_on_points(domain, uh, points)
    u_grid = vals.reshape(ny, nx)

    xmin, xmax, ymin, ymax = grid_spec["bbox"]
    xs = np.linspace(xmin, xmax, nx)
    ys = np.linspace(ymin, ymax, ny)
    XX, YY = np.meshgrid(xs, ys, indexing="xy")
    u_exact_grid = _exact_numpy(XX, YY)
    grid_rel_l2 = float(np.linalg.norm(u_grid - u_exact_grid) / max(np.linalg.norm(u_exact_grid), 1e-30))

    solver_info = {
        "mesh_resolution": mesh_resolution,
        "element_degree": element_degree,
        "ksp_type": "preonly",
        "pc_type": "lu",
        "rtol": rtol,
        "iterations": 1,
        "verification_l2_error": float(l2_err),
        "verification_rel_l2_error": float(rel_l2_err),
        "verification_grid_rel_l2_error": grid_rel_l2,
        "wavenumber": k,
    }

    return {"u": u_grid, "solver_info": solver_info}
