import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
import ufl
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc as fem_petsc
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
# special_notes: pressure_pinning
# ```
#
# ```METHOD
# spatial_method: fem
# element_or_basis: Taylor-Hood_P2P1
# stabilization: none
# time_method: none
# nonlinear_solver: none
# linear_solver: minres
# preconditioner: hypre
# special_treatment: pressure_pinning
# pde_skill: stokes
# ```


def _default_grid(case_spec: dict):
    out = case_spec.get("output", {})
    grid = out.get("grid", {})
    nx = int(grid.get("nx", 64))
    ny = int(grid.get("ny", 64))
    bbox = grid.get("bbox", [0.0, 1.0, 0.0, 1.0])
    return nx, ny, bbox


def _choose_resolution(case_spec: dict):
    nx_out, ny_out, _ = _default_grid(case_spec)
    base = max(48, min(160, 2 * max(nx_out, ny_out)))
    return int(base)


def _build_problem(comm, n):
    msh = mesh.create_unit_square(comm, n, n, cell_type=mesh.CellType.triangle)
    gdim = msh.geometry.dim
    cell = msh.topology.cell_name()

    vel_el = basix_element("Lagrange", cell, 2, shape=(gdim,))
    pre_el = basix_element("Lagrange", cell, 1)
    W = fem.functionspace(msh, basix_mixed_element([vel_el, pre_el]))

    V, _ = W.sub(0).collapse()
    Q, _ = W.sub(1).collapse()

    nu = fem.Constant(msh, PETSc.ScalarType(1.0))
    f = fem.Constant(msh, PETSc.ScalarType((0.0, 0.0)))

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

    fdim = msh.topology.dim - 1

    def left(x):
        return np.isclose(x[0], 0.0)

    def bottom(x):
        return np.isclose(x[1], 0.0)

    def top(x):
        return np.isclose(x[1], 1.0)

    inflow_facets = mesh.locate_entities_boundary(msh, fdim, left)
    bottom_facets = mesh.locate_entities_boundary(msh, fdim, bottom)
    top_facets = mesh.locate_entities_boundary(msh, fdim, top)

    u_in = fem.Function(V)
    u_in.interpolate(lambda x: np.vstack((4.0 * x[1] * (1.0 - x[1]), np.zeros(x.shape[1]))))

    u_zero = fem.Function(V)
    u_zero.x.array[:] = 0.0

    dofs_in = fem.locate_dofs_topological((W.sub(0), V), fdim, inflow_facets)
    dofs_bottom = fem.locate_dofs_topological((W.sub(0), V), fdim, bottom_facets)
    dofs_top = fem.locate_dofs_topological((W.sub(0), V), fdim, top_facets)

    bc_in = fem.dirichletbc(u_in, dofs_in, W.sub(0))
    bc_bottom = fem.dirichletbc(u_zero, dofs_bottom, W.sub(0))
    bc_top = fem.dirichletbc(u_zero, dofs_top, W.sub(0))

    p0_dofs = fem.locate_dofs_geometrical(
        (W.sub(1), Q),
        lambda x: np.isclose(x[0], 0.0) & np.isclose(x[1], 0.0),
    )
    p_zero = fem.Function(Q)
    p_zero.x.array[:] = 0.0
    bc_p = fem.dirichletbc(p_zero, p0_dofs, W.sub(1))

    bcs = [bc_in, bc_bottom, bc_top, bc_p]
    return msh, W, V, Q, a, L, bcs


def _solve_stokes(msh, W, a, L, bcs):
    opts = {
        "ksp_type": "minres",
        "pc_type": "hypre",
        "ksp_rtol": 1.0e-10,
        "ksp_atol": 1.0e-12,
        "ksp_max_it": 5000,
    }
    try:
        problem = fem_petsc.LinearProblem(
            a, L, bcs=bcs, petsc_options_prefix="stokes_", petsc_options=opts
        )
        wh = problem.solve()
        ksp = problem.solver
        its = int(ksp.getIterationNumber())
        ksp_type = ksp.getType()
        pc_type = ksp.getPC().getType()
        rtol = float(ksp.getTolerances()[0])
        return wh, {
            "ksp_type": str(ksp_type),
            "pc_type": str(pc_type),
            "rtol": rtol,
            "iterations": its,
        }
    except Exception:
        opts = {
            "ksp_type": "preonly",
            "pc_type": "lu",
        }
        problem = fem_petsc.LinearProblem(
            a, L, bcs=bcs, petsc_options_prefix="stokes_fallback_", petsc_options=opts
        )
        wh = problem.solve()
        return wh, {
            "ksp_type": "preonly",
            "pc_type": "lu",
            "rtol": 0.0,
            "iterations": 1,
        }


def _sample_velocity_magnitude(u_fun, msh, nx, ny, bbox):
    xmin, xmax, ymin, ymax = bbox
    xs = np.linspace(xmin, xmax, nx)
    ys = np.linspace(ymin, ymax, ny)
    XX, YY = np.meshgrid(xs, ys, indexing="xy")
    pts2 = np.column_stack([XX.ravel(), YY.ravel(), np.zeros(nx * ny, dtype=np.float64)])

    tree = geometry.bb_tree(msh, msh.topology.dim)
    cell_candidates = geometry.compute_collisions_points(tree, pts2)
    colliding = geometry.compute_colliding_cells(msh, cell_candidates, pts2)

    values = np.full((nx * ny,), np.nan, dtype=np.float64)
    points_on_proc = []
    cells = []
    idxs = []
    for i in range(pts2.shape[0]):
        links = colliding.links(i)
        if len(links) > 0:
            points_on_proc.append(pts2[i])
            cells.append(links[0])
            idxs.append(i)

    if len(points_on_proc) > 0:
        vals = u_fun.eval(np.array(points_on_proc, dtype=np.float64), np.array(cells, dtype=np.int32))
        mag = np.linalg.norm(vals, axis=1)
        values[np.array(idxs, dtype=np.int32)] = mag

    if np.any(np.isnan(values)):
        nan_idx = np.where(np.isnan(values))[0]
        tol_shift = 1.0e-12
        pts_fix = pts2[nan_idx].copy()
        pts_fix[:, 0] = np.clip(pts_fix[:, 0], xmin + tol_shift, xmax - tol_shift)
        pts_fix[:, 1] = np.clip(pts_fix[:, 1], ymin + tol_shift, ymax - tol_shift)
        cell_candidates = geometry.compute_collisions_points(tree, pts_fix)
        colliding = geometry.compute_colliding_cells(msh, cell_candidates, pts_fix)
        points2 = []
        cells2 = []
        map2 = []
        for j in range(pts_fix.shape[0]):
            links = colliding.links(j)
            if len(links) > 0:
                points2.append(pts_fix[j])
                cells2.append(links[0])
                map2.append(j)
        if len(points2) > 0:
            vals2 = u_fun.eval(np.array(points2, dtype=np.float64), np.array(cells2, dtype=np.int32))
            mag2 = np.linalg.norm(vals2, axis=1)
            values[nan_idx[np.array(map2, dtype=np.int32)]] = mag2

    values = np.nan_to_num(values, nan=0.0)
    return values.reshape(ny, nx)


def _compute_diagnostics(msh, u_fun):
    x = ufl.SpatialCoordinate(msh)
    n = ufl.FacetNormal(msh)

    div_l2_local = fem.assemble_scalar(fem.form(ufl.inner(ufl.div(u_fun), ufl.div(u_fun)) * ufl.dx))
    div_l2 = msh.comm.allreduce(div_l2_local, op=MPI.SUM) ** 0.5

    inflow_flux_local = fem.assemble_scalar(
        fem.form(ufl.conditional(ufl.lt(x[0], 1.0e-8), ufl.dot(u_fun, n), 0.0) * ufl.ds)
    )
    outflow_flux_local = fem.assemble_scalar(
        fem.form(ufl.conditional(ufl.gt(x[0], 1.0 - 1.0e-8), ufl.dot(u_fun, n), 0.0) * ufl.ds)
    )
    wall_flux_local = fem.assemble_scalar(
        fem.form(
            (
                ufl.conditional(ufl.lt(x[1], 1.0e-8), ufl.dot(u_fun, n), 0.0)
                + ufl.conditional(ufl.gt(x[1], 1.0 - 1.0e-8), ufl.dot(u_fun, n), 0.0)
            )
            * ufl.ds
        )
    )

    inflow_flux = msh.comm.allreduce(inflow_flux_local, op=MPI.SUM)
    outflow_flux = msh.comm.allreduce(outflow_flux_local, op=MPI.SUM)
    wall_flux = msh.comm.allreduce(wall_flux_local, op=MPI.SUM)

    return {
        "divergence_l2": float(div_l2),
        "inflow_flux": float(inflow_flux),
        "outflow_flux": float(outflow_flux),
        "wall_flux": float(wall_flux),
        "mass_balance_abs": float(abs(inflow_flux + outflow_flux + wall_flux)),
    }


def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD
    n = _choose_resolution(case_spec)
    msh, W, V, Q, a, L, bcs = _build_problem(comm, n)
    wh, lin_info = _solve_stokes(msh, W, a, L, bcs)
    u_h, _ = wh.sub(0).collapse()

    nx, ny, bbox = _default_grid(case_spec)
    u_grid = _sample_velocity_magnitude(u_h, msh, nx, ny, bbox)

    diagnostics = _compute_diagnostics(msh, u_h)

    solver_info = {
        "mesh_resolution": int(n),
        "element_degree": 2,
        "ksp_type": lin_info["ksp_type"],
        "pc_type": lin_info["pc_type"],
        "rtol": float(lin_info["rtol"]),
        "iterations": int(lin_info["iterations"]),
        "accuracy_verification": diagnostics,
    }

    return {"u": u_grid, "solver_info": solver_info}


if __name__ == "__main__":
    case_spec = {
        "output": {"grid": {"nx": 64, "ny": 64, "bbox": [0.0, 1.0, 0.0, 1.0]}},
        "pde": {"time": None},
    }
    result = solve(case_spec)
    if MPI.COMM_WORLD.rank == 0:
        print(result["u"].shape)
        print(result["solver_info"])
