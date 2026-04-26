import time
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc

from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl


# ```DIAGNOSIS
# equation_type:        other
# spatial_dim:          2
# domain_geometry:      rectangle
# unknowns:             vector
# coupling:             none
# linearity:            linear
# time_dependence:      steady
# stiffness:            N/A
# dominant_physics:     diffusion
# peclet_or_reynolds:   N/A
# solution_regularity:  smooth
# bc_type:              mixed
# special_notes:        none
# ```
#
# ```METHOD
# spatial_method:       fem
# element_or_basis:     Lagrange_P2
# stabilization:        none
# time_method:          none
# nonlinear_solver:     none
# linear_solver:        preonly
# preconditioner:       lu
# special_treatment:    none
# pde_skill:            none
# ```


def _default_case_spec(case_spec):
    out = {} if case_spec is None else dict(case_spec)
    out.setdefault("pde", {})
    out.setdefault("output", {})
    out["output"].setdefault(
        "grid", {"nx": 64, "ny": 64, "bbox": [0.0, 1.0, 0.0, 1.0]}
    )
    out.setdefault("agent_params", {})
    return out


def _make_bc(V, msh):
    fdim = msh.topology.dim - 1

    def y0(x):
        return np.isclose(x[1], 0.0)

    def y1(x):
        return np.isclose(x[1], 1.0)

    def x1(x):
        return np.isclose(x[0], 1.0)

    zero = fem.Function(V)
    zero.x.array[:] = 0.0
    zero.x.scatter_forward()

    bcs = []
    for marker in (y0, y1, x1):
        facets = mesh.locate_entities_boundary(msh, fdim, marker)
        dofs = fem.locate_dofs_topological(V, fdim, facets)
        if len(dofs) > 0:
            bcs.append(fem.dirichletbc(zero, dofs))
    return bcs


def _solve_velocity(msh, degree):
    gdim = msh.geometry.dim
    V = fem.functionspace(msh, ("Lagrange", degree, (gdim,)))
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    nu = 0.3
    f = fem.Constant(msh, PETSc.ScalarType((1.0, 0.0)))

    a = nu * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx + 1.0e-8 * ufl.inner(u, v) * ufl.dx
    L = ufl.inner(f, v) * ufl.dx

    bcs = _make_bc(V, msh)
    problem = petsc.LinearProblem(
        a,
        L,
        bcs=bcs,
        petsc_options_prefix="vp_",
        petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
    )
    uh = problem.solve()
    uh.x.scatter_forward()
    return V, uh


def _verification(msh, uh):
    div_sq = fem.assemble_scalar(fem.form(ufl.inner(ufl.div(uh), ufl.div(uh)) * ufl.dx))
    grad_sq = fem.assemble_scalar(fem.form(ufl.inner(ufl.grad(uh), ufl.grad(uh)) * ufl.dx))
    l2_sq = fem.assemble_scalar(fem.form(ufl.inner(uh, uh) * ufl.dx))
    comm = msh.comm
    return {
        "divergence_l2": float(np.sqrt(comm.allreduce(div_sq, op=MPI.SUM))),
        "velocity_h1_seminorm": float(np.sqrt(comm.allreduce(grad_sq, op=MPI.SUM))),
        "velocity_l2": float(np.sqrt(comm.allreduce(l2_sq, op=MPI.SUM))),
    }


def _sample_velocity_magnitude(u_fun, msh, nx, ny, bbox):
    xmin, xmax, ymin, ymax = bbox
    xs = np.linspace(xmin, xmax, nx)
    ys = np.linspace(ymin, ymax, ny)
    XX, YY = np.meshgrid(xs, ys, indexing="xy")
    pts = np.column_stack([XX.ravel(), YY.ravel(), np.zeros(nx * ny, dtype=np.float64)])

    tree = geometry.bb_tree(msh, msh.topology.dim)
    cell_candidates = geometry.compute_collisions_points(tree, pts)
    colliding_cells = geometry.compute_colliding_cells(msh, cell_candidates, pts)

    values = np.full((pts.shape[0], msh.geometry.dim), np.nan, dtype=np.float64)
    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    for i in range(pts.shape[0]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(pts[i])
            cells_on_proc.append(links[0])
            eval_map.append(i)

    if points_on_proc:
        vals = u_fun.eval(
            np.array(points_on_proc, dtype=np.float64),
            np.array(cells_on_proc, dtype=np.int32),
        )
        values[np.array(eval_map, dtype=np.int32), :] = vals

    gathered = msh.comm.gather(values, root=0)
    if msh.comm.rank == 0:
        merged = np.full_like(values, np.nan)
        for arr in gathered:
            mask = np.isnan(merged[:, 0]) & np.isfinite(arr[:, 0])
            merged[mask] = arr[mask]
        merged = np.nan_to_num(merged, nan=0.0)
        mag = np.linalg.norm(merged, axis=1).reshape(ny, nx)
    else:
        mag = None
    return msh.comm.bcast(mag, root=0)


def solve(case_spec: dict) -> dict:
    case_spec = _default_case_spec(case_spec)
    params = case_spec.get("agent_params", {})
    mesh_resolution = int(params.get("mesh_resolution", 96))
    degree = int(params.get("degree_u", 2))

    t0 = time.perf_counter()
    msh = mesh.create_unit_square(
        MPI.COMM_WORLD, mesh_resolution, mesh_resolution, cell_type=mesh.CellType.triangle
    )
    _, uh = _solve_velocity(msh, degree)

    grid = case_spec["output"]["grid"]
    u_grid = _sample_velocity_magnitude(
        uh, msh, int(grid["nx"]), int(grid["ny"]), grid["bbox"]
    )

    solver_info = {
        "mesh_resolution": mesh_resolution,
        "element_degree": degree,
        "ksp_type": "preonly",
        "pc_type": "lu",
        "rtol": 1e-10,
        "iterations": 1,
        "nonlinear_iterations": [0],
        "verification": _verification(msh, uh),
        "wall_time_sec": float(time.perf_counter() - t0),
    }
    return {"u": u_grid, "solver_info": solver_info}


if __name__ == "__main__":
    case = {
        "output": {"grid": {"nx": 32, "ny": 32, "bbox": [0.0, 1.0, 0.0, 1.0]}},
        "pde": {"time": None},
    }
    result = solve(case)
    if MPI.COMM_WORLD.rank == 0:
        print(result["u"].shape)
        print(result["solver_info"])
        print(float(np.min(result["u"])), float(np.max(result["u"])))
