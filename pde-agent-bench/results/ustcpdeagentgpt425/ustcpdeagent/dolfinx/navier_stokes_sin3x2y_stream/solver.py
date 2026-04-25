import time
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
import ufl
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
from basix.ufl import element as basix_element, mixed_element as basix_mixed_element

ScalarType = PETSc.ScalarType

# ```DIAGNOSIS
# equation_type:        navier_stokes
# spatial_dim:          2
# domain_geometry:      rectangle
# unknowns:             vector+scalar
# coupling:             saddle_point
# linearity:            nonlinear
# time_dependence:      steady
# stiffness:            N/A
# dominant_physics:     mixed
# peclet_or_reynolds:   moderate
# solution_regularity:  smooth
# bc_type:              all_dirichlet
# special_notes:        manufactured_solution
# ```
#
# ```METHOD
# spatial_method:       fem
# element_or_basis:     Taylor-Hood_P2P1
# stabilization:        none
# time_method:          none
# nonlinear_solver:     none
# linear_solver:        gmres
# preconditioner:       ilu
# special_treatment:    pressure_pinning
# pde_skill:            navier_stokes
# ```


def _manufactured_fields(msh):
    x = ufl.SpatialCoordinate(msh)
    pi = ufl.pi
    nu = ScalarType(0.1)
    u_ex = ufl.as_vector(
        (
            2 * pi * ufl.cos(2 * pi * x[1]) * ufl.sin(3 * pi * x[0]),
            -3 * pi * ufl.cos(3 * pi * x[0]) * ufl.sin(2 * pi * x[1]),
        )
    )
    p_ex = ufl.cos(pi * x[0]) * ufl.cos(2 * pi * x[1])
    f = ufl.grad(u_ex) * u_ex - nu * ufl.div(ufl.grad(u_ex)) + ufl.grad(p_ex)
    return u_ex, p_ex, f


def _sample_velocity_magnitude(u_func, msh, nx, ny, bbox):
    xmin, xmax, ymin, ymax = bbox
    xs = np.linspace(xmin, xmax, nx)
    ys = np.linspace(ymin, ymax, ny)
    XX, YY = np.meshgrid(xs, ys, indexing="xy")
    pts = np.column_stack([XX.ravel(), YY.ravel(), np.zeros(nx * ny, dtype=np.float64)])

    tree = geometry.bb_tree(msh, msh.topology.dim)
    candidates = geometry.compute_collisions_points(tree, pts)
    colliding = geometry.compute_colliding_cells(msh, candidates, pts)

    points_on_proc = []
    cells_on_proc = []
    eval_ids = []
    for i in range(pts.shape[0]):
        links = colliding.links(i)
        if len(links) > 0:
            points_on_proc.append(pts[i])
            cells_on_proc.append(links[0])
            eval_ids.append(i)

    local_vals = np.full(pts.shape[0], np.nan, dtype=np.float64)
    if points_on_proc:
        vals = u_func.eval(np.array(points_on_proc, dtype=np.float64), np.array(cells_on_proc, dtype=np.int32))
        mags = np.linalg.norm(np.asarray(vals), axis=1)
        local_vals[np.array(eval_ids, dtype=np.int32)] = mags

    gathered = msh.comm.gather(local_vals, root=0)
    if msh.comm.rank == 0:
        out = np.full(pts.shape[0], np.nan, dtype=np.float64)
        for arr in gathered:
            mask = np.isnan(out) & ~np.isnan(arr)
            out[mask] = arr[mask]
        out = np.nan_to_num(out, nan=0.0)
        return out.reshape(ny, nx)
    return None


def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD
    rank = comm.rank
    t_start = time.perf_counter()

    grid = case_spec["output"]["grid"]
    nx = int(grid["nx"])
    ny = int(grid["ny"])
    bbox = grid["bbox"]

    mesh_resolution = int(case_spec.get("mesh_resolution", 64))
    degree_u = int(case_spec.get("degree_u", 2))
    degree_p = int(case_spec.get("degree_p", 1))
    nonlin_rtol = float(case_spec.get("newton_rtol", 1e-10))
    nonlin_max_it = int(case_spec.get("newton_max_it", 25))

    msh = mesh.create_unit_square(comm, mesh_resolution, mesh_resolution, cell_type=mesh.CellType.triangle)
    gdim = msh.geometry.dim

    vel_el = basix_element("Lagrange", msh.topology.cell_name(), degree_u, shape=(gdim,))
    pre_el = basix_element("Lagrange", msh.topology.cell_name(), degree_p)
    W = fem.functionspace(msh, basix_mixed_element([vel_el, pre_el]))
    V, _ = W.sub(0).collapse()
    Q, _ = W.sub(1).collapse()

    u_ex_ufl, p_ex_ufl, f_ufl = _manufactured_fields(msh)

    u_bc = fem.Function(V)
    u_bc.interpolate(fem.Expression(u_ex_ufl, V.element.interpolation_points))

    fdim = msh.topology.dim - 1
    facets = mesh.locate_entities_boundary(msh, fdim, lambda x: np.ones(x.shape[1], dtype=bool))
    u_dofs = fem.locate_dofs_topological((W.sub(0), V), fdim, facets)
    bc_u = fem.dirichletbc(u_bc, u_dofs, W.sub(0))

    p_dofs = fem.locate_dofs_geometrical(
        (W.sub(1), Q), lambda x: np.isclose(x[0], 0.0) & np.isclose(x[1], 0.0)
    )
    p0 = fem.Function(Q)
    p0.x.array[:] = 0.0
    bcs = [bc_u]
    if len(p_dofs) > 0:
        bcs.append(fem.dirichletbc(p0, p_dofs, W.sub(1)))

    (u, p) = ufl.TrialFunctions(W)
    (v, q) = ufl.TestFunctions(W)
    nu = ScalarType(0.1)

    a_stokes = (
        2 * nu * ufl.inner(ufl.sym(ufl.grad(u)), ufl.sym(ufl.grad(v))) * ufl.dx
        - ufl.inner(p, ufl.div(v)) * ufl.dx
        + ufl.inner(ufl.div(u), q) * ufl.dx
    )
    L_stokes = ufl.inner(f_ufl, v) * ufl.dx

    ksp_type = "gmres"
    pc_type = "ilu"
    rtol = 1e-10
    iterations = 0

    try:
        stokes_problem = petsc.LinearProblem(
            a_stokes,
            L_stokes,
            bcs=bcs,
            petsc_options_prefix="stokes_",
            petsc_options={"ksp_type": ksp_type, "pc_type": pc_type, "ksp_rtol": rtol},
        )
        w = stokes_problem.solve()
    except Exception:
        ksp_type = "preonly"
        pc_type = "lu"
        stokes_problem = petsc.LinearProblem(
            a_stokes,
            L_stokes,
            bcs=bcs,
            petsc_options_prefix="stokes_lu_",
            petsc_options={"ksp_type": ksp_type, "pc_type": pc_type},
        )
        w = stokes_problem.solve()

    uh_prev, _ = w.sub(0).collapse(), w.sub(1).collapse()
    nonlinear_iterations = []

    for k in range(nonlin_max_it):
        a_picard = (
            2 * nu * ufl.inner(ufl.sym(ufl.grad(u)), ufl.sym(ufl.grad(v))) * ufl.dx
            + ufl.inner(ufl.grad(u) * uh_prev, v) * ufl.dx
            - ufl.inner(p, ufl.div(v)) * ufl.dx
            + ufl.inner(ufl.div(u), q) * ufl.dx
        )
        L_picard = ufl.inner(f_ufl, v) * ufl.dx

        try:
            picard_problem = petsc.LinearProblem(
                a_picard,
                L_picard,
                bcs=bcs,
                petsc_options_prefix=f"picard_{k}_",
                petsc_options={"ksp_type": ksp_type, "pc_type": pc_type, "ksp_rtol": rtol},
            )
            w_new = picard_problem.solve()
        except Exception:
            ksp_type = "preonly"
            pc_type = "lu"
            picard_problem = petsc.LinearProblem(
                a_picard,
                L_picard,
                bcs=bcs,
                petsc_options_prefix=f"picard_lu_{k}_",
                petsc_options={"ksp_type": ksp_type, "pc_type": pc_type},
            )
            w_new = picard_problem.solve()

        uh_new, _ = w_new.sub(0).collapse(), w_new.sub(1).collapse()
        diff = uh_new.x.array - uh_prev.x.array
        num = comm.allreduce(np.dot(diff, diff), op=MPI.SUM)
        den = comm.allreduce(max(np.dot(uh_new.x.array, uh_new.x.array), 1e-30), op=MPI.SUM)
        rel = np.sqrt(num / den)
        nonlinear_iterations.append(k + 1)
        w = w_new
        uh_prev.x.array[:] = uh_new.x.array
        uh_prev.x.scatter_forward()
        if rel < nonlin_rtol:
            break

    uh, ph = w.sub(0).collapse(), w.sub(1).collapse()

    u_ex = fem.Function(V)
    u_ex.interpolate(fem.Expression(u_ex_ufl, V.element.interpolation_points))
    err_sq_local = fem.assemble_scalar(fem.form(ufl.inner(uh - u_ex, uh - u_ex) * ufl.dx))
    ref_sq_local = fem.assemble_scalar(fem.form(ufl.inner(u_ex, u_ex) * ufl.dx))
    l2_error = np.sqrt(comm.allreduce(err_sq_local, op=MPI.SUM))
    rel_l2_error = l2_error / max(np.sqrt(comm.allreduce(ref_sq_local, op=MPI.SUM)), 1e-30)

    u_grid = _sample_velocity_magnitude(uh, msh, nx, ny, bbox)
    wall_time = time.perf_counter() - t_start

    solver_info = {
        "mesh_resolution": mesh_resolution,
        "element_degree": degree_u,
        "ksp_type": ksp_type,
        "pc_type": pc_type,
        "rtol": rtol,
        "iterations": int(iterations),
        "nonlinear_iterations": nonlinear_iterations if nonlinear_iterations else [0],
        "l2_error": float(l2_error),
        "relative_l2_error": float(rel_l2_error),
        "wall_time_sec": float(wall_time),
    }

    if rank == 0:
        return {"u": u_grid, "solver_info": solver_info}
    return {"u": np.zeros((ny, nx), dtype=np.float64), "solver_info": solver_info}


if __name__ == "__main__":
    case_spec = {
        "output": {"grid": {"nx": 32, "ny": 32, "bbox": [0.0, 1.0, 0.0, 1.0]}},
        "pde": {"time": None},
    }
    result = solve(case_spec)
    if MPI.COMM_WORLD.rank == 0:
        print(result["u"].shape)
        print(result["solver_info"])
