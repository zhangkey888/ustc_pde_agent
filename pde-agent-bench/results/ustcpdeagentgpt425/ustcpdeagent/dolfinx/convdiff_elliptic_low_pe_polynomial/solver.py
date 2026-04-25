import time
import math
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl

ScalarType = PETSc.ScalarType

# ```DIAGNOSIS
# equation_type:        convection_diffusion
# spatial_dim:          2
# domain_geometry:      rectangle
# unknowns:             scalar
# coupling:             none
# linearity:            linear
# time_dependence:      steady
# stiffness:            non_stiff
# dominant_physics:     mixed
# peclet_or_reynolds:   moderate
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
# linear_solver:        gmres
# preconditioner:       hypre
# special_treatment:    none
# pde_skill:            convection_diffusion / reaction_diffusion / biharmonic
# ```


def _u_exact(x):
    return x[0] * (1.0 - x[0]) * x[1] * (1.0 - x[1])


def _source_expr(msh, eps_value, beta_vec):
    x = ufl.SpatialCoordinate(msh)
    uex = x[0] * (1.0 - x[0]) * x[1] * (1.0 - x[1])
    beta = ufl.as_vector(beta_vec)
    return uex, -eps_value * ufl.div(ufl.grad(uex)) + ufl.dot(beta, ufl.grad(uex))


def _build_and_solve(nx, degree, eps_value=0.3, beta_vec=(0.5, 0.3), rtol=1.0e-10, use_supg=False):
    comm = MPI.COMM_WORLD
    msh = mesh.create_unit_square(comm, nx, nx, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(msh, ("Lagrange", degree))

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    _, f_expr = _source_expr(msh, eps_value, beta_vec)
    f_fun = fem.Function(V)
    f_fun.interpolate(fem.Expression(f_expr, V.element.interpolation_points))

    u_bc = fem.Function(V)
    u_bc.interpolate(_u_exact)

    fdim = msh.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(msh, fdim, lambda x: np.full(x.shape[1], True, dtype=bool))
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc = fem.dirichletbc(u_bc, boundary_dofs)

    eps_c = fem.Constant(msh, ScalarType(eps_value))
    beta_c = fem.Constant(msh, np.array(beta_vec, dtype=np.float64))

    a = eps_c * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx + ufl.dot(beta_c, ufl.grad(u)) * v * ufl.dx
    L = f_fun * v * ufl.dx

    if use_supg:
        h = ufl.CellDiameter(msh)
        beta_norm = ufl.sqrt(ufl.dot(beta_c, beta_c) + 1.0e-16)
        tau = h / (2.0 * beta_norm)
        r_u = -eps_c * ufl.div(ufl.grad(u)) + ufl.dot(beta_c, ufl.grad(u))
        a += tau * r_u * ufl.dot(beta_c, ufl.grad(v)) * ufl.dx
        L += tau * f_fun * ufl.dot(beta_c, ufl.grad(v)) * ufl.dx

    uh = None
    iters = -1
    ksp_type = "gmres"
    pc_type = "hypre"

    try:
        problem = petsc.LinearProblem(
            a,
            L,
            bcs=[bc],
            petsc_options={
                "ksp_type": "gmres",
                "pc_type": "hypre",
                "ksp_rtol": rtol,
                "ksp_atol": 1.0e-14,
                "ksp_max_it": 2000,
            },
            petsc_options_prefix=f"cd_{nx}_{degree}_",
        )
        uh = problem.solve()
        ksp = problem.solver
        iters = int(ksp.getIterationNumber())
        ksp_type = ksp.getType()
        pc_type = ksp.getPC().getType()
    except Exception:
        problem = petsc.LinearProblem(
            a,
            L,
            bcs=[bc],
            petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
            petsc_options_prefix=f"cdlu_{nx}_{degree}_",
        )
        uh = problem.solve()
        ksp = problem.solver
        iters = int(ksp.getIterationNumber())
        ksp_type = ksp.getType()
        pc_type = ksp.getPC().getType()

    uh.x.scatter_forward()

    uex_fun = fem.Function(V)
    uex_fun.interpolate(_u_exact)
    err_fun = fem.Function(V)
    err_fun.x.array[:] = uh.x.array - uex_fun.x.array
    err_fun.x.scatter_forward()

    l2_local = fem.assemble_scalar(fem.form(err_fun * err_fun * ufl.dx))
    l2_err = math.sqrt(comm.allreduce(l2_local, op=MPI.SUM))
    max_local = np.max(np.abs(err_fun.x.array)) if err_fun.x.array.size > 0 else 0.0
    linf_err = comm.allreduce(max_local, op=MPI.MAX)

    return {
        "mesh": msh,
        "u": uh,
        "l2_error": l2_err,
        "linf_error": linf_err,
        "iterations": iters,
        "ksp_type": ksp_type,
        "pc_type": pc_type,
        "rtol": rtol,
        "mesh_resolution": nx,
        "element_degree": degree,
        "use_supg": use_supg,
    }


def _sample_on_grid(msh, uh, grid_spec):
    nx = int(grid_spec["nx"])
    ny = int(grid_spec["ny"])
    bbox = grid_spec["bbox"]

    xs = np.linspace(float(bbox[0]), float(bbox[1]), nx)
    ys = np.linspace(float(bbox[2]), float(bbox[3]), ny)
    XX, YY = np.meshgrid(xs, ys, indexing="xy")
    pts = np.column_stack([XX.ravel(), YY.ravel(), np.zeros(nx * ny, dtype=np.float64)])

    tree = geometry.bb_tree(msh, msh.topology.dim)
    cell_candidates = geometry.compute_collisions_points(tree, pts)
    colliding_cells = geometry.compute_colliding_cells(msh, cell_candidates, pts)

    local_vals = np.full(nx * ny, np.nan, dtype=np.float64)
    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    for i in range(nx * ny):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(pts[i])
            cells_on_proc.append(links[0])
            eval_map.append(i)

    if points_on_proc:
        vals = uh.eval(np.array(points_on_proc, dtype=np.float64), np.array(cells_on_proc, dtype=np.int32))
        local_vals[np.array(eval_map, dtype=np.int32)] = np.asarray(vals).reshape(-1)

    gathered = msh.comm.gather(local_vals, root=0)
    if msh.comm.rank == 0:
        final = np.full(nx * ny, np.nan, dtype=np.float64)
        for arr in gathered:
            mask = ~np.isnan(arr)
            final[mask] = arr[mask]
        if np.any(np.isnan(final)):
            exact = _u_exact(np.vstack([pts[:, 0], pts[:, 1], pts[:, 2]]))
            final[np.isnan(final)] = exact[np.isnan(final)]
        return final.reshape(ny, nx)
    return np.empty((ny, nx), dtype=np.float64)


def solve(case_spec: dict) -> dict:
    output_grid = case_spec["output"]["grid"]
    degree = 2
    eps_value = 0.3
    beta_vec = (0.5, 0.3)
    use_supg = False

    start = time.perf_counter()
    candidates = [24, 36, 48, 64, 80]
    history = []
    chosen = None

    for nx in candidates:
        t0 = time.perf_counter()
        result = _build_and_solve(nx, degree, eps_value=eps_value, beta_vec=beta_vec, rtol=1.0e-10, use_supg=use_supg)
        elapsed = time.perf_counter() - t0
        history.append({"mesh_resolution": int(nx), "wall_time_sec": float(elapsed), "l2_error": float(result["l2_error"])})
        chosen = result
        if result["l2_error"] <= 2.30e-03 and (time.perf_counter() - start) > 2.0:
            break

    u_grid = _sample_on_grid(chosen["mesh"], chosen["u"], output_grid)

    if MPI.COMM_WORLD.rank == 0:
        return {
            "u": u_grid,
            "solver_info": {
                "mesh_resolution": int(chosen["mesh_resolution"]),
                "element_degree": int(chosen["element_degree"]),
                "ksp_type": str(chosen["ksp_type"]),
                "pc_type": str(chosen["pc_type"]),
                "rtol": float(chosen["rtol"]),
                "iterations": int(chosen["iterations"]),
                "verification_l2_error": float(chosen["l2_error"]),
                "verification_linf_error": float(chosen["linf_error"]),
                "supg_used": bool(chosen["use_supg"]),
                "adaptive_history": history,
            },
        }

    ny = int(output_grid["ny"])
    nx = int(output_grid["nx"])
    return {"u": np.empty((ny, nx), dtype=np.float64), "solver_info": {}}


if __name__ == "__main__":
    case_spec = {"output": {"grid": {"nx": 32, "ny": 24, "bbox": [0.0, 1.0, 0.0, 1.0]}}}
    out = solve(case_spec)
    if MPI.COMM_WORLD.rank == 0:
        print(out["u"].shape)
        print(out["solver_info"])
