import math
import time
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from basix.ufl import element as basix_element, mixed_element as basix_mixed_element

# ```DIAGNOSIS
# equation_type:        navier_stokes
# spatial_dim:          2
# domain_geometry:      rectangle
# unknowns:             vector+scalar
# coupling:             saddle_point
# linearity:            nonlinear
# time_dependence:      steady
# stiffness:            stiff
# dominant_physics:     mixed
# peclet_or_reynolds:   moderate
# solution_regularity:  smooth
# bc_type:              all_dirichlet
# special_notes:        pressure_pinning / manufactured_solution
# ```
# ```METHOD
# spatial_method:       fem
# element_or_basis:     Taylor-Hood_P2P1
# stabilization:        none
# time_method:          none
# nonlinear_solver:     newton
# linear_solver:        gmres
# preconditioner:       ilu
# special_treatment:    pressure_pinning
# pde_skill:            navier_stokes
# ```

ScalarType = PETSc.ScalarType


def _exact_velocity_values(x):
    pi = np.pi
    vals = np.zeros((2, x.shape[1]), dtype=np.float64)
    vals[0] = pi * np.cos(pi * x[1]) * np.sin(2.0 * pi * x[0])
    vals[1] = -2.0 * pi * np.cos(2.0 * pi * x[0]) * np.sin(pi * x[1])
    return vals


def _exact_pressure_values(x):
    pi = np.pi
    return np.sin(pi * x[0]) * np.cos(pi * x[1])


def _manufactured_ufl(msh, nu):
    x = ufl.SpatialCoordinate(msh)
    pi = ufl.pi
    u_ex = ufl.as_vector(
        [
            pi * ufl.cos(pi * x[1]) * ufl.sin(2.0 * pi * x[0]),
            -2.0 * pi * ufl.cos(2.0 * pi * x[0]) * ufl.sin(pi * x[1]),
        ]
    )
    p_ex = ufl.sin(pi * x[0]) * ufl.cos(pi * x[1])
    f = ufl.grad(u_ex) * u_ex - nu * ufl.div(ufl.grad(u_ex)) + ufl.grad(p_ex)
    return u_ex, p_ex, f


def _all_boundary(x):
    return np.ones(x.shape[1], dtype=bool)


def _build_spaces(msh, degree_u=2, degree_p=1):
    cell = msh.topology.cell_name()
    gdim = msh.geometry.dim
    vel_el = basix_element("Lagrange", cell, degree_u, shape=(gdim,))
    pre_el = basix_element("Lagrange", cell, degree_p)
    W = fem.functionspace(msh, basix_mixed_element([vel_el, pre_el]))
    V, _ = W.sub(0).collapse()
    Q, _ = W.sub(1).collapse()
    return W, V, Q


def _build_bcs(msh, W, V, Q):
    fdim = msh.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(msh, fdim, _all_boundary)

    u_bc = fem.Function(V)
    u_bc.interpolate(_exact_velocity_values)
    u_dofs = fem.locate_dofs_topological((W.sub(0), V), fdim, boundary_facets)
    bc_u = fem.dirichletbc(u_bc, u_dofs, W.sub(0))

    p0 = fem.Function(Q)
    p0.x.array[:] = 0.0
    p_dofs = fem.locate_dofs_geometrical(
        (W.sub(1), Q),
        lambda x: np.isclose(x[0], 0.0) & np.isclose(x[1], 0.0),
    )
    bcs = [bc_u]
    if len(p_dofs) > 0:
        bc_p = fem.dirichletbc(p0, p_dofs, W.sub(1))
        bcs.append(bc_p)
    return bcs


def _solve_stokes_init(msh, W, bcs, nu, ksp_type, pc_type, rtol):
    (u, p) = ufl.TrialFunctions(W)
    (v, q) = ufl.TestFunctions(W)
    _, _, f = _manufactured_ufl(msh, nu)

    a = (
        nu * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
        - ufl.inner(p, ufl.div(v)) * ufl.dx
        + ufl.inner(ufl.div(u), q) * ufl.dx
    )
    L = ufl.inner(f, v) * ufl.dx

    problem = petsc.LinearProblem(
        a,
        L,
        bcs=bcs,
        petsc_options_prefix="stokes_init_",
        petsc_options={
            "ksp_type": ksp_type,
            "ksp_rtol": rtol,
            "pc_type": pc_type,
        },
    )
    wh = problem.solve()
    wh.x.scatter_forward()
    return wh


def _solve_ns(msh, W, bcs, w0, nu, ksp_type, pc_type, rtol, max_it):
    w = fem.Function(W)
    w.x.array[:] = w0.x.array

    (u, p) = ufl.split(w)
    (v, q) = ufl.TestFunctions(W)
    _, _, f = _manufactured_ufl(msh, nu)

    F = (
        nu * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
        + ufl.inner(ufl.grad(u) * u, v) * ufl.dx
        - ufl.inner(p, ufl.div(v)) * ufl.dx
        + ufl.inner(ufl.div(u), q) * ufl.dx
        - ufl.inner(f, v) * ufl.dx
    )
    J = ufl.derivative(F, w)

    problem = petsc.NonlinearProblem(
        F,
        w,
        bcs=bcs,
        J=J,
        petsc_options_prefix="ns_",
        petsc_options={
            "snes_type": "newtonls",
            "snes_linesearch_type": "bt",
            "snes_rtol": rtol,
            "snes_atol": 1e-10,
            "snes_max_it": max_it,
            "ksp_type": ksp_type,
            "ksp_rtol": rtol,
            "pc_type": pc_type,
        },
    )
    sol = problem.solve()
    sol.x.scatter_forward()
    return sol


def _compute_errors(msh, uh, ph):
    V = uh.function_space
    Q = ph.function_space
    u_ex_fun = fem.Function(V)
    p_ex_fun = fem.Function(Q)
    u_ex_fun.interpolate(_exact_velocity_values)
    p_ex_fun.interpolate(_exact_pressure_values)

    e_u_form = fem.form(ufl.inner(uh - u_ex_fun, uh - u_ex_fun) * ufl.dx)
    e_p_form = fem.form((ph - p_ex_fun) * (ph - p_ex_fun) * ufl.dx)
    n_u_form = fem.form(ufl.inner(u_ex_fun, u_ex_fun) * ufl.dx)
    n_p_form = fem.form(p_ex_fun * p_ex_fun * ufl.dx)

    e_u = math.sqrt(msh.comm.allreduce(fem.assemble_scalar(e_u_form), op=MPI.SUM))
    e_p = math.sqrt(msh.comm.allreduce(fem.assemble_scalar(e_p_form), op=MPI.SUM))
    n_u = math.sqrt(msh.comm.allreduce(fem.assemble_scalar(n_u_form), op=MPI.SUM))
    n_p = math.sqrt(msh.comm.allreduce(fem.assemble_scalar(n_p_form), op=MPI.SUM))

    return {
        "l2_error_u": e_u,
        "l2_error_p": e_p,
        "rel_l2_error_u": e_u / max(n_u, 1e-15),
        "rel_l2_error_p": e_p / max(n_p, 1e-15),
    }


def _sample_velocity_magnitude(u_fun, grid):
    msh = u_fun.function_space.mesh
    nx = int(grid["nx"])
    ny = int(grid["ny"])
    xmin, xmax, ymin, ymax = grid["bbox"]

    xs = np.linspace(xmin, xmax, nx)
    ys = np.linspace(ymin, ymax, ny)
    XX, YY = np.meshgrid(xs, ys, indexing="xy")
    pts = np.column_stack([XX.ravel(), YY.ravel(), np.zeros(nx * ny, dtype=np.float64)])

    bb_tree = geometry.bb_tree(msh, msh.topology.dim)
    candidates = geometry.compute_collisions_points(bb_tree, pts)
    colliding = geometry.compute_colliding_cells(msh, candidates, pts)

    vals_local = np.zeros((pts.shape[0], msh.geometry.dim), dtype=np.float64)
    mask_local = np.zeros(pts.shape[0], dtype=np.int32)

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
        vals_local[np.array(ids, dtype=np.int32)] = np.real(vals)
        mask_local[np.array(ids, dtype=np.int32)] = 1

    comm = msh.comm
    vals_global = np.zeros_like(vals_local)
    mask_global = np.zeros_like(mask_local)
    comm.Allreduce(vals_local, vals_global, op=MPI.SUM)
    comm.Allreduce(mask_local, mask_global, op=MPI.SUM)

    mag = np.linalg.norm(vals_global, axis=1).reshape(ny, nx)
    return mag


def solve(case_spec: dict) -> dict:
    t0 = time.perf_counter()
    comm = MPI.COMM_WORLD

    nu = ScalarType(float(case_spec.get("viscosity", case_spec.get("pde", {}).get("nu", 0.2))))
    mesh_resolution = int(case_spec.get("mesh_resolution", 40))
    degree_u = int(case_spec.get("degree_u", 2))
    degree_p = int(case_spec.get("degree_p", max(1, degree_u - 1)))
    newton_rtol = float(case_spec.get("newton_rtol", 1e-8))
    newton_max_it = int(case_spec.get("newton_max_it", 25))

    msh = mesh.create_unit_square(comm, mesh_resolution, mesh_resolution, cell_type=mesh.CellType.triangle)
    W, V, Q = _build_spaces(msh, degree_u=degree_u, degree_p=degree_p)
    bcs = _build_bcs(msh, W, V, Q)

    ksp_type = "gmres"
    pc_type = "ilu"
    iterations = 0
    nonlinear_iterations = [0]

    try:
        w0 = _solve_stokes_init(msh, W, bcs, nu, ksp_type, pc_type, 1e-9)
        wh = _solve_ns(msh, W, bcs, w0, nu, ksp_type, pc_type, newton_rtol, newton_max_it)
    except Exception:
        ksp_type = "preonly"
        pc_type = "lu"
        w0 = _solve_stokes_init(msh, W, bcs, nu, ksp_type, pc_type, 1e-12)
        wh = _solve_ns(msh, W, bcs, w0, nu, ksp_type, pc_type, min(newton_rtol, 1e-9), max(newton_max_it, 30))

    uh, _ = wh.sub(0).collapse()
    ph, _ = wh.sub(1).collapse()
    errors = _compute_errors(msh, uh, ph)

    u_grid = _sample_velocity_magnitude(uh, case_spec["output"]["grid"])

    solver_info = {
        "mesh_resolution": mesh_resolution,
        "element_degree": degree_u,
        "ksp_type": ksp_type,
        "pc_type": pc_type,
        "rtol": newton_rtol,
        "iterations": int(iterations),
        "nonlinear_iterations": nonlinear_iterations,
        "l2_error_u": float(errors["l2_error_u"]),
        "l2_error_p": float(errors["l2_error_p"]),
        "rel_l2_error_u": float(errors["rel_l2_error_u"]),
        "rel_l2_error_p": float(errors["rel_l2_error_p"]),
        "wall_time_sec": float(time.perf_counter() - t0),
    }

    return {"u": u_grid, "solver_info": solver_info}


if __name__ == "__main__":
    case_spec = {
        "viscosity": 0.2,
        "output": {"grid": {"nx": 32, "ny": 32, "bbox": [0.0, 1.0, 0.0, 1.0]}},
    }
    result = solve(case_spec)
    if MPI.COMM_WORLD.rank == 0:
        print(result["u"].shape)
        print(result["solver_info"])
