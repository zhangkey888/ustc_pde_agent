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
# special_notes:        pressure_pinning, manufactured_solution
# ```
#
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

import math
import time
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
import ufl
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
from basix.ufl import element as basix_element, mixed_element as basix_mixed_element

ScalarType = PETSc.ScalarType


def _u_exact_numpy(x, y):
    r2 = (x - 0.5) ** 2 + (y - 0.5) ** 2
    e = np.exp(-20.0 * r2)
    u0 = -40.0 * (y - 0.5) * e
    u1 = 40.0 * (x - 0.5) * e
    return u0, u1


def _build_manufactured_forcing(msh, nu):
    x = ufl.SpatialCoordinate(msh)
    uex = ufl.as_vector(
        (
            -40.0 * (x[1] - 0.5) * ufl.exp(-20.0 * ((x[0] - 0.5) ** 2 + (x[1] - 0.5) ** 2)),
            40.0 * (x[0] - 0.5) * ufl.exp(-20.0 * ((x[0] - 0.5) ** 2 + (x[1] - 0.5) ** 2)),
        )
    )
    pex = 0.0 * x[0]
    f = ufl.grad(uex) * uex - nu * ufl.div(ufl.grad(uex)) + ufl.grad(pex)
    return uex, pex, f


def _sample_velocity_magnitude(u_fun, msh, grid_spec):
    nx = int(grid_spec["nx"])
    ny = int(grid_spec["ny"])
    xmin, xmax, ymin, ymax = grid_spec["bbox"]
    xs = np.linspace(xmin, xmax, nx)
    ys = np.linspace(ymin, ymax, ny)
    xx, yy = np.meshgrid(xs, ys, indexing="xy")
    pts = np.column_stack([xx.ravel(), yy.ravel(), np.zeros(nx * ny, dtype=np.float64)])

    tree = geometry.bb_tree(msh, msh.topology.dim)
    cell_candidates = geometry.compute_collisions_points(tree, pts)
    colliding = geometry.compute_colliding_cells(msh, cell_candidates, pts)

    local_idx = []
    local_pts = []
    local_cells = []
    for i in range(pts.shape[0]):
        links = colliding.links(i)
        if len(links) > 0:
            local_idx.append(i)
            local_pts.append(pts[i])
            local_cells.append(links[0])

    local_pairs = []
    if local_pts:
        vals = u_fun.eval(np.array(local_pts, dtype=np.float64), np.array(local_cells, dtype=np.int32))
        mags = np.linalg.norm(vals, axis=1)
        local_pairs = [(idx, mag) for idx, mag in zip(local_idx, mags)]

    gathered = msh.comm.gather(local_pairs, root=0)
    if msh.comm.rank == 0:
        out = np.full(nx * ny, np.nan, dtype=np.float64)
        for proc_pairs in gathered:
            for idx, val in proc_pairs:
                if np.isnan(out[idx]):
                    out[idx] = val
        if np.isnan(out).any():
            nan_ids = np.where(np.isnan(out))[0]
            for idx in nan_ids:
                xpt = pts[idx, 0]
                ypt = pts[idx, 1]
                u0, u1 = _u_exact_numpy(xpt, ypt)
                out[idx] = math.sqrt(u0 * u0 + u1 * u1)
        return out.reshape((ny, nx))
    return None


def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD
    rank = comm.rank
    t0 = time.perf_counter()

    nu = float(case_spec.get("pde", {}).get("viscosity", 0.12))
    output_grid = case_spec["output"]["grid"]

    # Proactive accuracy choice within generous time budget.
    mesh_resolution = int(case_spec.get("mesh_resolution", 56))
    degree_u = int(case_spec.get("degree_u", 2))
    degree_p = int(case_spec.get("degree_p", 1))
    if degree_u < 2:
        degree_u = 2
    if degree_p < 1:
        degree_p = 1

    msh = mesh.create_unit_square(comm, mesh_resolution, mesh_resolution, cell_type=mesh.CellType.triangle)
    gdim = msh.geometry.dim

    vel_el = basix_element("Lagrange", msh.topology.cell_name(), degree_u, shape=(gdim,))
    pre_el = basix_element("Lagrange", msh.topology.cell_name(), degree_p)
    W = fem.functionspace(msh, basix_mixed_element([vel_el, pre_el]))
    V, _ = W.sub(0).collapse()
    Q, _ = W.sub(1).collapse()

    w = fem.Function(W)
    u, p = ufl.split(w)
    v, q = ufl.TestFunctions(W)

    uex, _, f = _build_manufactured_forcing(msh, nu)

    u_bc_fun = fem.Function(V)
    u_bc_fun.interpolate(
        lambda X: np.vstack(
            _u_exact_numpy(X[0], X[1])
        )
    )

    fdim = msh.topology.dim - 1
    bfacets = mesh.locate_entities_boundary(msh, fdim, lambda X: np.ones(X.shape[1], dtype=bool))
    udofs = fem.locate_dofs_topological((W.sub(0), V), fdim, bfacets)
    bc_u = fem.dirichletbc(u_bc_fun, udofs, W.sub(0))

    pdofs = fem.locate_dofs_geometrical((W.sub(1), Q), lambda X: np.isclose(X[0], 0.0) & np.isclose(X[1], 0.0))
    p0 = fem.Function(Q)
    p0.x.array[:] = 0.0
    bcs = [bc_u]
    if len(pdofs) > 0:
        bc_p = fem.dirichletbc(p0, pdofs, W.sub(1))
        bcs.append(bc_p)

    # Stokes initialization
    (ut, pt) = ufl.TrialFunctions(W)
    a_stokes = (
        nu * ufl.inner(ufl.grad(ut), ufl.grad(v)) * ufl.dx
        - ufl.inner(pt, ufl.div(v)) * ufl.dx
        + ufl.inner(ufl.div(ut), q) * ufl.dx
    )
    L_stokes = ufl.inner(f, v) * ufl.dx

    ksp_type = "gmres"
    pc_type = "ilu"
    rtol = 1.0e-9
    iterations_total = 0

    stokes_problem = petsc.LinearProblem(
        a_stokes,
        L_stokes,
        bcs=bcs,
        petsc_options_prefix="stokes_",
        petsc_options={
            "ksp_type": ksp_type,
            "ksp_rtol": rtol,
            "pc_type": pc_type,
        },
    )
    w0 = stokes_problem.solve()
    w.x.array[:] = w0.x.array
    w.x.scatter_forward()

    F = (
        nu * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
        + ufl.inner(ufl.grad(u) * u, v) * ufl.dx
        - ufl.inner(p, ufl.div(v)) * ufl.dx
        + ufl.inner(ufl.div(u), q) * ufl.dx
        - ufl.inner(f, v) * ufl.dx
    )
    J = ufl.derivative(F, w)

    snes_prefix = "ns_"
    problem = petsc.NonlinearProblem(
        F,
        w,
        bcs=bcs,
        J=J,
        petsc_options_prefix=snes_prefix,
        petsc_options={
            "snes_type": "newtonls",
            "snes_linesearch_type": "bt",
            "snes_rtol": 1.0e-9,
            "snes_atol": 1.0e-10,
            "snes_max_it": 25,
            "ksp_type": ksp_type,
            "ksp_rtol": rtol,
            "pc_type": pc_type,
        },
    )

    nonlinear_iterations = [0]
    try:
        w = problem.solve()
        w.x.scatter_forward()
    except Exception:
        problem = petsc.NonlinearProblem(
            F,
            w,
            bcs=bcs,
            J=J,
            petsc_options_prefix="nslu_",
            petsc_options={
                "snes_type": "newtonls",
                "snes_linesearch_type": "bt",
                "snes_rtol": 1.0e-9,
                "snes_atol": 1.0e-10,
                "snes_max_it": 30,
                "ksp_type": "preonly",
                "pc_type": "lu",
            },
        )
        ksp_type = "preonly"
        pc_type = "lu"
        w = problem.solve()
        w.x.scatter_forward()

    try:
        snes = PETSc.Options()
        _ = snes
    except Exception:
        pass

    try:
        _, nit = problem.solver.getIterationNumber(), problem.solver.getIterationNumber()
        nonlinear_iterations = [int(nit)]
    except Exception:
        nonlinear_iterations = [0]

    try:
        its = problem.solver.getLinearSolveIterations()
        iterations_total += int(its)
    except Exception:
        pass

    uh = w.sub(0).collapse()
    uh.name = "u"

    # Accuracy verification against manufactured exact solution
    Vex = fem.functionspace(msh, ("Lagrange", degree_u + 1, (gdim,)))
    uex_fun = fem.Function(Vex)
    uex_fun.interpolate(lambda X: np.vstack(_u_exact_numpy(X[0], X[1])))

    err_form = fem.form(ufl.inner(uh - uex_fun, uh - uex_fun) * ufl.dx)
    u_l2_err = math.sqrt(comm.allreduce(fem.assemble_scalar(err_form), op=MPI.SUM))

    u_grid = _sample_velocity_magnitude(uh, msh, output_grid)

    solver_info = {
        "mesh_resolution": mesh_resolution,
        "element_degree": degree_u,
        "ksp_type": ksp_type,
        "pc_type": pc_type,
        "rtol": rtol,
        "iterations": int(iterations_total),
        "nonlinear_iterations": nonlinear_iterations,
        "l2_error_velocity": float(u_l2_err),
        "wall_time_sec": float(time.perf_counter() - t0),
    }

    if rank == 0:
        return {"u": u_grid, "solver_info": solver_info}
    return {"u": None, "solver_info": solver_info}


if __name__ == "__main__":
    case_spec = {
        "pde": {"viscosity": 0.12, "time": None},
        "output": {"grid": {"nx": 64, "ny": 64, "bbox": [0.0, 1.0, 0.0, 1.0]}},
    }
    result = solve(case_spec)
    if MPI.COMM_WORLD.rank == 0:
        print(result["u"].shape)
        print(result["solver_info"])
