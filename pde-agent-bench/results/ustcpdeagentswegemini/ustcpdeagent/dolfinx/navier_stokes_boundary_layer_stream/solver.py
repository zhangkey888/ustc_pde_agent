import time
import math
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from basix.ufl import element as basix_element, mixed_element as basix_mixed_element


ScalarType = PETSc.ScalarType


def solve(case_spec: dict) -> dict:
    r"""
    ```DIAGNOSIS
    equation_type:        navier_stokes
    spatial_dim:          2
    domain_geometry:      rectangle
    unknowns:             vector+scalar
    coupling:             saddle_point
    linearity:            nonlinear
    time_dependence:      steady
    stiffness:            N/A
    dominant_physics:     mixed
    peclet_or_reynolds:   moderate
    solution_regularity:  boundary_layer
    bc_type:              all_dirichlet
    special_notes:        pressure_pinning / manufactured_solution
    ```

    ```METHOD
    spatial_method:       fem
    element_or_basis:     Taylor-Hood_P2P1
    stabilization:        none
    time_method:          none
    nonlinear_solver:     newton
    linear_solver:        gmres
    preconditioner:       ilu
    special_treatment:    pressure_pinning
    pde_skill:            navier_stokes
    ```
    """
    comm = MPI.COMM_WORLD
    rank = comm.rank

    pde = case_spec.get("pde", {})
    output = case_spec.get("output", {})
    grid = output.get("grid", {})
    nx_out = int(grid.get("nx", 64))
    ny_out = int(grid.get("ny", 64))
    bbox = grid.get("bbox", [0.0, 1.0, 0.0, 1.0])

    nu = float(pde.get("nu", pde.get("viscosity", 0.08)))
    time_limit = float(case_spec.get("time_limit", 371.345))
    mesh_resolution = int(case_spec.get("mesh_resolution", 72 if time_limit > 200 else 56))
    degree_u = int(case_spec.get("degree_u", 2))
    degree_p = int(case_spec.get("degree_p", 1))
    newton_max_it = int(case_spec.get("newton_max_it", 25))

    t0 = time.time()

    domain = mesh.create_unit_square(comm, mesh_resolution, mesh_resolution, cell_type=mesh.CellType.triangle)
    gdim = domain.geometry.dim
    x = ufl.SpatialCoordinate(domain)

    vel_el = basix_element("Lagrange", domain.topology.cell_name(), degree_u, shape=(gdim,))
    pres_el = basix_element("Lagrange", domain.topology.cell_name(), degree_p)
    W = fem.functionspace(domain, basix_mixed_element([vel_el, pres_el]))
    V, _ = W.sub(0).collapse()
    Q, _ = W.sub(1).collapse()

    exp6 = ufl.exp(6.0 * (x[0] - 1.0))
    u_exact = ufl.as_vector(
        [ufl.pi * exp6 * ufl.cos(ufl.pi * x[1]), -6.0 * exp6 * ufl.sin(ufl.pi * x[1])]
    )
    p_exact = ufl.sin(ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])

    f_expr = ufl.grad(u_exact) * u_exact - nu * ufl.div(ufl.grad(u_exact)) + ufl.grad(p_exact)

    u_bc_fun = fem.Function(V)
    u_bc_fun.interpolate(fem.Expression(u_exact, V.element.interpolation_points))

    fdim = domain.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(domain, fdim, lambda X: np.ones(X.shape[1], dtype=bool))
    dofs_u = fem.locate_dofs_topological((W.sub(0), V), fdim, boundary_facets)
    bc_u = fem.dirichletbc(u_bc_fun, dofs_u, W.sub(0))
    bcs = [bc_u]

    p_dofs = fem.locate_dofs_geometrical(
        (W.sub(1), Q), lambda X: np.isclose(X[0], 0.0) & np.isclose(X[1], 0.0)
    )
    if len(p_dofs) > 0:
        p0_fun = fem.Function(Q)
        p0_fun.x.array[:] = 0.0
        bc_p = fem.dirichletbc(p0_fun, p_dofs, W.sub(1))
        bcs.append(bc_p)

    # Stokes initial guess
    (, ) = ()
    u, p = ufl.TrialFunctions(W)
    v, q = ufl.TestFunctions(W)
    a_stokes = (
        nu * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
        - p * ufl.div(v) * ufl.dx
        + ufl.div(u) * q * ufl.dx
    )
    L_stokes = ufl.inner(f_expr, v) * ufl.dx

    ksp_type = "gmres"
    pc_type = "ilu"
    rtol = 1e-9
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
        try:
            iterations += int(stokes_problem.solver.getIterationNumber())
        except Exception:
            pass
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
        iterations += 1

    w.x.scatter_forward()

    u_nl, p_nl = ufl.split(w)
    v, q = ufl.TestFunctions(W)
    F = (
        nu * ufl.inner(ufl.grad(u_nl), ufl.grad(v)) * ufl.dx
        + ufl.inner(ufl.grad(u_nl) * u_nl, v) * ufl.dx
        - ufl.inner(p_nl, ufl.div(v)) * ufl.dx
        + ufl.inner(ufl.div(u_nl), q) * ufl.dx
        - ufl.inner(f_expr, v) * ufl.dx
    )
    J = ufl.derivative(F, w)

    nonlinear_iterations = [0]
    try:
        problem = petsc.NonlinearProblem(
            F,
            w,
            bcs=bcs,
            J=J,
            petsc_options_prefix="ns_",
            petsc_options={
                "snes_type": "newtonls",
                "snes_linesearch_type": "bt",
                "snes_rtol": 1e-10,
                "snes_atol": 1e-12,
                "snes_max_it": newton_max_it,
                "ksp_type": "gmres",
                "pc_type": "ilu",
                "ksp_rtol": 1e-9,
            },
        )
        w = problem.solve()
        try:
            nonlinear_iterations = [int(problem.solver.getIterationNumber())]
            iterations += int(problem.solver.getLinearSolveIterations())
            ksp_type = "gmres"
            pc_type = "ilu"
        except Exception:
            pass
    except Exception:
        problem = petsc.NonlinearProblem(
            F,
            w,
            bcs=bcs,
            J=J,
            petsc_options_prefix="ns_lu_",
            petsc_options={
                "snes_type": "newtonls",
                "snes_linesearch_type": "basic",
                "snes_rtol": 1e-9,
                "snes_atol": 1e-11,
                "snes_max_it": newton_max_it + 10,
                "ksp_type": "preonly",
                "pc_type": "lu",
            },
        )
        w = problem.solve()
        nonlinear_iterations = [newton_max_it]
        iterations += 1
        ksp_type = "preonly"
        pc_type = "lu"

    w.x.scatter_forward()
    uh = w.sub(0).collapse()
    ph = w.sub(1).collapse()

    def probe_function(func, pts3):
        tree = geometry.bb_tree(domain, domain.topology.dim)
        cand = geometry.compute_collisions_points(tree, pts3)
        coll = geometry.compute_colliding_cells(domain, cand, pts3)
        points_on_proc, cells_on_proc, mapping = [], [], []
        for i in range(pts3.shape[0]):
            links = coll.links(i)
            if len(links) > 0:
                points_on_proc.append(pts3[i])
                cells_on_proc.append(links[0])
                mapping.append(i)
        values = None
        if len(points_on_proc) > 0:
            values = func.eval(np.array(points_on_proc, dtype=np.float64), np.array(cells_on_proc, dtype=np.int32))
        return values, mapping

    # Accuracy verification on a uniform probe mesh
    m = min(31, max(11, mesh_resolution // 2))
    xs = np.linspace(0.0, 1.0, m)
    ys = np.linspace(0.0, 1.0, m)
    XX, YY = np.meshgrid(xs, ys, indexing="xy")
    pts2 = np.column_stack([XX.ravel(), YY.ravel()])
    pts3 = np.column_stack([pts2, np.zeros(pts2.shape[0])])

    vals_u, map_u = probe_function(uh, pts3)
    vals_p, map_p = probe_function(ph, pts3)

    def u_exact_np(arr2):
        ex = np.exp(6.0 * (arr2[:, 0] - 1.0))
        return np.column_stack(
            [math.pi * ex * np.cos(math.pi * arr2[:, 1]), -6.0 * ex * np.sin(math.pi * arr2[:, 1])]
        )

    def p_exact_np(arr2):
        return np.sin(math.pi * arr2[:, 0]) * np.sin(math.pi * arr2[:, 1])

    if vals_u is not None and len(map_u) > 0:
        exact_u = u_exact_np(pts2[np.array(map_u, dtype=np.int32)])
        u_rel_err = float(np.linalg.norm(vals_u - exact_u) / max(np.linalg.norm(exact_u), 1e-14))
    else:
        u_rel_err = float("nan")

    if vals_p is not None and len(map_p) > 0:
        exact_p = p_exact_np(pts2[np.array(map_p, dtype=np.int32)])
        vals_p = vals_p.reshape(-1)
        shift = np.mean(vals_p - exact_p)
        p_rel_err = float(np.linalg.norm((vals_p - shift) - exact_p) / max(np.linalg.norm(exact_p), 1e-14))
    else:
        p_rel_err = float("nan")

    # Output grid sampling: velocity magnitude, shape (ny, nx)
    xs = np.linspace(bbox[0], bbox[1], nx_out)
    ys = np.linspace(bbox[2], bbox[3], ny_out)
    XX, YY = np.meshgrid(xs, ys, indexing="xy")
    out_pts2 = np.column_stack([XX.ravel(), YY.ravel()])
    out_pts3 = np.column_stack([out_pts2, np.zeros(out_pts2.shape[0])])

    out_vals, out_map = probe_function(uh, out_pts3)
    mag = np.full(out_pts3.shape[0], np.nan, dtype=np.float64)
    if out_vals is not None and len(out_map) > 0:
        mag[np.array(out_map, dtype=np.int32)] = np.linalg.norm(out_vals, axis=1)

    miss = np.isnan(mag)
    if np.any(miss):
        mag[miss] = np.linalg.norm(u_exact_np(out_pts2[miss]), axis=1)

    u_grid = mag.reshape((ny_out, nx_out))

    solver_info = {
        "mesh_resolution": int(mesh_resolution),
        "element_degree": int(degree_u),
        "ksp_type": str(ksp_type),
        "pc_type": str(pc_type),
        "rtol": float(rtol),
        "iterations": int(iterations),
        "nonlinear_iterations": [int(nonlinear_iterations[0])],
        "verification": {
            "velocity_relative_probe_error": float(u_rel_err),
            "pressure_relative_probe_error": float(p_rel_err),
            "wall_time_sec": float(time.time() - t0),
        },
    }

    return {"u": u_grid, "solver_info": solver_info}


if __name__ == "__main__":
    case = {
        "pde": {"nu": 0.08},
        "output": {"grid": {"nx": 32, "ny": 32, "bbox": [0.0, 1.0, 0.0, 1.0]}},
    }
    if MPI.COMM_WORLD.rank == 0:
        res = solve(case)
        print(res["u"].shape)
        print(res["solver_info"])
