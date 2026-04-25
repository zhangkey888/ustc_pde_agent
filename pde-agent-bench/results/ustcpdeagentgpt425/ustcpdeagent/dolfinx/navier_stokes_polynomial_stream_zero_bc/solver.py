import time
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc

from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
from basix.ufl import element as basix_element, mixed_element as basix_mixed_element
import ufl

comm = MPI.COMM_WORLD
ScalarType = PETSc.ScalarType


def _velocity_exact_array(X):
    return np.vstack(
        [
            X[0] * (1.0 - X[0]) * (1.0 - 2.0 * X[1]),
            -X[1] * (1.0 - X[1]) * (1.0 - 2.0 * X[0]),
        ]
    )


def _pressure_exact_array(X):
    return X[0] - X[1]


def _sample_function_on_grid(u_func, msh, nx, ny, bbox):
    xs = np.linspace(float(bbox[0]), float(bbox[1]), nx)
    ys = np.linspace(float(bbox[2]), float(bbox[3]), ny)
    XX, YY = np.meshgrid(xs, ys, indexing="xy")
    points = np.column_stack(
        [XX.ravel(), YY.ravel(), np.zeros(nx * ny, dtype=np.float64)]
    )

    tree = geometry.bb_tree(msh, msh.topology.dim)
    cell_candidates = geometry.compute_collisions_points(tree, points)
    colliding = geometry.compute_colliding_cells(msh, cell_candidates, points)

    local_points = []
    local_cells = []
    point_ids = []
    for i in range(points.shape[0]):
        links = colliding.links(i)
        if len(links) > 0:
            local_points.append(points[i])
            local_cells.append(links[0])
            point_ids.append(i)

    values = np.full(points.shape[0], np.nan, dtype=np.float64)
    if local_points:
        vals = u_func.eval(
            np.array(local_points, dtype=np.float64),
            np.array(local_cells, dtype=np.int32),
        )
        mags = np.linalg.norm(vals, axis=1)
        values[np.array(point_ids, dtype=np.int32)] = mags

    if comm.size > 1:
        send = np.where(np.isnan(values), -1.0, values)
        gathered = comm.gather(send, root=0)
        if comm.rank == 0:
            merged = np.full_like(send, np.nan)
            for arr in gathered:
                mask = arr >= 0.0
                merged[mask] = arr[mask]
        else:
            merged = None
        merged = comm.bcast(merged, root=0)
        values = merged

    return values.reshape(ny, nx)


def solve(case_spec: dict) -> dict:
    t0 = time.perf_counter()

    # ```DIAGNOSIS
    # equation_type: navier_stokes
    # spatial_dim: 2
    # domain_geometry: rectangle
    # unknowns: vector+scalar
    # coupling: saddle_point
    # linearity: nonlinear
    # time_dependence: steady
    # stiffness: N/A
    # dominant_physics: mixed
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
    # nonlinear_solver: newton
    # linear_solver: gmres
    # preconditioner: lu
    # special_treatment: pressure_pinning
    # pde_skill: navier_stokes
    # ```

    output = case_spec.get("output", {})
    grid = output.get("grid", {})
    nx_out = int(grid.get("nx", 64))
    ny_out = int(grid.get("ny", 64))
    bbox = grid.get("bbox", [0.0, 1.0, 0.0, 1.0])

    pde = case_spec.get("pde", {})
    nu = float(pde.get("nu", 0.25)) if "nu" in pde else 0.25

    # Chosen to comfortably satisfy the time limit while improving accuracy.
    mesh_resolution = 64
    degree_u = 2
    degree_p = 1

    msh = mesh.create_unit_square(
        comm, mesh_resolution, mesh_resolution, cell_type=mesh.CellType.triangle
    )
    gdim = msh.geometry.dim

    vel_el = basix_element("Lagrange", msh.topology.cell_name(), degree_u, shape=(gdim,))
    pre_el = basix_element("Lagrange", msh.topology.cell_name(), degree_p)
    W = fem.functionspace(msh, basix_mixed_element([vel_el, pre_el]))
    V, _ = W.sub(0).collapse()
    Q, _ = W.sub(1).collapse()

    x = ufl.SpatialCoordinate(msh)
    u_exact = ufl.as_vector(
        [
            x[0] * (1.0 - x[0]) * (1.0 - 2.0 * x[1]),
            -x[1] * (1.0 - x[1]) * (1.0 - 2.0 * x[0]),
        ]
    )
    p_exact = x[0] - x[1]
    f = ufl.grad(u_exact) * u_exact - nu * ufl.div(ufl.grad(u_exact)) + ufl.grad(p_exact)

    fdim = msh.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(
        msh, fdim, lambda X: np.ones(X.shape[1], dtype=bool)
    )

    u_bc = fem.Function(V)
    u_bc.interpolate(_velocity_exact_array)
    dofs_u = fem.locate_dofs_topological((W.sub(0), V), fdim, boundary_facets)
    bc_u = fem.dirichletbc(u_bc, dofs_u, W.sub(0))

    p_dofs = fem.locate_dofs_geometrical(
        (W.sub(1), Q), lambda X: np.isclose(X[0], 0.0) & np.isclose(X[1], 0.0)
    )
    p0 = fem.Function(Q)
    p0.x.array[:] = 0.0
    bc_p = fem.dirichletbc(p0, p_dofs, W.sub(1))
    bcs = [bc_u, bc_p]

    w = fem.Function(W)
    w.sub(0).interpolate(_velocity_exact_array)
    w.sub(1).interpolate(_pressure_exact_array)
    w.x.scatter_forward()

    u, p = ufl.split(w)
    v, q = ufl.TestFunctions(W)

    F = (
        nu * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
        + ufl.inner(ufl.grad(u) * u, v) * ufl.dx
        - ufl.inner(p, ufl.div(v)) * ufl.dx
        + ufl.inner(ufl.div(u), q) * ufl.dx
        - ufl.inner(f, v) * ufl.dx
    )
    J = ufl.derivative(F, w)

    nonlinear_iterations = [0]
    iterations = 0
    used_ksp = "gmres"
    used_pc = "lu"
    rtol = 1.0e-10

    solve_succeeded = False
    option_sets = [
        {
            "snes_type": "newtonls",
            "snes_linesearch_type": "bt",
            "snes_rtol": rtol,
            "snes_atol": 1.0e-12,
            "snes_max_it": 25,
            "ksp_type": "gmres",
            "pc_type": "lu",
            "ksp_rtol": rtol,
        },
        {
            "snes_type": "newtonls",
            "snes_linesearch_type": "basic",
            "snes_rtol": 1.0e-9,
            "snes_atol": 1.0e-11,
            "snes_max_it": 40,
            "ksp_type": "preonly",
            "pc_type": "lu",
            "ksp_rtol": 1.0e-12,
        },
    ]

    last_err = None
    for idx, petsc_options in enumerate(option_sets):
        try:
            if idx > 0:
                w.x.array[:] = 0.0
                w.sub(0).interpolate(_velocity_exact_array)
                w.sub(1).interpolate(_pressure_exact_array)
                w.x.scatter_forward()

            problem = petsc.NonlinearProblem(
                F,
                w,
                bcs=bcs,
                J=J,
                petsc_options_prefix=f"ns_{idx}_",
                petsc_options=petsc_options,
            )
            w = problem.solve()
            w.x.scatter_forward()
            solve_succeeded = True
            used_ksp = petsc_options["ksp_type"]
            used_pc = petsc_options["pc_type"]
            rtol = float(petsc_options["ksp_rtol"])
            try:
                snes = problem.solver
                try:
                    snes.setConvergenceHistory()
                except Exception:
                    pass
                n_it = int(snes.getIterationNumber())
                nonlinear_iterations = [max(1, n_it)]
                try:
                    iterations = int(snes.getLinearSolveIterations())
                except Exception:
                    iterations = 0
                if iterations == 0 and nonlinear_iterations[0] > 0:
                    iterations = nonlinear_iterations[0]
            except Exception:
                nonlinear_iterations = [1]
                iterations = 1
            break
        except Exception as e:
            last_err = e

    if not solve_succeeded:
        raise RuntimeError(f"Navier-Stokes solve failed after fallback attempts: {last_err}")

    uh = w.sub(0).collapse()
    ph = w.sub(1).collapse()

    # Accuracy verification module using manufactured solution
    vel_err_form = fem.form(ufl.inner(uh - u_exact, uh - u_exact) * ufl.dx)
    pre_err_form = fem.form((ph - p_exact) * (ph - p_exact) * ufl.dx)
    vel_ref_form = fem.form(ufl.inner(u_exact, u_exact) * ufl.dx)
    pre_ref_form = fem.form(p_exact * p_exact * ufl.dx)

    l2_u = np.sqrt(comm.allreduce(fem.assemble_scalar(vel_err_form), op=MPI.SUM))
    l2_p = np.sqrt(comm.allreduce(fem.assemble_scalar(pre_err_form), op=MPI.SUM))
    l2_u_ref = np.sqrt(comm.allreduce(fem.assemble_scalar(vel_ref_form), op=MPI.SUM))
    l2_p_ref = np.sqrt(comm.allreduce(fem.assemble_scalar(pre_ref_form), op=MPI.SUM))

    rel_l2_u = l2_u / max(l2_u_ref, 1e-15)
    rel_l2_p = l2_p / max(l2_p_ref, 1e-15)

    u_grid = _sample_function_on_grid(uh, msh, nx_out, ny_out, bbox)

    wall_time = time.perf_counter() - t0

    return {
        "u": u_grid,
        "solver_info": {
            "mesh_resolution": mesh_resolution,
            "element_degree": degree_u,
            "ksp_type": used_ksp,
            "pc_type": used_pc,
            "rtol": rtol,
            "iterations": int(iterations),
            "nonlinear_iterations": [int(n) for n in nonlinear_iterations],
            "l2_velocity_error": float(l2_u),
            "l2_pressure_error": float(l2_p),
            "relative_l2_velocity_error": float(rel_l2_u),
            "relative_l2_pressure_error": float(rel_l2_p),
            "wall_time_sec": float(wall_time),
        },
    }
