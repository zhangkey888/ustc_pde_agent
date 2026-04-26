import time
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
import ufl
from basix.ufl import element as basix_element, mixed_element as basix_mixed_element
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc

ScalarType = PETSc.ScalarType


def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD
    t0 = time.time()

    nu = float(case_spec.get("pde", {}).get("nu", case_spec.get("viscosity", 0.1)))
    # Adaptive default tuned for accuracy under generous time budget
    mesh_resolution = int(case_spec.get("solver", {}).get("mesh_resolution", 40))
    degree_u = int(case_spec.get("solver", {}).get("degree_u", 2))
    degree_p = int(case_spec.get("solver", {}).get("degree_p", 1))
    newton_rtol = float(case_spec.get("solver", {}).get("newton_rtol", 1e-9))
    newton_max_it = int(case_spec.get("solver", {}).get("newton_max_it", 30))

    msh = mesh.create_unit_square(comm, mesh_resolution, mesh_resolution, cell_type=mesh.CellType.triangle)
    gdim = msh.geometry.dim
    cell_name = msh.topology.cell_name()

    vel_el = basix_element("Lagrange", cell_name, degree_u, shape=(gdim,))
    pre_el = basix_element("Lagrange", cell_name, degree_p)
    W = fem.functionspace(msh, basix_mixed_element([vel_el, pre_el]))
    V, _ = W.sub(0).collapse()
    Q, _ = W.sub(1).collapse()

    x = ufl.SpatialCoordinate(msh)
    pi = np.pi

    u_exact_ufl = ufl.as_vector(
        [
            ufl.pi * ufl.cos(ufl.pi * x[1]) * ufl.sin(ufl.pi * x[0]),
            -ufl.pi * ufl.cos(ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1]),
        ]
    )
    p_exact_ufl = 0.0 * x[0]

    conv_exact = ufl.grad(u_exact_ufl) * u_exact_ufl
    f_ufl = conv_exact - nu * ufl.div(ufl.grad(u_exact_ufl)) + ufl.grad(p_exact_ufl)

    def u_exact_callable(X):
        return np.vstack(
            [
                pi * np.cos(pi * X[1]) * np.sin(pi * X[0]),
                -pi * np.cos(pi * X[0]) * np.sin(pi * X[1]),
            ]
        )

    u_bc_fun = fem.Function(V)
    u_bc_fun.interpolate(u_exact_callable)

    fdim = msh.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(msh, fdim, lambda X: np.ones(X.shape[1], dtype=bool))
    dofs_u = fem.locate_dofs_topological((W.sub(0), V), fdim, boundary_facets)
    bc_u = fem.dirichletbc(u_bc_fun, dofs_u, W.sub(0))

    p_dofs = fem.locate_dofs_geometrical(
        (W.sub(1), Q), lambda X: np.isclose(X[0], 0.0) & np.isclose(X[1], 0.0)
    )
    p0_fun = fem.Function(Q)
    p0_fun.x.array[:] = 0.0
    bc_p = fem.dirichletbc(p0_fun, p_dofs, W.sub(1))
    bcs = [bc_u, bc_p]

    # Picard initialization
    wk = fem.Function(W)
    wk.x.array[:] = 0.0
    uk, pk = ufl.split(wk)
    dw = ufl.TrialFunction(W)
    (u, p) = ufl.split(dw)
    (v, q) = ufl.TestFunctions(W)

    a_picard = (
        nu * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
        + ufl.inner(ufl.grad(u) * uk, v) * ufl.dx
        - ufl.inner(p, ufl.div(v)) * ufl.dx
        + ufl.inner(ufl.div(u), q) * ufl.dx
    )
    L_picard = ufl.inner(f_ufl, v) * ufl.dx

    picard_iterations = 0
    total_linear_iterations = 0
    for _ in range(4):
        problem_picard = petsc.LinearProblem(
            a_picard,
            L_picard,
            bcs=bcs,
            petsc_options_prefix="ns_picard_",
            petsc_options={
                "ksp_type": "gmres",
                "pc_type": "lu",
                "ksp_rtol": 1e-10,
            },
        )
        w_new = problem_picard.solve()
        it = problem_picard.solver.getIterationNumber()
        total_linear_iterations += int(it)
        diff = np.linalg.norm(w_new.x.array - wk.x.array)
        wk.x.array[:] = w_new.x.array
        wk.x.scatter_forward()
        picard_iterations += 1
        if diff < 1e-10:
            break

    # Newton solve
    w = fem.Function(W)
    w.x.array[:] = wk.x.array
    w.x.scatter_forward()
    (u_nl, p_nl) = ufl.split(w)
    (v, q) = ufl.TestFunctions(W)

    F = (
        nu * ufl.inner(ufl.grad(u_nl), ufl.grad(v)) * ufl.dx
        + ufl.inner(ufl.grad(u_nl) * u_nl, v) * ufl.dx
        - ufl.inner(p_nl, ufl.div(v)) * ufl.dx
        + ufl.inner(ufl.div(u_nl), q) * ufl.dx
        - ufl.inner(f_ufl, v) * ufl.dx
    )
    J = ufl.derivative(F, w)

    nonlinear_problem = petsc.NonlinearProblem(
        F,
        w,
        bcs=bcs,
        J=J,
        petsc_options_prefix="ns_newton_",
        petsc_options={
            "snes_type": "newtonls",
            "snes_linesearch_type": "bt",
            "snes_rtol": newton_rtol,
            "snes_atol": 1e-12,
            "snes_max_it": newton_max_it,
            "ksp_type": "gmres",
            "pc_type": "lu",
            "ksp_rtol": 1e-10,
        },
    )

    try:
        w = nonlinear_problem.solve()
    except Exception:
        nonlinear_problem = petsc.NonlinearProblem(
            F,
            w,
            bcs=bcs,
            J=J,
            petsc_options_prefix="ns_newton_fallback_",
            petsc_options={
                "snes_type": "newtonls",
                "snes_linesearch_type": "basic",
                "snes_rtol": 1e-8,
                "snes_atol": 1e-10,
                "snes_max_it": max(newton_max_it, 40),
                "ksp_type": "preonly",
                "pc_type": "lu",
            },
        )
        w = nonlinear_problem.solve()

    w.x.scatter_forward()

    u_h = w.sub(0).collapse()
    p_h = w.sub(1).collapse()

    # Accuracy verification
    u_ex = fem.Function(V)
    u_ex.interpolate(u_exact_callable)
    err_fun = fem.Function(V)
    err_fun.x.array[:] = u_h.x.array - u_ex.x.array
    err_fun.x.scatter_forward()
    l2_error = np.sqrt(comm.allreduce(fem.assemble_scalar(fem.form(ufl.inner(err_fun, err_fun) * ufl.dx)), op=MPI.SUM))

    # Sample velocity magnitude on requested grid
    grid = case_spec["output"]["grid"]
    nx = int(grid["nx"])
    ny = int(grid["ny"])
    bbox = grid["bbox"]
    xs = np.linspace(bbox[0], bbox[1], nx)
    ys = np.linspace(bbox[2], bbox[3], ny)
    XX, YY = np.meshgrid(xs, ys)
    pts = np.column_stack([XX.ravel(), YY.ravel(), np.zeros(nx * ny, dtype=np.float64)])

    bb = geometry.bb_tree(msh, msh.topology.dim)
    candidates = geometry.compute_collisions_points(bb, pts)
    colliding = geometry.compute_colliding_cells(msh, candidates, pts)

    local_ids = []
    local_pts = []
    local_cells = []
    for i in range(pts.shape[0]):
        links = colliding.links(i)
        if len(links) > 0:
            local_ids.append(i)
            local_pts.append(pts[i])
            local_cells.append(links[0])

    local_mag = np.full(pts.shape[0], np.nan, dtype=np.float64)
    if local_pts:
        vals = u_h.eval(np.array(local_pts, dtype=np.float64), np.array(local_cells, dtype=np.int32))
        local_mag[np.array(local_ids, dtype=np.int32)] = np.linalg.norm(vals, axis=1)

    recv = comm.gather(local_mag, root=0)
    if comm.rank == 0:
        mag = np.full(pts.shape[0], np.nan, dtype=np.float64)
        for arr in recv:
            mask = ~np.isnan(arr)
            mag[mask] = arr[mask]
        if np.isnan(mag).any():
            raise RuntimeError("Failed to evaluate solution at all requested grid points.")
        u_grid = mag.reshape(ny, nx)
    else:
        u_grid = None

    u_grid = comm.bcast(u_grid, root=0)

    solver_info = {
        "mesh_resolution": mesh_resolution,
        "element_degree": degree_u,
        "ksp_type": "gmres",
        "pc_type": "lu",
        "rtol": 1e-10,
        "iterations": int(total_linear_iterations),
        "nonlinear_iterations": [picard_iterations],
        "l2_error": float(l2_error),
        "wall_time_sec": float(time.time() - t0),
    }

    return {"u": u_grid, "solver_info": solver_info}


if __name__ == "__main__":
    case_spec = {
        "pde": {"nu": 0.1, "time": None},
        "output": {"grid": {"nx": 64, "ny": 64, "bbox": [0.0, 1.0, 0.0, 1.0]}},
    }
    out = solve(case_spec)
    if MPI.COMM_WORLD.rank == 0:
        print(out["u"].shape)
        print(out["solver_info"])
