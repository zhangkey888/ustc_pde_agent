import time
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
import ufl
from basix.ufl import element as basix_element, mixed_element as basix_mixed_element
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc


def solve(case_spec: dict) -> dict:
    """
    Solve steady incompressible Navier-Stokes with manufactured solution on [0,1]^2
    using dolfinx on a quadrilateral mesh. Returns sampled velocity magnitude field.
    """
    comm = MPI.COMM_WORLD
    rank = comm.rank
    ScalarType = PETSc.ScalarType

    # ---------------------------
    # DIAGNOSIS
    # ---------------------------
    # equation_type: navier_stokes
    # spatial_dim: 2
    # domain_geometry: rectangle
    # unknowns: vector+scalar
    # coupling: saddle_point
    # linearity: nonlinear
    # time_dependence: steady
    # stiffness: N/A
    # dominant_physics: mixed
    # peclet_or_reynolds: moderate
    # solution_regularity: smooth
    # bc_type: all_dirichlet
    # special_notes: pressure_pinning, manufactured_solution

    # ---------------------------
    # METHOD
    # ---------------------------
    # spatial_method: fem
    # element_or_basis: Taylor-Hood_Q2Q1
    # stabilization: none
    # time_method: none
    # nonlinear_solver: newton
    # linear_solver: gmres
    # preconditioner: lu
    # special_treatment: pressure_pinning
    # pde_skill: navier_stokes

    pde = case_spec.get("pde", {})
    nu = float(pde.get("nu", case_spec.get("viscosity", 0.1)))
    if nu is None:
        nu = 0.1

    # Adaptive accuracy/time tradeoff by selecting a fairly fine default mesh while
    # keeping within the generous time budget.
    nx_default = int(case_spec.get("mesh_resolution", 96))
    ny_default = nx_default

    domain = mesh.create_rectangle(
        comm,
        [np.array([0.0, 0.0]), np.array([1.0, 1.0])],
        [nx_default, ny_default],
        cell_type=mesh.CellType.quadrilateral,
    )

    gdim = domain.geometry.dim
    cell_name = domain.topology.cell_name()

    vel_el = basix_element("Lagrange", cell_name, 2, shape=(gdim,))
    pre_el = basix_element("Lagrange", cell_name, 1)
    W = fem.functionspace(domain, basix_mixed_element([vel_el, pre_el]))
    V, _ = W.sub(0).collapse()
    Q, _ = W.sub(1).collapse()

    x = ufl.SpatialCoordinate(domain)
    pi = np.pi

    u_exact_ufl = ufl.as_vector(
        [
            ufl.pi * ufl.cos(ufl.pi * x[1]) * ufl.sin(ufl.pi * x[0]),
            -ufl.pi * ufl.cos(ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1]),
        ]
    )
    p_exact_ufl = x[0] - x[0]

    def eps(u):
        return ufl.sym(ufl.grad(u))

    f_ufl = ufl.grad(u_exact_ufl) * u_exact_ufl - nu * ufl.div(ufl.grad(u_exact_ufl)) + ufl.grad(p_exact_ufl)

    # Boundary conditions
    u_bc_fun = fem.Function(V)
    u_bc_expr = fem.Expression(u_exact_ufl, V.element.interpolation_points)
    u_bc_fun.interpolate(u_bc_expr)

    fdim = domain.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda X: np.ones(X.shape[1], dtype=bool)
    )
    u_dofs = fem.locate_dofs_topological((W.sub(0), V), fdim, boundary_facets)
    bc_u = fem.dirichletbc(u_bc_fun, u_dofs, W.sub(0))

    p_pin_dofs = fem.locate_dofs_geometrical(
        (W.sub(1), Q),
        lambda X: np.isclose(X[0], 0.0) & np.isclose(X[1], 0.0),
    )
    p0 = fem.Function(Q)
    p0.x.array[:] = 0.0
    bc_p = fem.dirichletbc(p0, p_pin_dofs, W.sub(1))
    bcs = [bc_u, bc_p]

    # Initial guess from Stokes problem
    (u_tr, p_tr) = ufl.TrialFunctions(W)
    (v_te, q_te) = ufl.TestFunctions(W)
    a_stokes = (
        2.0 * nu * ufl.inner(eps(u_tr), eps(v_te)) * ufl.dx
        - ufl.inner(p_tr, ufl.div(v_te)) * ufl.dx
        + ufl.inner(ufl.div(u_tr), q_te) * ufl.dx
    )
    L_stokes = ufl.inner(f_ufl, v_te) * ufl.dx

    stokes_problem = petsc.LinearProblem(
        a_stokes,
        L_stokes,
        bcs=bcs,
        petsc_options_prefix="ns_stokes_",
        petsc_options={
            "ksp_type": "preonly",
            "pc_type": "lu",
        },
    )
    w = stokes_problem.solve()
    w.x.scatter_forward()

    # Nonlinear NS solve
    (u, p) = ufl.split(w)
    (v, q) = ufl.TestFunctions(W)

    F = (
        2.0 * nu * ufl.inner(eps(u), eps(v)) * ufl.dx
        + ufl.inner(ufl.grad(u) * u, v) * ufl.dx
        - ufl.inner(f_ufl, v) * ufl.dx
        - ufl.inner(p, ufl.div(v)) * ufl.dx
        + ufl.inner(ufl.div(u), q) * ufl.dx
    )
    J = ufl.derivative(F, w)

    snes_prefix = "ns_nonlin_"
    problem = petsc.NonlinearProblem(
        F,
        w,
        bcs=bcs,
        J=J,
        petsc_options_prefix=snes_prefix,
        petsc_options={
            "snes_type": "newtonls",
            "snes_linesearch_type": "bt",
            "snes_rtol": 1.0e-11,
            "snes_atol": 1.0e-12,
            "snes_max_it": 25,
            "ksp_type": "preonly",
            "pc_type": "lu",
        },
    )

    t0 = time.perf_counter()
    w = problem.solve()
    w.x.scatter_forward()
    wall = time.perf_counter() - t0

    # Extract subfunctions
    u_h = w.sub(0).collapse()
    p_h = w.sub(1).collapse()

    # Accuracy verification: L2 error against exact solution
    u_exact_fun = fem.Function(V)
    u_exact_fun.interpolate(u_bc_expr)
    err_fun = fem.Function(V)
    err_fun.x.array[:] = u_h.x.array - u_exact_fun.x.array
    err_fun.x.scatter_forward()

    l2_local = fem.assemble_scalar(fem.form(ufl.inner(err_fun, err_fun) * ufl.dx))
    l2_global = comm.allreduce(l2_local, op=MPI.SUM)
    l2_error = float(np.sqrt(l2_global))

    # Sample velocity magnitude on requested grid
    grid = case_spec["output"]["grid"]
    nx = int(grid["nx"])
    ny = int(grid["ny"])
    bbox = grid["bbox"]
    xs = np.linspace(float(bbox[0]), float(bbox[1]), nx)
    ys = np.linspace(float(bbox[2]), float(bbox[3]), ny)
    XX, YY = np.meshgrid(xs, ys, indexing="xy")
    pts = np.column_stack([XX.ravel(), YY.ravel(), np.zeros(nx * ny, dtype=np.float64)])

    tree = geometry.bb_tree(domain, domain.topology.dim)
    candidates = geometry.compute_collisions_points(tree, pts)
    colliding = geometry.compute_colliding_cells(domain, candidates, pts)

    points_on_proc = []
    cells = []
    ids = []
    for i in range(pts.shape[0]):
        links = colliding.links(i)
        if len(links) > 0:
            points_on_proc.append(pts[i])
            cells.append(links[0])
            ids.append(i)

    local_vals = np.full((pts.shape[0],), np.nan, dtype=np.float64)
    if len(points_on_proc) > 0:
        vals = u_h.eval(np.array(points_on_proc, dtype=np.float64), np.array(cells, dtype=np.int32))
        mags = np.linalg.norm(vals, axis=1)
        local_vals[np.array(ids, dtype=np.int64)] = mags

    gathered = comm.gather(local_vals, root=0)
    if rank == 0:
        final_vals = np.full((pts.shape[0],), np.nan, dtype=np.float64)
        for arr in gathered:
            mask = np.isfinite(arr)
            final_vals[mask] = arr[mask]
        if np.any(~np.isfinite(final_vals)):
            raise RuntimeError("Failed to evaluate solution at all sample points.")
        u_grid = final_vals.reshape(ny, nx)
    else:
        u_grid = None

    # Solver metadata
    snes = PETSc.Options()
    try:
        nonlinear_its = int(problem.solver.getIterationNumber())
        ksp_its = int(problem.solver.getLinearSolveIterations())
        ksp_obj = problem.solver.getKSP()
        ksp_type = ksp_obj.getType()
        pc_type = ksp_obj.getPC().getType()
    except Exception:
        nonlinear_its = 0
        ksp_its = 0
        ksp_type = "preonly"
        pc_type = "lu"

    result = {
        "u": u_grid,
        "solver_info": {
            "mesh_resolution": nx_default,
            "element_degree": 2,
            "ksp_type": str(ksp_type),
            "pc_type": str(pc_type),
            "rtol": 1.0e-11,
            "iterations": int(ksp_its),
            "nonlinear_iterations": [int(nonlinear_its)],
            "l2_error": l2_error,
            "wall_time_sec": wall,
        },
    }
    return result


if __name__ == "__main__":
    case_spec = {
        "pde": {"nu": 0.1, "time": None},
        "output": {"grid": {"nx": 64, "ny": 64, "bbox": [0.0, 1.0, 0.0, 1.0]}},
    }
    out = solve(case_spec)
    if MPI.COMM_WORLD.rank == 0:
        print(out["u"].shape)
        print(out["solver_info"])
