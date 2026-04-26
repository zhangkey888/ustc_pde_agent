import math
import time
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl

ScalarType = PETSc.ScalarType


def _sample_function_on_grid(msh, uh, grid):
    nx = int(grid["nx"])
    ny = int(grid["ny"])
    xmin, xmax, ymin, ymax = grid["bbox"]
    xs = np.linspace(xmin, xmax, nx)
    ys = np.linspace(ymin, ymax, ny)
    xx, yy = np.meshgrid(xs, ys, indexing="xy")
    pts = np.zeros((nx * ny, 3), dtype=np.float64)
    pts[:, 0] = xx.ravel()
    pts[:, 1] = yy.ravel()

    tree = geometry.bb_tree(msh, msh.topology.dim)
    candidates = geometry.compute_collisions_points(tree, pts)
    colliding = geometry.compute_colliding_cells(msh, candidates, pts)

    local_vals = np.full(nx * ny, np.nan, dtype=np.float64)
    points_on_proc = []
    cells_on_proc = []
    eval_idx = []
    for i in range(pts.shape[0]):
        links = colliding.links(i)
        if len(links) > 0:
            points_on_proc.append(pts[i])
            cells_on_proc.append(links[0])
            eval_idx.append(i)

    if points_on_proc:
        vals = uh.eval(np.asarray(points_on_proc, dtype=np.float64),
                       np.asarray(cells_on_proc, dtype=np.int32))
        local_vals[np.asarray(eval_idx, dtype=np.int32)] = np.asarray(vals).reshape(-1)

    gathered = msh.comm.gather(local_vals, root=0)
    if msh.comm.rank == 0:
        merged = np.full(nx * ny, np.nan, dtype=np.float64)
        for arr in gathered:
            mask = ~np.isnan(arr)
            merged[mask] = arr[mask]
        merged[np.isnan(merged)] = 0.0
        out = merged.reshape(ny, nx)
    else:
        out = None
    return msh.comm.bcast(out, root=0)


def solve(case_spec: dict) -> dict:
    """
    ```DIAGNOSIS
    equation_type: heat
    spatial_dim: 2
    domain_geometry: rectangle
    unknowns: scalar
    coupling: none
    linearity: linear
    time_dependence: transient
    stiffness: stiff
    dominant_physics: diffusion
    peclet_or_reynolds: N/A
    solution_regularity: smooth
    bc_type: all_dirichlet
    special_notes: manufactured_solution
    ```

    ```METHOD
    spatial_method: fem
    element_or_basis: Lagrange_P3
    stabilization: none
    time_method: backward_euler
    nonlinear_solver: none
    linear_solver: cg
    preconditioner: hypre
    special_treatment: none
    pde_skill: heat
    ```
    """
    comm = MPI.COMM_WORLD
    t_wall0 = time.perf_counter()

    pde = case_spec.get("pde", {})
    pde_time = pde.get("time", {})
    coeffs = case_spec.get("coefficients", {})

    t0 = float(pde_time.get("t0", 0.0))
    t_end = float(pde_time.get("t_end", 0.08))
    dt_suggested = float(pde_time.get("dt", 0.008))
    time_scheme = str(pde_time.get("scheme", "backward_euler")).lower()
    if time_scheme != "backward_euler":
        time_scheme = "backward_euler"
    kappa = float(coeffs.get("kappa", 1.0))

    # Accuracy/time trade-off: use higher-order FEM and a smaller dt than suggested.
    # This case is smooth and manufactured, so P3 + BE is efficient and accurate.
    element_degree = 3
    mesh_resolution = 40
    dt_target = min(dt_suggested / 4.0, 0.002)
    n_steps = max(1, int(round((t_end - t0) / dt_target)))
    dt = (t_end - t0) / n_steps

    msh = mesh.create_unit_square(comm, mesh_resolution, mesh_resolution, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(msh, ("Lagrange", element_degree))

    x = ufl.SpatialCoordinate(msh)
    t_c = fem.Constant(msh, ScalarType(t0))
    dt_c = fem.Constant(msh, ScalarType(dt))
    kappa_c = fem.Constant(msh, ScalarType(kappa))

    u_exact_ufl = ufl.exp(-t_c) * ufl.sin(3 * ufl.pi * x[0]) * ufl.sin(3 * ufl.pi * x[1])
    f_ufl = (-1.0 + 18.0 * (ufl.pi ** 2) * kappa_c) * u_exact_ufl

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    uh = fem.Function(V)
    u_n = fem.Function(V)
    u_bc = fem.Function(V)
    u_exact_fun = fem.Function(V)
    f_fun = fem.Function(V)

    def exact_callable(tt):
        return lambda X: np.exp(-tt) * np.sin(3.0 * np.pi * X[0]) * np.sin(3.0 * np.pi * X[1])

    def update_time_dependent_functions(tt):
        u_bc.interpolate(exact_callable(tt))
        u_bc.x.scatter_forward()
        u_exact_fun.interpolate(exact_callable(tt))
        u_exact_fun.x.scatter_forward()
        f_fun.interpolate(
            lambda X: ((-1.0 + 18.0 * np.pi * np.pi * kappa) * np.exp(-tt)
                       * np.sin(3.0 * np.pi * X[0]) * np.sin(3.0 * np.pi * X[1]))
        )
        f_fun.x.scatter_forward()

    update_time_dependent_functions(t0)
    u_n.x.array[:] = u_exact_fun.x.array
    u_n.x.scatter_forward()

    fdim = msh.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(msh, fdim, lambda X: np.ones(X.shape[1], dtype=bool))
    bdofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc = fem.dirichletbc(u_bc, bdofs)

    a = (u * v + dt_c * kappa_c * ufl.inner(ufl.grad(u), ufl.grad(v))) * ufl.dx
    L = (u_n * v + dt_c * f_fun * v) * ufl.dx

    a_form = fem.form(a)
    L_form = fem.form(L)

    A = petsc.assemble_matrix(a_form, bcs=[bc])
    A.assemble()
    b = petsc.create_vector(L_form.function_spaces)

    solver = PETSc.KSP().create(comm)
    solver.setOperators(A)
    solver.setType("cg")
    solver.getPC().setType("hypre")
    solver.setTolerances(rtol=1.0e-10, atol=1.0e-12, max_it=5000)
    solver.setFromOptions()

    total_iterations = 0
    for step in range(1, n_steps + 1):
        t_now = t0 + step * dt
        t_c.value = ScalarType(t_now)
        update_time_dependent_functions(t_now)

        with b.localForm() as b_local:
            b_local.set(0.0)
        petsc.assemble_vector(b, L_form)
        petsc.apply_lifting(b, [a_form], bcs=[[bc]])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        petsc.set_bc(b, [bc])

        solver.solve(b, uh.x.petsc_vec)
        uh.x.scatter_forward()

        if solver.getConvergedReason() <= 0:
            fallback = PETSc.KSP().create(comm)
            fallback.setOperators(A)
            fallback.setType("preonly")
            fallback.getPC().setType("lu")
            fallback.setFromOptions()
            fallback.solve(b, uh.x.petsc_vec)
            uh.x.scatter_forward()
            total_iterations += 1
        else:
            total_iterations += solver.getIterationNumber()

        u_n.x.array[:] = uh.x.array
        u_n.x.scatter_forward()

    update_time_dependent_functions(t_end)
    err = fem.Function(V)
    err.x.array[:] = uh.x.array - u_exact_fun.x.array
    err.x.scatter_forward()

    l2_sq_local = fem.assemble_scalar(fem.form(ufl.inner(err, err) * ufl.dx))
    l2_sq = comm.allreduce(l2_sq_local, op=MPI.SUM)
    l2_error = math.sqrt(l2_sq)

    grid = case_spec["output"]["grid"]
    u_grid = _sample_function_on_grid(msh, uh, grid)

    nx = int(grid["nx"])
    ny = int(grid["ny"])
    xmin, xmax, ymin, ymax = grid["bbox"]
    xs = np.linspace(xmin, xmax, nx)
    ys = np.linspace(ymin, ymax, ny)
    XX, YY = np.meshgrid(xs, ys, indexing="xy")
    u_initial = np.exp(-t0) * np.sin(3.0 * np.pi * XX) * np.sin(3.0 * np.pi * YY)

    solver_info = {
        "mesh_resolution": int(mesh_resolution),
        "element_degree": int(element_degree),
        "ksp_type": str(solver.getType()),
        "pc_type": str(solver.getPC().getType()),
        "rtol": float(1.0e-10),
        "iterations": int(total_iterations),
        "dt": float(dt),
        "n_steps": int(n_steps),
        "time_scheme": time_scheme,
        "l2_error": float(l2_error),
        "wall_time_sec": float(time.perf_counter() - t_wall0),
    }

    return {
        "u": np.asarray(u_grid, dtype=np.float64),
        "u_initial": np.asarray(u_initial, dtype=np.float64),
        "solver_info": solver_info,
    }
