import math
import time
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl

ScalarType = PETSc.ScalarType


def _exact_and_rhs(msh, eps=0.2, beta_vec=(0.8, 0.3)):
    x = ufl.SpatialCoordinate(msh)
    u_ex = ufl.cos(ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])
    beta = ufl.as_vector((ScalarType(beta_vec[0]), ScalarType(beta_vec[1])))
    f = -eps * ufl.div(ufl.grad(u_ex)) + ufl.dot(beta, ufl.grad(u_ex))
    return u_ex, beta, f


def _all_boundary(x):
    return np.ones(x.shape[1], dtype=bool)


def _sample_function_on_grid(uh, nx, ny, bbox):
    msh = uh.function_space.mesh
    xs = np.linspace(bbox[0], bbox[1], nx)
    ys = np.linspace(bbox[2], bbox[3], ny)
    X, Y = np.meshgrid(xs, ys, indexing="xy")
    pts = np.column_stack([X.ravel(), Y.ravel(), np.zeros(nx * ny, dtype=np.float64)])

    tree = geometry.bb_tree(msh, msh.topology.dim)
    cell_candidates = geometry.compute_collisions_points(tree, pts)
    colliding = geometry.compute_colliding_cells(msh, cell_candidates, pts)

    vals = np.full(pts.shape[0], np.nan, dtype=np.float64)
    points_on_proc = []
    cells = []
    eval_ids = []
    for i in range(pts.shape[0]):
        links = colliding.links(i)
        if len(links) > 0:
            points_on_proc.append(pts[i])
            cells.append(links[0])
            eval_ids.append(i)

    if points_on_proc:
        local_vals = uh.eval(np.array(points_on_proc, dtype=np.float64),
                             np.array(cells, dtype=np.int32))
        vals[np.array(eval_ids, dtype=np.int32)] = np.asarray(local_vals).reshape(-1)

    gathered = msh.comm.gather(vals, root=0)
    if msh.comm.rank == 0:
        out = np.full_like(gathered[0], np.nan)
        for arr in gathered:
            mask = np.isfinite(arr)
            out[mask] = arr[mask]
        if np.any(~np.isfinite(out)):
            raise RuntimeError("Failed to evaluate solution at all requested output points.")
        return out.reshape(ny, nx)
    return None


def _solve_once(n, degree, eps=0.2, beta_vec=(0.8, 0.3), use_supg=True,
                ksp_type="gmres", pc_type="ilu", rtol=1e-10):
    comm = MPI.COMM_WORLD
    msh = mesh.create_unit_square(comm, n, n, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(msh, ("Lagrange", degree))

    u_ex, beta, f = _exact_and_rhs(msh, eps=eps, beta_vec=beta_vec)

    uD = fem.Function(V)
    uD.interpolate(fem.Expression(u_ex, V.element.interpolation_points))
    fdim = msh.topology.dim - 1
    facets = mesh.locate_entities_boundary(msh, fdim, _all_boundary)
    dofs = fem.locate_dofs_topological(V, fdim, facets)
    bc = fem.dirichletbc(uD, dofs)

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    a = eps * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx + ufl.dot(beta, ufl.grad(u)) * v * ufl.dx
    L = f * v * ufl.dx

    if use_supg:
        h = ufl.CellDiameter(msh)
        beta_norm = ufl.sqrt(ufl.dot(beta, beta) + 1.0e-16)
        Pek = beta_norm * h / (2.0 * eps)
        tau = h / (2.0 * beta_norm) * (ufl.cosh(Pek) / ufl.sinh(Pek) - 1.0 / Pek)
        strong_u = -eps * ufl.div(ufl.grad(u)) + ufl.dot(beta, ufl.grad(u))
        a += tau * strong_u * ufl.dot(beta, ufl.grad(v)) * ufl.dx
        L += tau * f * ufl.dot(beta, ufl.grad(v)) * ufl.dx

    problem = petsc.LinearProblem(
        a, L, bcs=[bc], petsc_options_prefix=f"cd_{n}_{degree}_",
        petsc_options={
            "ksp_type": ksp_type,
            "pc_type": pc_type,
            "ksp_rtol": rtol,
            "ksp_atol": 1e-14,
            "ksp_max_it": 5000
        }
    )

    t0 = time.perf_counter()
    uh = problem.solve()
    solve_time = time.perf_counter() - t0
    uh.x.scatter_forward()

    ksp = problem.solver
    l2_local = fem.assemble_scalar(fem.form((uh - u_ex) ** 2 * ufl.dx))
    l2_err = math.sqrt(comm.allreduce(l2_local, op=MPI.SUM))

    return {
        "mesh": msh,
        "V": V,
        "uh": uh,
        "mesh_resolution": n,
        "element_degree": degree,
        "ksp_type": ksp.getType(),
        "pc_type": ksp.getPC().getType(),
        "rtol": rtol,
        "iterations": int(ksp.getIterationNumber()),
        "l2_error": l2_err,
        "solve_time": solve_time,
    }


def solve(case_spec: dict) -> dict:
    """
    ```DIAGNOSIS
    equation_type:        convection_diffusion
    spatial_dim:          2
    domain_geometry:      rectangle
    unknowns:             scalar
    coupling:             none
    linearity:            linear
    time_dependence:      steady
    stiffness:            non_stiff
    dominant_physics:     mixed
    peclet_or_reynolds:   moderate
    solution_regularity:  smooth
    bc_type:              all_dirichlet
    special_notes:        manufactured_solution
    ```

    ```METHOD
    spatial_method:       fem
    element_or_basis:     Lagrange_P2
    stabilization:        supg
    time_method:          none
    nonlinear_solver:     none
    linear_solver:        gmres
    preconditioner:       ilu
    special_treatment:    none
    pde_skill:            convection_diffusion
    ```
    """
    grid = case_spec["output"]["grid"]
    nx = int(grid["nx"])
    ny = int(grid["ny"])
    bbox = grid["bbox"]

    # Adaptive accuracy/time trade-off:
    # escalate accuracy because the time budget is generous for this elliptic problem.
    candidates = [(40, 1), (56, 1), (48, 2), (64, 2), (80, 2)]
    best = None
    target_error = 1.66e-3
    soft_budget = 7.0

    for n, degree in candidates:
        res = _solve_once(n=n, degree=degree, use_supg=True)
        if best is None or res["l2_error"] < best["l2_error"]:
            best = res
        if res["l2_error"] <= target_error and res["solve_time"] >= 0.6 * soft_budget:
            best = res
            break

    u_grid = _sample_function_on_grid(best["uh"], nx=nx, ny=ny, bbox=bbox)

    solver_info = {
        "mesh_resolution": int(best["mesh_resolution"]),
        "element_degree": int(best["element_degree"]),
        "ksp_type": str(best["ksp_type"]),
        "pc_type": str(best["pc_type"]),
        "rtol": float(best["rtol"]),
        "iterations": int(best["iterations"]),
        "l2_error_verification": float(best["l2_error"]),
    }

    if MPI.COMM_WORLD.rank == 0:
        return {"u": u_grid, "solver_info": solver_info}
    return {"u": None, "solver_info": solver_info}
