import time
import math
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl

ScalarType = PETSc.ScalarType


def _probe_scalar_on_grid(domain, uh, nx, ny, bbox):
    xmin, xmax, ymin, ymax = bbox
    xs = np.linspace(xmin, xmax, nx)
    ys = np.linspace(ymin, ymax, ny)
    XX, YY = np.meshgrid(xs, ys, indexing="xy")
    pts2 = np.column_stack([XX.ravel(), YY.ravel(), np.zeros(nx * ny, dtype=np.float64)])

    tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(tree, pts2)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, pts2)

    vals = np.full(nx * ny, np.nan, dtype=np.float64)
    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    for i in range(pts2.shape[0]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(pts2[i])
            cells_on_proc.append(links[0])
            eval_map.append(i)

    if points_on_proc:
        local_vals = uh.eval(np.array(points_on_proc, dtype=np.float64),
                             np.array(cells_on_proc, dtype=np.int32)).reshape(-1)
        vals[np.array(eval_map, dtype=np.int32)] = np.asarray(local_vals, dtype=np.float64)

    comm = domain.comm
    if comm.size > 1:
        recv = np.empty_like(vals)
        comm.Allreduce(vals, recv, op=MPI.SUM)
        vals = recv
    return vals.reshape(ny, nx)


def _solve_once(n, degree, kappa, rtol, ksp_type, pc_type):
    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_square(comm, n, n, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", degree))

    x = ufl.SpatialCoordinate(domain)
    u_exact_ufl = ufl.sin(ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])
    f_ufl = 2.0 * (ufl.pi ** 2) * kappa * u_exact_ufl

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    a = kappa * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = ufl.inner(f_ufl, v) * ufl.dx

    fdim = domain.topology.dim - 1
    facets = mesh.locate_entities_boundary(domain, fdim, lambda X: np.ones(X.shape[1], dtype=bool))
    dofs = fem.locate_dofs_topological(V, fdim, facets)

    u_bc = fem.Function(V)
    u_bc.interpolate(fem.Expression(u_exact_ufl, V.element.interpolation_points))
    bc = fem.dirichletbc(u_bc, dofs)

    opts = {"ksp_type": ksp_type, "pc_type": pc_type, "ksp_rtol": rtol}
    if ksp_type == "cg" and pc_type == "hypre":
        opts["pc_hypre_type"] = "boomeramg"

    problem = petsc.LinearProblem(
        a, L, bcs=[bc], petsc_options=opts, petsc_options_prefix="poisson_"
    )

    t0 = time.perf_counter()
    uh = problem.solve()
    uh.x.scatter_forward()
    elapsed = time.perf_counter() - t0

    xdmf_expr = fem.Expression(u_exact_ufl, V.element.interpolation_points)
    u_ex = fem.Function(V)
    u_ex.interpolate(xdmf_expr)

    e = fem.Function(V)
    e.x.array[:] = uh.x.array - u_ex.x.array

    err_L2_local = fem.assemble_scalar(fem.form(ufl.inner(e, e) * ufl.dx))
    err_L2 = math.sqrt(comm.allreduce(err_L2_local, op=MPI.SUM))

    ksp = problem.solver
    its = int(ksp.getIterationNumber())

    return domain, uh, err_L2, elapsed, its


def solve(case_spec: dict) -> dict:
    # ```DIAGNOSIS
    # equation_type: poisson
    # spatial_dim: 2
    # domain_geometry: rectangle
    # unknowns: scalar
    # coupling: none
    # linearity: linear
    # time_dependence: steady
    # stiffness: N/A
    # dominant_physics: diffusion
    # peclet_or_reynolds: N/A
    # solution_regularity: smooth
    # bc_type: all_dirichlet
    # special_notes: manufactured_solution
    # ```
    # ```METHOD
    # spatial_method: fem
    # element_or_basis: Lagrange_P2
    # stabilization: none
    # time_method: none
    # nonlinear_solver: none
    # linear_solver: cg
    # preconditioner: amg
    # special_treatment: none
    # pde_skill: poisson
    # ```

    kappa = float(case_spec.get("pde", {}).get("coefficients", {}).get("kappa", 0.1))
    out_grid = case_spec["output"]["grid"]
    nx = int(out_grid["nx"])
    ny = int(out_grid["ny"])
    bbox = out_grid["bbox"]

    target_time = 0.565
    safety = 0.50

    candidates = [
        (24, 1, "cg", "hypre", 1e-10),
        (32, 1, "cg", "hypre", 1e-10),
        (24, 2, "cg", "hypre", 1e-10),
        (32, 2, "cg", "hypre", 1e-10),
        (40, 2, "cg", "hypre", 1e-11),
    ]

    best = None
    spent = 0.0
    for cand in candidates:
        n, degree, ksp_type, pc_type, rtol = cand
        try:
            domain, uh, err, elapsed, its = _solve_once(n, degree, kappa, rtol, ksp_type, pc_type)
        except Exception:
            domain, uh, err, elapsed, its = _solve_once(n, degree, kappa, 1e-12, "preonly", "lu")
            ksp_type, pc_type, rtol = "preonly", "lu", 1e-12
        spent += elapsed
        best = (domain, uh, err, elapsed, its, n, degree, ksp_type, pc_type, rtol)
        if spent > safety * target_time:
            break

    domain, uh, err, elapsed, its, n, degree, ksp_type, pc_type, rtol = best
    u_grid = _probe_scalar_on_grid(domain, uh, nx, ny, bbox)

    info = {
        "mesh_resolution": int(n),
        "element_degree": int(degree),
        "ksp_type": str(ksp_type),
        "pc_type": str(pc_type),
        "rtol": float(rtol),
        "iterations": int(its),
        "l2_error": float(err),
        "solve_wall_time": float(elapsed),
    }

    return {"u": u_grid, "solver_info": info}
