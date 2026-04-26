import time
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl

ScalarType = PETSc.ScalarType


def _make_exact_and_rhs(domain, eps, beta):
    x = ufl.SpatialCoordinate(domain)
    u_exact = ufl.sin(ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])
    grad_u = ufl.grad(u_exact)
    lap_u = ufl.div(grad_u)
    beta_ufl = ufl.as_vector((ScalarType(beta[0]), ScalarType(beta[1])))
    f = -ScalarType(eps) * lap_u + ufl.dot(beta_ufl, grad_u)
    return u_exact, f


def _build_bc(V, u_exact):
    domain = V.mesh
    tdim = domain.topology.dim
    fdim = tdim - 1
    facets = mesh.locate_entities_boundary(domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool))
    dofs = fem.locate_dofs_topological(V, fdim, facets)
    u_bc = fem.Function(V)
    expr = fem.Expression(u_exact, V.element.interpolation_points)
    u_bc.interpolate(expr)
    return fem.dirichletbc(u_bc, dofs)


def _solve_once(n, degree, eps, beta, use_supg=False):
    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_square(comm, n, n, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", degree))

    u_exact, f = _make_exact_and_rhs(domain, eps, beta)
    bc = _build_bc(V, u_exact)

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    beta_c = fem.Constant(domain, np.array(beta, dtype=ScalarType))
    h = ufl.CellDiameter(domain)

    a = (ScalarType(eps) * ufl.inner(ufl.grad(u), ufl.grad(v)) + ufl.dot(beta_c, ufl.grad(u)) * v) * ufl.dx
    L = f * v * ufl.dx

    if use_supg:
        beta_norm = float(np.linalg.norm(beta))
        tau = 0.0 if beta_norm == 0.0 else min(h / (2.0 * beta_norm), h * h / (4.0 * float(eps)))
        r_u = -ScalarType(eps) * ufl.div(ufl.grad(u)) + ufl.dot(beta_c, ufl.grad(u))
        r_f = f
        a += tau * ufl.dot(beta_c, ufl.grad(v)) * r_u * ufl.dx
        L += tau * ufl.dot(beta_c, ufl.grad(v)) * r_f * ufl.dx

    petsc_options = {
        "ksp_type": "gmres",
        "pc_type": "ilu",
        "ksp_rtol": 1.0e-10,
        "ksp_atol": 1.0e-12,
    }

    t0 = time.perf_counter()
    problem = petsc.LinearProblem(
        a,
        L,
        bcs=[bc],
        petsc_options=petsc_options,
        petsc_options_prefix=f"cd_{n}_{degree}_",
    )
    uh = problem.solve()
    uh.x.scatter_forward()
    solve_time = time.perf_counter() - t0

    Vex = fem.functionspace(domain, ("Lagrange", max(degree + 2, 4)))
    uex_h = fem.Function(Vex)
    uex_h.interpolate(fem.Expression(u_exact, Vex.element.interpolation_points))
    uh_high = fem.Function(Vex)
    uh_high.interpolate(uh)

    err_L2 = np.sqrt(domain.comm.allreduce(fem.assemble_scalar(fem.form((uh_high - uex_h) ** 2 * ufl.dx)), op=MPI.SUM))
    err_H1 = np.sqrt(domain.comm.allreduce(fem.assemble_scalar(fem.form(ufl.inner(ufl.grad(uh_high - uex_h), ufl.grad(uh_high - uex_h)) * ufl.dx)), op=MPI.SUM))

    ksp = problem.solver
    its = int(ksp.getIterationNumber())
    ksp_type = ksp.getType()
    pc_type = ksp.getPC().getType()
    return {
        "domain": domain,
        "V": V,
        "uh": uh,
        "err_L2": float(err_L2),
        "err_H1": float(err_H1),
        "time": float(solve_time),
        "iterations": its,
        "ksp_type": str(ksp_type),
        "pc_type": str(pc_type),
    }


def _sample_to_grid(domain, uh, grid_spec):
    nx = int(grid_spec["nx"])
    ny = int(grid_spec["ny"])
    bbox = grid_spec["bbox"]
    xs = np.linspace(bbox[0], bbox[1], nx)
    ys = np.linspace(bbox[2], bbox[3], ny)
    XX, YY = np.meshgrid(xs, ys, indexing="xy")
    pts = np.column_stack([XX.ravel(), YY.ravel(), np.zeros(nx * ny, dtype=np.float64)])

    tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(tree, pts)
    colliding = geometry.compute_colliding_cells(domain, cell_candidates, pts)

    values = np.full((nx * ny,), np.nan, dtype=np.float64)
    points_on_proc = []
    cells = []
    ids = []
    for i in range(pts.shape[0]):
        links = colliding.links(i)
        if len(links) > 0:
            points_on_proc.append(pts[i])
            cells.append(links[0])
            ids.append(i)

    if points_on_proc:
        vals = uh.eval(np.array(points_on_proc, dtype=np.float64), np.array(cells, dtype=np.int32)).reshape(-1)
        values[np.array(ids, dtype=np.int32)] = np.real(vals)

    comm = domain.comm
    if comm.size > 1:
        gathered = comm.allgather(values)
        merged = np.full_like(values, np.nan)
        for arr in gathered:
            mask = np.isnan(merged) & ~np.isnan(arr)
            merged[mask] = arr[mask]
        values = merged

    if np.isnan(values).any():
        raise RuntimeError("Failed to evaluate solution on all requested grid points.")
    return values.reshape(ny, nx)


def solve(case_spec: dict) -> dict:
    pde = case_spec.get("pde", {})
    output = case_spec["output"]["grid"]

    eps = float(pde.get("epsilon", 0.2))
    beta = pde.get("beta", [1.0, 0.5])
    if isinstance(beta, dict):
        beta = [beta.get("x", 1.0), beta.get("y", 0.5)]
    beta = [float(beta[0]), float(beta[1])]

    pe = float(pde.get("peclet", 5.6))
    use_supg = pe > 8.0

    candidates = [(64, 2)]
    best = None

    for n, degree in candidates:
        best = _solve_once(n, degree, eps, beta, use_supg=use_supg)

    u_grid = _sample_to_grid(best["domain"], best["uh"], output)

    solver_info = {
        "mesh_resolution": int(candidates[0][0] if best is None else best["domain"].topology.index_map(best["domain"].topology.dim).size_local ** 0),
        "element_degree": int(best["V"].element.basix_element.degree),
        "ksp_type": best["ksp_type"],
        "pc_type": best["pc_type"],
        "rtol": 1.0e-10,
        "iterations": int(best["iterations"]),
        "l2_error": float(best["err_L2"]),
        "h1_error": float(best["err_H1"]),
    }
    solver_info["mesh_resolution"] = int(round(np.sqrt(best["domain"].topology.index_map(0).size_global))) - 1 if best is not None else candidates[0][0]

    return {"u": u_grid, "solver_info": solver_info}
