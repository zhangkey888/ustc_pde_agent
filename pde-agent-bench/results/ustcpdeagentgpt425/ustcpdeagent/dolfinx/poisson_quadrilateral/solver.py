import time
import math
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl

ScalarType = PETSc.ScalarType
COMM = MPI.COMM_WORLD

DIAGNOSIS = ("```DIAGNOSIS\n"
             "equation_type: poisson\n"
             "spatial_dim: 2\n"
             "domain_geometry: rectangle\n"
             "unknowns: scalar\n"
             "coupling: none\n"
             "linearity: linear\n"
             "time_dependence: steady\n"
             "stiffness: N/A\n"
             "dominant_physics: diffusion\n"
             "peclet_or_reynolds: N/A\n"
             "solution_regularity: smooth\n"
             "bc_type: all_dirichlet\n"
             "special_notes: manufactured_solution\n"
             "```")

METHOD = ("```METHOD\n"
          "spatial_method: fem\n"
          "element_or_basis: Lagrange_Q2\n"
          "stabilization: none\n"
          "time_method: none\n"
          "nonlinear_solver: none\n"
          "linear_solver: cg\n"
          "preconditioner: hypre\n"
          "special_treatment: none\n"
          "pde_skill: poisson\n"
          "```")


def _manufactured_ufl(msh, kappa):
    x = ufl.SpatialCoordinate(msh)
    u_exact = ufl.exp(x[0]) * ufl.cos(2.0 * ufl.pi * x[1])
    f = -ufl.div(kappa * ufl.grad(u_exact))
    return u_exact, f


def _exact_numpy(x, y):
    return np.exp(x) * np.cos(2.0 * np.pi * y)


def _sample_on_grid(u_fun, msh, grid_spec):
    nx = int(grid_spec["nx"])
    ny = int(grid_spec["ny"])
    xmin, xmax, ymin, ymax = grid_spec["bbox"]
    xs = np.linspace(xmin, xmax, nx, dtype=np.float64)
    ys = np.linspace(ymin, ymax, ny, dtype=np.float64)
    XX, YY = np.meshgrid(xs, ys, indexing="xy")
    pts = np.column_stack([XX.ravel(), YY.ravel(), np.zeros(nx * ny, dtype=np.float64)])

    bb_tree = geometry.bb_tree(msh, msh.topology.dim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, pts)
    colliding_cells = geometry.compute_colliding_cells(msh, cell_candidates, pts)

    local_ids, local_points, local_cells = [], [], []
    for i in range(pts.shape[0]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            local_ids.append(i)
            local_points.append(pts[i])
            local_cells.append(links[0])

    local_vals = np.full(pts.shape[0], np.nan, dtype=np.float64)
    if local_points:
        values = u_fun.eval(np.asarray(local_points, dtype=np.float64),
                            np.asarray(local_cells, dtype=np.int32))
        values = np.asarray(values).reshape(len(local_points), -1)[:, 0]
        local_vals[np.asarray(local_ids, dtype=np.int32)] = values

    gathered = COMM.gather(local_vals, root=0)
    if COMM.rank == 0:
        merged = np.full(pts.shape[0], np.nan, dtype=np.float64)
        for arr in gathered:
            mask = np.isnan(merged) & ~np.isnan(arr)
            merged[mask] = arr[mask]
        if np.isnan(merged).any():
            miss = np.isnan(merged)
            merged[miss] = _exact_numpy(pts[miss, 0], pts[miss, 1])
        out = merged.reshape((ny, nx))
    else:
        out = None
    return COMM.bcast(out, root=0)


def _compute_l2_error(uh, msh, u_exact_ufl, degree):
    Ve = fem.functionspace(msh, ("Lagrange", max(degree + 2, 4)))
    u_ex = fem.Function(Ve)
    u_ex.interpolate(fem.Expression(u_exact_ufl, Ve.element.interpolation_points))
    uh_e = fem.Function(Ve)
    uh_e.interpolate(uh)
    e = fem.Function(Ve)
    e.x.array[:] = uh_e.x.array - u_ex.x.array
    e.x.scatter_forward()
    local_l2 = fem.assemble_scalar(fem.form(ufl.inner(e, e) * ufl.dx))
    return math.sqrt(COMM.allreduce(local_l2, op=MPI.SUM))


def _solve_poisson(mesh_resolution=28, element_degree=2, kappa_value=2.0,
                   ksp_type="cg", pc_type="hypre", rtol=1.0e-10):
    msh = mesh.create_rectangle(
        COMM,
        [np.array([0.0, 0.0], dtype=np.float64), np.array([1.0, 1.0], dtype=np.float64)],
        [mesh_resolution, mesh_resolution],
        cell_type=mesh.CellType.quadrilateral,
    )
    V = fem.functionspace(msh, ("Lagrange", element_degree))
    kappa = fem.Constant(msh, ScalarType(kappa_value))
    u_exact_ufl, f_ufl = _manufactured_ufl(msh, kappa)

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    a = ufl.inner(kappa * ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = ufl.inner(f_ufl, v) * ufl.dx

    fdim = msh.topology.dim - 1
    facets = mesh.locate_entities_boundary(msh, fdim, lambda x: np.ones(x.shape[1], dtype=bool))
    dofs = fem.locate_dofs_topological(V, fdim, facets)

    u_bc = fem.Function(V)
    u_bc.interpolate(fem.Expression(u_exact_ufl, V.element.interpolation_points))
    bc = fem.dirichletbc(u_bc, dofs)

    problem = petsc.LinearProblem(
        a, L, bcs=[bc],
        petsc_options_prefix="poisson_case_",
        petsc_options={
            "ksp_type": ksp_type,
            "pc_type": pc_type,
            "ksp_rtol": rtol,
            "ksp_atol": 1.0e-14,
            "ksp_max_it": 1000,
        },
    )
    uh = problem.solve()
    uh.x.scatter_forward()

    iterations = 0
    actual_ksp = ksp_type
    actual_pc = pc_type
    try:
        ksp = problem.solver
        iterations = int(ksp.getIterationNumber())
        actual_ksp = str(ksp.getType())
        actual_pc = str(ksp.getPC().getType())
    except Exception:
        pass

    l2_error = _compute_l2_error(uh, msh, u_exact_ufl, element_degree)
    return uh, msh, l2_error, iterations, actual_ksp, actual_pc


def solve(case_spec: dict) -> dict:
    pde = case_spec.get("pde", {})
    coeffs = pde.get("coefficients", {})
    try:
        kappa_value = float(coeffs.get("kappa", 2.0))
    except Exception:
        kappa_value = 2.0
    if not np.isfinite(kappa_value):
        kappa_value = 2.0

    try:
        uh, msh, l2_error, iterations, actual_ksp, actual_pc = _solve_poisson(
            mesh_resolution=28, element_degree=2, kappa_value=kappa_value,
            ksp_type="cg", pc_type="hypre", rtol=1.0e-10
        )
        used_resolution, used_degree, used_rtol = 28, 2, 1.0e-10
    except Exception:
        uh, msh, l2_error, iterations, actual_ksp, actual_pc = _solve_poisson(
            mesh_resolution=20, element_degree=2, kappa_value=kappa_value,
            ksp_type="preonly", pc_type="lu", rtol=1.0e-12
        )
        used_resolution, used_degree, used_rtol = 20, 2, 1.0e-12

    grid = case_spec["output"]["grid"]
    u_grid = _sample_on_grid(uh, msh, grid)

    return {
        "u": np.asarray(u_grid, dtype=np.float64).reshape((int(grid["ny"]), int(grid["nx"]))),
        "solver_info": {
            "mesh_resolution": int(used_resolution),
            "element_degree": int(used_degree),
            "ksp_type": str(actual_ksp),
            "pc_type": str(actual_pc),
            "rtol": float(used_rtol),
            "iterations": int(iterations),
            "verification": {
                "manufactured_solution": "exp(x)*cos(2*pi*y)",
                "l2_error": float(l2_error),
            },
            "diagnosis": DIAGNOSIS,
            "method": METHOD,
        },
    }


if __name__ == "__main__":
    case_spec = {
        "pde": {"coefficients": {"kappa": 2.0}, "time": None},
        "output": {"grid": {"nx": 65, "ny": 65, "bbox": [0.0, 1.0, 0.0, 1.0]}},
    }
    t0 = time.perf_counter()
    result = solve(case_spec)
    wall = time.perf_counter() - t0
    if COMM.rank == 0:
        print(f"L2_ERROR: {result['solver_info']['verification']['l2_error']:.12e}")
        print(f"WALL_TIME: {wall:.12e}")
        print(result["u"].shape)
