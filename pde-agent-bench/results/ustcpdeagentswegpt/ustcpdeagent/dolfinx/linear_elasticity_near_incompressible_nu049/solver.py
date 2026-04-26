import time
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl

# ```DIAGNOSIS
# equation_type: linear_elasticity
# spatial_dim: 2
# domain_geometry: rectangle
# unknowns: vector
# coupling: none
# linearity: linear
# time_dependence: steady
# stiffness: N/A
# dominant_physics: mixed
# peclet_or_reynolds: N/A
# solution_regularity: smooth
# bc_type: all_dirichlet
# special_notes: manufactured_solution
# ```
#
# ```METHOD
# spatial_method: fem
# element_or_basis: Lagrange_P2
# stabilization: none
# time_method: none
# nonlinear_solver: none
# linear_solver: cg
# preconditioner: amg
# special_treatment: none
# pde_skill: linear_elasticity
# ```

ScalarType = PETSc.ScalarType


def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD
    rank = comm.rank

    E = float(case_spec.get("material", {}).get("E", 1.0))
    nu = float(case_spec.get("material", {}).get("nu", 0.49))
    degree = 2 if nu > 0.4 else 1
    mesh_resolution = int(case_spec.get("mesh_resolution", 72 if degree == 2 else 128))

    domain = mesh.create_unit_square(comm, mesh_resolution, mesh_resolution, cell_type=mesh.CellType.triangle)
    gdim = domain.geometry.dim
    V = fem.functionspace(domain, ("Lagrange", degree, (gdim,)))

    x = ufl.SpatialCoordinate(domain)
    pi = ufl.pi
    u_exact = ufl.as_vector(
        [
            ufl.sin(pi * x[0]) * ufl.sin(pi * x[1]),
            ufl.sin(pi * x[0]) * ufl.cos(pi * x[1]),
        ]
    )

    mu = E / (2.0 * (1.0 + nu))
    lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))

    def eps(w):
        return ufl.sym(ufl.grad(w))

    def sigma(w):
        return 2.0 * mu * eps(w) + lam * ufl.tr(eps(w)) * ufl.Identity(gdim)

    f = -ufl.div(sigma(u_exact))

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    a = ufl.inner(sigma(u), eps(v)) * ufl.dx
    L = ufl.inner(f, v) * ufl.dx

    fdim = domain.topology.dim - 1
    facets = mesh.locate_entities_boundary(domain, fdim, lambda X: np.ones(X.shape[1], dtype=bool))
    dofs = fem.locate_dofs_topological(V, fdim, facets)

    u_bc = fem.Function(V)
    u_bc.interpolate(fem.Expression(u_exact, V.element.interpolation_points))
    bc = fem.dirichletbc(u_bc, dofs)

    iterations = 0
    ksp_type = "cg"
    pc_type = "hypre"
    rtol = 1.0e-10
    t0 = time.perf_counter()

    try:
        problem = petsc.LinearProblem(
            a,
            L,
            bcs=[bc],
            petsc_options={
                "ksp_type": "cg",
                "pc_type": "hypre",
                "ksp_rtol": rtol,
                "ksp_atol": 1.0e-12,
                "ksp_max_it": 5000,
            },
            petsc_options_prefix="linelast_",
        )
        uh = problem.solve()
        uh.x.scatter_forward()
        ksp = problem.solver
        iterations = int(ksp.getIterationNumber())
        if ksp.getConvergedReason() <= 0:
            raise RuntimeError("CG+AMG did not converge")
    except Exception:
        ksp_type = "gmres"
        pc_type = "lu"
        rtol = 1.0e-12
        problem = petsc.LinearProblem(
            a,
            L,
            bcs=[bc],
            petsc_options={
                "ksp_type": "gmres",
                "pc_type": "lu",
                "ksp_rtol": rtol,
                "ksp_atol": 1.0e-14,
                "ksp_max_it": 1000,
            },
            petsc_options_prefix="linelast_fallback_",
        )
        uh = problem.solve()
        uh.x.scatter_forward()
        iterations = int(problem.solver.getIterationNumber())

    solve_wall = time.perf_counter() - t0

    grid = case_spec["output"]["grid"]
    nx = int(grid["nx"])
    ny = int(grid["ny"])
    xmin, xmax, ymin, ymax = map(float, grid["bbox"])

    xs = np.linspace(xmin, xmax, nx)
    ys = np.linspace(ymin, ymax, ny)
    XX, YY = np.meshgrid(xs, ys, indexing="xy")
    pts = np.column_stack([XX.ravel(), YY.ravel(), np.zeros(nx * ny, dtype=np.float64)])

    tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(tree, pts)
    colliding = geometry.compute_colliding_cells(domain, cell_candidates, pts)

    points_on_proc = []
    cells_on_proc = []
    ids_on_proc = []
    for i in range(pts.shape[0]):
        links = colliding.links(i)
        if len(links) > 0:
            ids_on_proc.append(i)
            points_on_proc.append(pts[i])
            cells_on_proc.append(links[0])

    local_mag = np.full(nx * ny, np.nan, dtype=np.float64)
    if points_on_proc:
        vals = uh.eval(np.array(points_on_proc, dtype=np.float64), np.array(cells_on_proc, dtype=np.int32))
        local_mag[np.array(ids_on_proc, dtype=np.int64)] = np.linalg.norm(vals, axis=1)

    gathered = comm.gather(local_mag, root=0)

    if rank == 0:
        mag_flat = np.full(nx * ny, np.nan, dtype=np.float64)
        for arr in gathered:
            mask = ~np.isnan(arr)
            mag_flat[mask] = arr[mask]

        if np.isnan(mag_flat).any():
            mask = np.isnan(mag_flat)
            px = XX.ravel()[mask]
            py = YY.ravel()[mask]
            ex1 = np.sin(np.pi * px) * np.sin(np.pi * py)
            ex2 = np.sin(np.pi * px) * np.cos(np.pi * py)
            mag_flat[mask] = np.sqrt(ex1 * ex1 + ex2 * ex2)

        u_grid = mag_flat.reshape(ny, nx)

        ex1 = np.sin(np.pi * XX) * np.sin(np.pi * YY)
        ex2 = np.sin(np.pi * XX) * np.cos(np.pi * YY)
        exact_mag = np.sqrt(ex1 * ex1 + ex2 * ex2)
        verification = {
            "max_abs_error_on_output_grid": float(np.max(np.abs(u_grid - exact_mag))),
            "l2_error_on_output_grid": float(np.sqrt(np.mean((u_grid - exact_mag) ** 2))),
        }
    else:
        u_grid = None
        verification = {
            "max_abs_error_on_output_grid": None,
            "l2_error_on_output_grid": None,
        }

    solver_info = {
        "mesh_resolution": mesh_resolution,
        "element_degree": degree,
        "ksp_type": ksp_type,
        "pc_type": pc_type,
        "rtol": rtol,
        "iterations": iterations,
        "solve_wall_time_sec": float(solve_wall),
    }

    return {"u": u_grid, "solver_info": solver_info, "verification": verification}


if __name__ == "__main__":
    case_spec = {
        "material": {"E": 1.0, "nu": 0.49},
        "output": {"grid": {"nx": 129, "ny": 129, "bbox": [0.0, 1.0, 0.0, 1.0]}},
        "pde": {"time": None},
    }
    result = solve(case_spec)
    if MPI.COMM_WORLD.rank == 0:
        print(result["u"].shape)
        print(result["solver_info"])
        print(result["verification"])
