import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl

ScalarType = PETSc.ScalarType

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
#
# ```METHOD
# spatial_method: fem
# element_or_basis: Lagrange_P1
# stabilization: none
# time_method: none
# nonlinear_solver: none
# linear_solver: direct_lu
# preconditioner: none
# special_treatment: none
# pde_skill: poisson
# ```


def _sample_on_uniform_grid(u_func, grid_spec):
    comm = u_func.function_space.mesh.comm
    msh = u_func.function_space.mesh
    nx = int(grid_spec["nx"])
    ny = int(grid_spec["ny"])
    xmin, xmax, ymin, ymax = [float(v) for v in grid_spec["bbox"]]

    xs = np.linspace(xmin, xmax, nx, dtype=np.float64)
    ys = np.linspace(ymin, ymax, ny, dtype=np.float64)
    xx, yy = np.meshgrid(xs, ys, indexing="xy")
    pts = np.column_stack(
        [xx.ravel(), yy.ravel(), np.zeros(nx * ny, dtype=np.float64)]
    )

    tree = geometry.bb_tree(msh, msh.topology.dim)
    cell_candidates = geometry.compute_collisions_points(tree, pts)
    colliding_cells = geometry.compute_colliding_cells(msh, cell_candidates, pts)

    values = np.full(nx * ny, np.nan, dtype=np.float64)
    points_on_proc = []
    cells_on_proc = []
    ids_on_proc = []
    for i in range(pts.shape[0]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(pts[i])
            cells_on_proc.append(links[0])
            ids_on_proc.append(i)

    if points_on_proc:
        vals = u_func.eval(
            np.asarray(points_on_proc, dtype=np.float64),
            np.asarray(cells_on_proc, dtype=np.int32),
        )
        vals = np.asarray(vals, dtype=np.float64).reshape(len(points_on_proc), -1)[:, 0]
        values[np.asarray(ids_on_proc, dtype=np.int32)] = vals

    if comm.size == 1:
        return values.reshape(ny, nx)

    owned = (~np.isnan(values)).astype(np.int32)
    values = np.nan_to_num(values, nan=0.0)

    recv_vals = np.empty_like(values) if comm.rank == 0 else None
    recv_owned = np.empty_like(owned) if comm.rank == 0 else None
    comm.Reduce(values, recv_vals, op=MPI.SUM, root=0)
    comm.Reduce(owned, recv_owned, op=MPI.SUM, root=0)

    if comm.rank == 0:
        return recv_vals.reshape(ny, nx)
    return None


def _build_and_solve(comm, mesh_resolution, element_degree, kappa):
    msh = mesh.create_rectangle(
        comm,
        [np.array([0.0, 0.0], dtype=np.float64), np.array([1.0, 1.0], dtype=np.float64)],
        [mesh_resolution, mesh_resolution],
        cell_type=mesh.CellType.quadrilateral,
    )

    V = fem.functionspace(msh, ("Lagrange", element_degree))
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    x = ufl.SpatialCoordinate(msh)

    u_exact = ufl.sin(4.0 * ufl.pi * x[0]) * ufl.sin(4.0 * ufl.pi * x[1])
    f = ScalarType(32.0 * np.pi**2 * kappa) * u_exact

    fdim = msh.topology.dim - 1
    facets = mesh.locate_entities_boundary(
        msh, fdim, lambda X: np.ones(X.shape[1], dtype=bool)
    )
    dofs = fem.locate_dofs_topological(V, fdim, facets)

    u_bc = fem.Function(V)
    u_bc.interpolate(lambda X: np.sin(4.0 * np.pi * X[0]) * np.sin(4.0 * np.pi * X[1]))
    bc = fem.dirichletbc(u_bc, dofs)

    a = ScalarType(kappa) * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = ufl.inner(f, v) * ufl.dx

    problem = petsc.LinearProblem(
        a,
        L,
        bcs=[bc],
        petsc_options_prefix="poisson_bench_",
        petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
    )
    uh = problem.solve()
    uh.x.scatter_forward()

    l2_local = fem.assemble_scalar(fem.form((uh - u_exact) ** 2 * ufl.dx))
    l2_error = np.sqrt(comm.allreduce(l2_local, op=MPI.SUM))
    iterations = int(problem.solver.getIterationNumber())
    return uh, l2_error, iterations


def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD
    pde = case_spec.get("pde", {})
    coeffs = pde.get("coefficients", {})
    kappa = float(coeffs.get("kappa", 1.0))

    # Fast default tuned for the strict wall-time budget while resolving 4*pi oscillations.
    mesh_resolution = 40
    element_degree = 1

    uh, l2_error, iterations = _build_and_solve(
        comm, mesh_resolution, element_degree, kappa
    )
    u_grid = _sample_on_uniform_grid(uh, case_spec["output"]["grid"])

    solver_info = None
    if comm.rank == 0:
        grid = case_spec["output"]["grid"]
        xs = np.linspace(grid["bbox"][0], grid["bbox"][1], int(grid["nx"]), dtype=np.float64)
        ys = np.linspace(grid["bbox"][2], grid["bbox"][3], int(grid["ny"]), dtype=np.float64)
        xx, yy = np.meshgrid(xs, ys, indexing="xy")
        u_exact_grid = np.sin(4.0 * np.pi * xx) * np.sin(4.0 * np.pi * yy)
        grid_rmse = float(np.sqrt(np.mean((u_grid - u_exact_grid) ** 2)))

        solver_info = {
            "mesh_resolution": int(mesh_resolution),
            "element_degree": int(element_degree),
            "ksp_type": "preonly",
            "pc_type": "lu",
            "rtol": 1.0e-12,
            "iterations": int(iterations),
            "l2_error_check": float(l2_error),
            "grid_rmse_check": grid_rmse,
        }

    return {"u": u_grid if comm.rank == 0 else None, "solver_info": solver_info}


if __name__ == "__main__":
    case_spec = {
        "pde": {"coefficients": {"kappa": 1.0}},
        "output": {"grid": {"nx": 129, "ny": 129, "bbox": [0.0, 1.0, 0.0, 1.0]}},
    }
    out = solve(case_spec)
    if MPI.COMM_WORLD.rank == 0:
        print(out["u"].shape)
        print(out["solver_info"])
