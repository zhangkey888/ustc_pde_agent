import math
import time
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl

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
# special_notes: manufactured_solution, variable_coeff
# ```
#
# ```METHOD
# spatial_method: fem
# element_or_basis: Lagrange_P2
# stabilization: none
# time_method: none
# nonlinear_solver: none
# linear_solver: direct_lu
# preconditioner: none
# special_treatment: none
# pde_skill: poisson
# ```

ScalarType = PETSc.ScalarType


def _exact_numpy(x, y):
    return np.sin(2.0 * np.pi * x) * np.sin(2.0 * np.pi * y)


def _probe_function(u_func, points):
    domain = u_func.function_space.mesh
    tree = geometry.bb_tree(domain, domain.topology.dim)
    candidates = geometry.compute_collisions_points(tree, points)
    colliding = geometry.compute_colliding_cells(domain, candidates, points)

    pts_local = []
    cells_local = []
    idx_local = []
    for i in range(points.shape[0]):
        links = colliding.links(i)
        if len(links) > 0:
            pts_local.append(points[i])
            cells_local.append(links[0])
            idx_local.append(i)

    values = np.full(points.shape[0], np.nan, dtype=np.float64)
    if pts_local:
        vals = u_func.eval(
            np.array(pts_local, dtype=np.float64),
            np.array(cells_local, dtype=np.int32),
        )
        values[np.array(idx_local, dtype=np.int32)] = np.asarray(vals).reshape(-1)
    return values


def _sample_on_uniform_grid(u_func, grid_spec):
    nx = int(grid_spec["nx"])
    ny = int(grid_spec["ny"])
    xmin, xmax, ymin, ymax = map(float, grid_spec["bbox"])

    xs = np.linspace(xmin, xmax, nx, dtype=np.float64)
    ys = np.linspace(ymin, ymax, ny, dtype=np.float64)
    X, Y = np.meshgrid(xs, ys, indexing="xy")
    pts = np.column_stack(
        [X.ravel(), Y.ravel(), np.zeros(nx * ny, dtype=np.float64)]
    )

    vals_local = _probe_function(u_func, pts)
    comm = u_func.function_space.mesh.comm
    vals = comm.allreduce(np.nan_to_num(vals_local, nan=0.0), op=MPI.SUM)
    return vals.reshape(ny, nx)


def _solve_poisson(mesh_resolution=64, element_degree=2, rtol=1e-12):
    comm = MPI.COMM_WORLD
    msh = mesh.create_unit_square(comm, mesh_resolution, mesh_resolution, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(msh, ("Lagrange", element_degree))

    x = ufl.SpatialCoordinate(msh)
    u_exact = ufl.sin(2.0 * np.pi * x[0]) * ufl.sin(2.0 * np.pi * x[1])
    kappa = 1.0 + 0.3 * ufl.sin(8.0 * np.pi * x[0]) * ufl.sin(8.0 * np.pi * x[1])
    f = -ufl.div(kappa * ufl.grad(u_exact))

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    a = ufl.inner(kappa * ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = ufl.inner(f, v) * ufl.dx

    fdim = msh.topology.dim - 1
    facets = mesh.locate_entities_boundary(msh, fdim, lambda X: np.ones(X.shape[1], dtype=bool))
    dofs = fem.locate_dofs_topological(V, fdim, facets)

    u_bc = fem.Function(V)
    u_bc.interpolate(lambda X: np.sin(2.0 * np.pi * X[0]) * np.sin(2.0 * np.pi * X[1]))
    bc = fem.dirichletbc(u_bc, dofs)

    t0 = time.perf_counter()
    problem = petsc.LinearProblem(
        a,
        L,
        bcs=[bc],
        petsc_options_prefix="poisson_",
        petsc_options={
            "ksp_type": "preonly",
            "pc_type": "lu",
        },
    )
    uh = problem.solve()
    uh.x.scatter_forward()
    solve_time = time.perf_counter() - t0

    ue = fem.Function(V)
    ue.interpolate(lambda X: np.sin(2.0 * np.pi * X[0]) * np.sin(2.0 * np.pi * X[1]))

    err_sq_local = fem.assemble_scalar(fem.form((uh - ue) ** 2 * ufl.dx))
    norm_sq_local = fem.assemble_scalar(fem.form(ue ** 2 * ufl.dx))
    err_l2 = math.sqrt(comm.allreduce(err_sq_local, op=MPI.SUM))
    norm_l2 = math.sqrt(comm.allreduce(norm_sq_local, op=MPI.SUM))
    rel_l2 = err_l2 / norm_l2 if norm_l2 > 0 else err_l2

    info = {
        "mesh_resolution": int(mesh_resolution),
        "element_degree": int(element_degree),
        "ksp_type": "preonly",
        "pc_type": "lu",
        "rtol": float(rtol),
        "iterations": 0,
        "l2_error": float(err_l2),
        "relative_l2_error": float(rel_l2),
        "solve_time": float(solve_time),
    }
    return uh, info


def solve(case_spec: dict) -> dict:
    uh, info = _solve_poisson(mesh_resolution=64, element_degree=2, rtol=1e-12)
    u_grid = _sample_on_uniform_grid(uh, case_spec["output"]["grid"])

    grid = case_spec["output"]["grid"]
    nx = int(grid["nx"])
    ny = int(grid["ny"])
    xmin, xmax, ymin, ymax = map(float, grid["bbox"])
    xs = np.linspace(xmin, xmax, nx, dtype=np.float64)
    ys = np.linspace(ymin, ymax, ny, dtype=np.float64)
    X, Y = np.meshgrid(xs, ys, indexing="xy")
    exact_grid = _exact_numpy(X, Y)
    grid_linf = float(np.max(np.abs(u_grid - exact_grid)))

    solver_info = {
        "mesh_resolution": int(info["mesh_resolution"]),
        "element_degree": int(info["element_degree"]),
        "ksp_type": str(info["ksp_type"]),
        "pc_type": str(info["pc_type"]),
        "rtol": float(info["rtol"]),
        "iterations": int(info["iterations"]),
        "l2_error": float(info["l2_error"]),
        "relative_l2_error": float(info["relative_l2_error"]),
        "grid_linf_error": grid_linf,
    }
    return {"u": u_grid, "solver_info": solver_info}


if __name__ == "__main__":
    case_spec = {
        "output": {"grid": {"nx": 64, "ny": 64, "bbox": [0.0, 1.0, 0.0, 1.0]}}
    }
    out = solve(case_spec)
    if MPI.COMM_WORLD.rank == 0:
        print(out["u"].shape)
        print(out["solver_info"])
