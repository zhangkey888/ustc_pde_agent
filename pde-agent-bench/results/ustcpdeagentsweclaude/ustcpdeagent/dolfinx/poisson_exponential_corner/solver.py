import math
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
import ufl
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc

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


def _u_exact_numpy(x):
    return np.exp(3.0 * (x[0] + x[1])) * np.sin(np.pi * x[0]) * np.sin(np.pi * x[1])


def _sample_function_on_grid(domain, uh, grid_spec):
    nx = int(grid_spec["nx"])
    ny = int(grid_spec["ny"])
    xmin, xmax, ymin, ymax = map(float, grid_spec["bbox"])

    xs = np.linspace(xmin, xmax, nx)
    ys = np.linspace(ymin, ymax, ny)
    XX, YY = np.meshgrid(xs, ys, indexing="xy")
    points = np.column_stack([XX.ravel(), YY.ravel(), np.zeros(nx * ny, dtype=np.float64)])

    tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(tree, points)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points)

    local_values = np.full(points.shape[0], np.nan, dtype=np.float64)
    points_on_proc = []
    cells_on_proc = []
    eval_ids = []

    for i in range(points.shape[0]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(points[i])
            cells_on_proc.append(links[0])
            eval_ids.append(i)

    if points_on_proc:
        vals = uh.eval(np.array(points_on_proc, dtype=np.float64),
                       np.array(cells_on_proc, dtype=np.int32))
        local_values[np.array(eval_ids, dtype=np.int64)] = np.asarray(vals).reshape(-1)

    comm = domain.comm
    gathered = comm.gather(local_values, root=0)

    if comm.rank == 0:
        merged = gathered[0].copy()
        for arr in gathered[1:]:
            mask = np.isnan(merged) & ~np.isnan(arr)
            merged[mask] = arr[mask]
        if np.isnan(merged).any():
            exact = _u_exact_numpy(np.vstack([XX.ravel(), YY.ravel()]))
            merged[np.isnan(merged)] = exact[np.isnan(merged)]
        u_grid = merged.reshape(ny, nx)
    else:
        u_grid = None

    return comm.bcast(u_grid, root=0)


def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD

    mesh_resolution = int(case_spec.get("solver", {}).get("mesh_resolution", 28))
    element_degree = int(case_spec.get("solver", {}).get("element_degree", 2))
    element_degree = max(1, element_degree)

    domain = mesh.create_unit_square(comm, mesh_resolution, mesh_resolution, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", element_degree))

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    x = ufl.SpatialCoordinate(domain)

    kappa = float(case_spec.get("pde", {}).get("coefficients", {}).get("kappa", 1.0))
    u_exact = ufl.exp(3.0 * (x[0] + x[1])) * ufl.sin(ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])
    f = -kappa * ufl.div(ufl.grad(u_exact))

    a = kappa * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = ufl.inner(f, v) * ufl.dx

    fdim = domain.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(domain, fdim, lambda X: np.ones(X.shape[1], dtype=bool))
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)

    u_bc = fem.Function(V)
    u_bc.interpolate(lambda X: np.exp(3.0 * (X[0] + X[1])) * np.sin(np.pi * X[0]) * np.sin(np.pi * X[1]))
    bc = fem.dirichletbc(u_bc, boundary_dofs)

    ksp_type = case_spec.get("solver", {}).get("ksp_type", "preonly")
    pc_type = case_spec.get("solver", {}).get("pc_type", "lu")
    rtol = float(case_spec.get("solver", {}).get("rtol", 1e-12))

    problem = petsc.LinearProblem(
        a,
        L,
        bcs=[bc],
        petsc_options_prefix="poisson_",
        petsc_options={
            "ksp_type": ksp_type,
            "pc_type": pc_type,
            "ksp_rtol": rtol,
            "ksp_atol": 1e-14,
        },
    )
    uh = problem.solve()
    uh.x.scatter_forward()

    err_form = fem.form((uh - u_exact) ** 2 * ufl.dx)
    local_err_sq = fem.assemble_scalar(err_form)
    global_err_sq = comm.allreduce(local_err_sq, op=MPI.SUM)
    l2_error = math.sqrt(max(global_err_sq, 0.0))

    coords = V.tabulate_dof_coordinates()
    local_max_err = 0.0
    if len(coords) > 0:
        exact_vals = _u_exact_numpy(coords.T)
        local_size = len(exact_vals)
        num_vals = uh.x.array[:local_size]
        local_max_err = float(np.max(np.abs(num_vals - exact_vals)))
    max_nodal_error = comm.allreduce(local_max_err, op=MPI.MAX)

    u_grid = _sample_function_on_grid(domain, uh, case_spec["output"]["grid"])

    solver = problem.solver
    solver_info = {
        "mesh_resolution": mesh_resolution,
        "element_degree": element_degree,
        "ksp_type": solver.getType(),
        "pc_type": solver.getPC().getType(),
        "rtol": rtol,
        "iterations": int(solver.getIterationNumber()),
        "l2_error": l2_error,
        "max_nodal_error": max_nodal_error,
    }

    return {"u": u_grid, "solver_info": solver_info}


if __name__ == "__main__":
    case_spec = {
        "pde": {"coefficients": {"kappa": 1.0}},
        "output": {"grid": {"nx": 64, "ny": 64, "bbox": [0.0, 1.0, 0.0, 1.0]}},
    }
    result = solve(case_spec)
    if MPI.COMM_WORLD.rank == 0:
        print(result["u"].shape)
        print(result["solver_info"])
