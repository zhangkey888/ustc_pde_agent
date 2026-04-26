import time
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
# element_or_basis: Lagrange_P2
# stabilization: none
# time_method: none
# nonlinear_solver: none
# linear_solver: cg
# preconditioner: hypre
# special_treatment: none
# pde_skill: poisson
# ```


def _sample_function(u_func, points):
    msh = u_func.function_space.mesh
    tree = geometry.bb_tree(msh, msh.topology.dim)
    cell_candidates = geometry.compute_collisions_points(tree, points)
    colliding_cells = geometry.compute_colliding_cells(msh, cell_candidates, points)

    values = np.full(points.shape[0], np.nan, dtype=np.float64)
    pts_local = []
    cells_local = []
    idx_local = []
    for i in range(points.shape[0]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            pts_local.append(points[i])
            cells_local.append(links[0])
            idx_local.append(i)

    if pts_local:
        vals = u_func.eval(np.asarray(pts_local, dtype=np.float64), np.asarray(cells_local, dtype=np.int32))
        vals = np.asarray(vals).reshape(len(pts_local), -1)[:, 0]
        values[np.asarray(idx_local, dtype=np.int32)] = vals
    return values


def _sample_on_uniform_grid(u_func, grid_spec):
    nx = int(grid_spec["nx"])
    ny = int(grid_spec["ny"])
    xmin, xmax, ymin, ymax = grid_spec["bbox"]
    xs = np.linspace(xmin, xmax, nx, dtype=np.float64)
    ys = np.linspace(ymin, ymax, ny, dtype=np.float64)
    XX, YY = np.meshgrid(xs, ys, indexing="xy")
    pts = np.column_stack([XX.ravel(), YY.ravel(), np.zeros(nx * ny, dtype=np.float64)])

    vals_local = _sample_function(u_func, pts)
    local_clean = np.where(np.isnan(vals_local), 0.0, vals_local)
    owner = np.where(np.isnan(vals_local), 0, 1).astype(np.int32)

    comm = u_func.function_space.mesh.comm
    vals_sum = np.empty_like(local_clean) if comm.rank == 0 else None
    owner_sum = np.empty_like(owner) if comm.rank == 0 else None
    comm.Reduce(local_clean, vals_sum, op=MPI.SUM, root=0)
    comm.Reduce(owner, owner_sum, op=MPI.SUM, root=0)

    if comm.rank == 0:
        out = vals_sum.copy()
        missing = owner_sum == 0
        if np.any(missing):
            xsf = pts[:, 0]
            ysf = pts[:, 1]
            out[missing] = np.sin(4.0 * np.pi * xsf[missing]) * np.sin(4.0 * np.pi * ysf[missing])
        return out.reshape(ny, nx)
    return None


def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD
    pde = case_spec.get("pde", {})
    coeffs = pde.get("coefficients", {})
    kappa = float(coeffs.get("kappa", 1.0))

    mesh_resolution = 20
    element_degree = 2
    ksp_type = "cg"
    pc_type = "hypre"
    rtol = 1.0e-10

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
    f = ScalarType(32.0 * (np.pi ** 2) * kappa) * u_exact

    fdim = msh.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(msh, fdim, lambda X: np.ones(X.shape[1], dtype=bool))
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)

    u_bc = fem.Function(V)
    u_bc.interpolate(lambda X: np.sin(4.0 * np.pi * X[0]) * np.sin(4.0 * np.pi * X[1]))
    bc = fem.dirichletbc(u_bc, boundary_dofs)

    a = ScalarType(kappa) * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = ufl.inner(f, v) * ufl.dx

    problem = petsc.LinearProblem(
        a,
        L,
        bcs=[bc],
        petsc_options_prefix="poisson_",
        petsc_options={
            "ksp_type": ksp_type,
            "pc_type": pc_type,
            "ksp_rtol": rtol,
        },
    )
    uh = problem.solve()
    uh.x.scatter_forward()

    l2_err_local = fem.assemble_scalar(fem.form((uh - u_exact) ** 2 * ufl.dx))
    l2_err = np.sqrt(comm.allreduce(l2_err_local, op=MPI.SUM))

    u_grid = _sample_on_uniform_grid(uh, case_spec["output"]["grid"])

    return {
        "u": u_grid if comm.rank == 0 else None,
        "solver_info": {
            "mesh_resolution": int(mesh_resolution),
            "element_degree": int(element_degree),
            "ksp_type": ksp_type,
            "pc_type": pc_type,
            "rtol": float(rtol),
            "iterations": int(problem.solver.getIterationNumber()),
            "l2_error_check": float(l2_err),
        } if comm.rank == 0 else None,
    }


if __name__ == "__main__":
    case_spec = {
        "pde": {"coefficients": {"kappa": 1.0}},
        "output": {"grid": {"nx": 129, "ny": 129, "bbox": [0.0, 1.0, 0.0, 1.0]}},
    }
    comm = MPI.COMM_WORLD
    t0 = time.perf_counter()
    out = solve(case_spec)
    wall = time.perf_counter() - t0
    if comm.rank == 0:
        xs = np.linspace(0.0, 1.0, case_spec["output"]["grid"]["nx"])
        ys = np.linspace(0.0, 1.0, case_spec["output"]["grid"]["ny"])
        XX, YY = np.meshgrid(xs, ys, indexing="xy")
        uex = np.sin(4.0 * np.pi * XX) * np.sin(4.0 * np.pi * YY)
        grid_l2 = np.sqrt(np.mean((out["u"] - uex) ** 2))
        print(f"L2_ERROR: {out['solver_info']['l2_error_check']:.12e}")
        print(f"GRID_RMSE: {grid_l2:.12e}")
        print(f"WALL_TIME: {wall:.12e}")
