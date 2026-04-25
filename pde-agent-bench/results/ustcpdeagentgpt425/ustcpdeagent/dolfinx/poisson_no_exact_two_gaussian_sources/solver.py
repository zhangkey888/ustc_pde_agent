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
# special_notes: none
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


def _build_forms(n, degree, comm):
    domain = mesh.create_unit_square(comm, n, n, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", degree))

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    x = ufl.SpatialCoordinate(domain)

    f = (
        ufl.exp(-250.0 * ((x[0] - 0.25) ** 2 + (x[1] - 0.25) ** 2))
        + ufl.exp(-250.0 * ((x[0] - 0.75) ** 2 + (x[1] - 0.70) ** 2))
    )

    kappa = fem.Constant(domain, ScalarType(1.0))
    a = kappa * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = ufl.inner(f, v) * ufl.dx

    fdim = domain.topology.dim - 1
    facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
    )
    dofs = fem.locate_dofs_topological(V, fdim, facets)
    bc = fem.dirichletbc(ScalarType(0.0), dofs, V)
    return domain, V, a, L, [bc]


def _solve_once(n, degree, comm, prefix):
    domain, V, a, L, bcs = _build_forms(n, degree, comm)
    try:
        problem = petsc.LinearProblem(
            a,
            L,
            bcs=bcs,
            petsc_options_prefix=prefix,
            petsc_options={
                "ksp_type": "cg",
                "pc_type": "hypre",
                "ksp_rtol": 1.0e-10,
                "ksp_atol": 1.0e-12,
                "ksp_max_it": 10000,
            },
        )
        uh = problem.solve()
        uh.x.scatter_forward()
        ksp = problem.solver
        meta = {
            "ksp_type": ksp.getType(),
            "pc_type": ksp.getPC().getType(),
            "rtol": float(ksp.getTolerances()[0]),
            "iterations": int(ksp.getIterationNumber()),
        }
    except Exception:
        problem = petsc.LinearProblem(
            a,
            L,
            bcs=bcs,
            petsc_options_prefix=prefix + "lu_",
            petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
        )
        uh = problem.solve()
        uh.x.scatter_forward()
        ksp = problem.solver
        meta = {
            "ksp_type": ksp.getType(),
            "pc_type": ksp.getPC().getType(),
            "rtol": 0.0,
            "iterations": int(ksp.getIterationNumber()),
        }

    # Accuracy verification: algebraic residual norm
    v = ufl.TestFunction(V)
    x = ufl.SpatialCoordinate(domain)
    f = (
        ufl.exp(-250.0 * ((x[0] - 0.25) ** 2 + (x[1] - 0.25) ** 2))
        + ufl.exp(-250.0 * ((x[0] - 0.75) ** 2 + (x[1] - 0.70) ** 2))
    )
    res_form = fem.form((ufl.inner(ufl.grad(uh), ufl.grad(v)) - f * v) * ufl.dx)
    b = petsc.create_vector(res_form.function_spaces)
    with b.localForm() as loc:
        loc.set(0.0)
    petsc.assemble_vector(b, res_form)
    b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    petsc.set_bc(b, bcs)
    meta["residual_l2_norm"] = float(b.norm())

    return domain, uh, meta


def _sample_solution(domain, uh, grid):
    nx = int(grid["nx"])
    ny = int(grid["ny"])
    xmin, xmax, ymin, ymax = grid["bbox"]

    xs = np.linspace(xmin, xmax, nx)
    ys = np.linspace(ymin, ymax, ny)
    X, Y = np.meshgrid(xs, ys, indexing="xy")
    pts = np.column_stack([X.ravel(), Y.ravel(), np.zeros(nx * ny, dtype=np.float64)])

    tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(tree, pts)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, pts)

    values_local = np.full(nx * ny, np.nan, dtype=np.float64)
    eval_points = []
    eval_cells = []
    eval_ids = []
    for i in range(pts.shape[0]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            eval_points.append(pts[i])
            eval_cells.append(links[0])
            eval_ids.append(i)

    if eval_points:
        vals = uh.eval(np.asarray(eval_points, dtype=np.float64),
                       np.asarray(eval_cells, dtype=np.int32))
        values_local[np.asarray(eval_ids, dtype=np.int64)] = np.asarray(vals).reshape(-1)

    gathered = domain.comm.gather(values_local, root=0)
    if domain.comm.rank == 0:
        values = np.full(nx * ny, np.nan, dtype=np.float64)
        for arr in gathered:
            mask = np.isnan(values) & ~np.isnan(arr)
            values[mask] = arr[mask]
        values = np.nan_to_num(values, nan=0.0)
        return values.reshape((ny, nx))
    return None


def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD
    t0 = time.perf_counter()

    grid = case_spec["output"]["grid"]
    target = max(int(grid["nx"]), int(grid["ny"]))

    # Start accurately and refine further if there is clear time headroom.
    if target <= 64:
        n = 96
    elif target <= 128:
        n = 128
    else:
        n = 160
    degree = 2

    domain, uh, meta = _solve_once(n, degree, comm, "poisson_")
    elapsed = time.perf_counter() - t0

    if elapsed < 5.0 and n < 192:
        try:
            domain2, uh2, meta2 = _solve_once(192, degree, comm, "poisson_refined_")
            domain, uh, meta, n = domain2, uh2, meta2, 192
        except Exception:
            pass

    u_grid = _sample_solution(domain, uh, grid)

    return {
        "u": u_grid,
        "solver_info": {
            "mesh_resolution": int(n),
            "element_degree": int(degree),
            "ksp_type": str(meta["ksp_type"]),
            "pc_type": str(meta["pc_type"]),
            "rtol": float(meta["rtol"]),
            "iterations": int(meta["iterations"]),
        },
    }


if __name__ == "__main__":
    case = {"output": {"grid": {"nx": 64, "ny": 64, "bbox": [0.0, 1.0, 0.0, 1.0]}}}
    out = solve(case)
    if MPI.COMM_WORLD.rank == 0:
        print(out["u"].shape)
        print(out["solver_info"])
