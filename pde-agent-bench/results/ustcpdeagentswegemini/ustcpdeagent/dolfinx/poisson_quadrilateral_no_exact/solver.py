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


def _solve_poisson(n, degree, ksp_type="cg", pc_type="hypre", rtol=1.0e-10):
    comm = MPI.COMM_WORLD
    domain = mesh.create_rectangle(
        comm,
        [np.array([0.0, 0.0]), np.array([1.0, 1.0])],
        [n, n],
        cell_type=mesh.CellType.quadrilateral,
    )
    V = fem.functionspace(domain, ("Lagrange", degree))

    tdim = domain.topology.dim
    fdim = tdim - 1
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
    )
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc = fem.dirichletbc(ScalarType(0.0), boundary_dofs, V)

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    kappa = fem.Constant(domain, ScalarType(1.0))
    f = fem.Constant(domain, ScalarType(1.0))

    a = ufl.inner(kappa * ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = f * v * ufl.dx

    try:
        problem = petsc.LinearProblem(
            a,
            L,
            bcs=[bc],
            petsc_options={
                "ksp_type": ksp_type,
                "pc_type": pc_type,
                "ksp_rtol": rtol,
            },
            petsc_options_prefix="poisson_",
        )
        uh = problem.solve()
        uh.x.scatter_forward()
        solver = problem.solver
        info = {
            "mesh_resolution": int(n),
            "element_degree": int(degree),
            "ksp_type": str(solver.getType()),
            "pc_type": str(solver.getPC().getType()),
            "rtol": float(rtol),
            "iterations": int(solver.getIterationNumber()),
        }
        return domain, uh, info
    except Exception:
        problem = petsc.LinearProblem(
            a,
            L,
            bcs=[bc],
            petsc_options={
                "ksp_type": "preonly",
                "pc_type": "lu",
                "ksp_rtol": rtol,
            },
            petsc_options_prefix="poisson_fallback_",
        )
        uh = problem.solve()
        uh.x.scatter_forward()
        solver = problem.solver
        info = {
            "mesh_resolution": int(n),
            "element_degree": int(degree),
            "ksp_type": str(solver.getType()),
            "pc_type": str(solver.getPC().getType()),
            "rtol": float(rtol),
            "iterations": int(solver.getIterationNumber()),
        }
        return domain, uh, info


def _sample_function(domain, uh, grid_spec):
    nx = int(grid_spec["nx"])
    ny = int(grid_spec["ny"])
    xmin, xmax, ymin, ymax = grid_spec["bbox"]

    xs = np.linspace(xmin, xmax, nx)
    ys = np.linspace(ymin, ymax, ny)
    XX, YY = np.meshgrid(xs, ys, indexing="xy")
    pts = np.column_stack([XX.ravel(), YY.ravel(), np.zeros(nx * ny)])

    tree = geometry.bb_tree(domain, domain.topology.dim)
    candidates = geometry.compute_collisions_points(tree, pts)
    colliding = geometry.compute_colliding_cells(domain, candidates, pts)

    local = np.full(nx * ny, np.nan, dtype=np.float64)
    points_on_proc = []
    cells_on_proc = []
    map_ids = []

    for i in range(nx * ny):
        links = colliding.links(i)
        if len(links) > 0:
            points_on_proc.append(pts[i])
            cells_on_proc.append(links[0])
            map_ids.append(i)

    if points_on_proc:
        vals = uh.eval(
            np.asarray(points_on_proc, dtype=np.float64),
            np.asarray(cells_on_proc, dtype=np.int32),
        )
        local[np.asarray(map_ids, dtype=np.int32)] = np.asarray(vals).reshape(-1)

    gathered = domain.comm.gather(local, root=0)
    if domain.comm.rank == 0:
        global_vals = np.full(nx * ny, np.nan, dtype=np.float64)
        for arr in gathered:
            mask = ~np.isnan(arr)
            global_vals[mask] = arr[mask]
        global_vals[np.isnan(global_vals)] = 0.0
        return global_vals.reshape((ny, nx))
    return None


def _verification(candidate_n, candidate_degree):
    comm = MPI.COMM_WORLD
    coarse_domain, coarse_u, _ = _solve_poisson(candidate_n, candidate_degree)
    ref_n = min(160, max(candidate_n + 16, 2 * candidate_n))
    ref_degree = min(3, candidate_degree + 1)
    ref_domain, ref_u, _ = _solve_poisson(ref_n, ref_degree)
    grid = {"nx": 65, "ny": 65, "bbox": [0.0, 1.0, 0.0, 1.0]}
    uc = _sample_function(coarse_domain, coarse_u, grid)
    ur = _sample_function(ref_domain, ref_u, grid)
    if comm.rank == 0:
        diff = uc - ur
        abs_err = float(np.sqrt(np.mean(diff**2)))
        rel_err = float(abs_err / (np.sqrt(np.mean(ur**2)) + 1.0e-14))
        return {"reference_abs_l2_grid": abs_err, "reference_rel_l2_grid": rel_err}
    return {}


def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD
    t0 = time.perf_counter()

    time_budget = 11.242
    degree = 2
    n = 80

    domain, uh, solver_info = _solve_poisson(n, degree, ksp_type="cg", pc_type="hypre", rtol=1.0e-10)
    elapsed = time.perf_counter() - t0

    if elapsed < 0.5 * time_budget:
        try:
            domain2, uh2, solver_info2 = _solve_poisson(108, 2, ksp_type="cg", pc_type="hypre", rtol=1.0e-10)
            if (time.perf_counter() - t0) < 0.95 * time_budget:
                domain, uh, solver_info = domain2, uh2, solver_info2
        except Exception:
            pass

    verification = _verification(solver_info["mesh_resolution"], solver_info["element_degree"])
    grid_u = _sample_function(domain, uh, case_spec["output"]["grid"])

    if comm.rank == 0:
        solver_info.update(verification)
        return {"u": grid_u, "solver_info": solver_info}
    return {"u": None, "solver_info": solver_info}
