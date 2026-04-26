import time
import math
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
# special_notes: variable_coeff
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


def _build_problem(n: int, degree: int = 2, comm=MPI.COMM_WORLD):
    domain = mesh.create_unit_square(comm, n, n, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", degree))

    x = ufl.SpatialCoordinate(domain)
    kappa_expr = 1.0 + 50.0 * ufl.exp(-150.0 * ((x[0] - 0.5) ** 2 + (x[1] - 0.5) ** 2))
    f_expr = ufl.exp(-250.0 * ((x[0] - 0.4) ** 2 + (x[1] - 0.6) ** 2))

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    a = ufl.inner(kappa_expr * ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = ufl.inner(f_expr, v) * ufl.dx

    fdim = domain.topology.dim - 1
    facets = mesh.locate_entities_boundary(domain, fdim, lambda X: np.ones(X.shape[1], dtype=bool))
    dofs = fem.locate_dofs_topological(V, fdim, facets)
    bc = fem.dirichletbc(ScalarType(0.0), dofs, V)

    return domain, V, a, L, bc, kappa_expr, f_expr


def _solve_once(n: int, degree: int = 2, comm=MPI.COMM_WORLD):
    domain, V, a, L, bc, kappa_expr, f_expr = _build_problem(n, degree, comm)

    opts = {
        "ksp_type": "preonly",
        "pc_type": "lu",
        "ksp_rtol": 1.0e-10,
    }
    problem = petsc.LinearProblem(
        a, L, bcs=[bc], petsc_options=opts, petsc_options_prefix=f"poisson_{n}_"
    )

    t0 = time.perf_counter()
    uh = problem.solve()
    uh.x.scatter_forward()
    solve_time = time.perf_counter() - t0

    # Residual-based verification: ||div(k grad u)+f||_L2 estimated by projecting cellwise strong residual surrogate
    V0 = fem.functionspace(domain, ("DG", 0))
    w = ufl.TestFunction(V0)
    q = ufl.TrialFunction(V0)
    cell_vol = ufl.CellVolume(domain)
    strong_res = f_expr + ufl.div(kappa_expr * ufl.grad(uh))
    a_r = ufl.inner(q, w) * ufl.dx
    L_r = ufl.inner(strong_res, w) * ufl.dx
    rprob = petsc.LinearProblem(
        a_r, L_r, petsc_options={"ksp_type": "preonly", "pc_type": "jacobi"}, petsc_options_prefix=f"res_{n}_"
    )
    rh = rprob.solve()
    rh.x.scatter_forward()
    residual_l2_local = fem.assemble_scalar(fem.form(ufl.inner(rh, rh) * ufl.dx))
    residual_l2 = math.sqrt(domain.comm.allreduce(residual_l2_local, op=MPI.SUM))

    return {
        "domain": domain,
        "V": V,
        "u": uh,
        "solve_time": solve_time,
        "residual_l2": residual_l2,
        "solver_info": {
            "mesh_resolution": int(n),
            "element_degree": int(degree),
            "ksp_type": "preonly",
            "pc_type": "lu",
            "rtol": 1.0e-10,
            "iterations": 1,
        },
    }


def _sample_on_grid(domain, uh, grid_spec: dict):
    nx = int(grid_spec["nx"])
    ny = int(grid_spec["ny"])
    xmin, xmax, ymin, ymax = grid_spec["bbox"]

    xs = np.linspace(xmin, xmax, nx)
    ys = np.linspace(ymin, ymax, ny)
    XX, YY = np.meshgrid(xs, ys, indexing="xy")
    pts = np.column_stack([XX.ravel(), YY.ravel(), np.zeros(nx * ny, dtype=np.float64)])

    tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(tree, pts)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, pts)

    local_vals = np.full(nx * ny, np.nan, dtype=np.float64)
    points_on_proc = []
    cells_on_proc = []
    eval_map = []

    for i in range(pts.shape[0]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(pts[i])
            cells_on_proc.append(links[0])
            eval_map.append(i)

    if len(points_on_proc) > 0:
        vals = uh.eval(np.array(points_on_proc, dtype=np.float64), np.array(cells_on_proc, dtype=np.int32))
        vals = np.asarray(vals).reshape(len(points_on_proc), -1)[:, 0]
        local_vals[np.array(eval_map, dtype=np.int32)] = vals

    gathered = domain.comm.gather(local_vals, root=0)
    if domain.comm.rank == 0:
        global_vals = np.full(nx * ny, np.nan, dtype=np.float64)
        for arr in gathered:
            mask = np.isnan(global_vals) & ~np.isnan(arr)
            global_vals[mask] = arr[mask]
        if np.isnan(global_vals).any():
            # Points on the boundary may occasionally fail collision ownership on some partitions;
            # repair using zeros only for any remaining NaNs on the Dirichlet boundary.
            rem = np.isnan(global_vals)
            bx = pts[:, 0]
            by = pts[:, 1]
            on_bdry = rem & (
                np.isclose(bx, xmin) | np.isclose(bx, xmax) | np.isclose(by, ymin) | np.isclose(by, ymax)
            )
            global_vals[on_bdry] = 0.0
            if np.isnan(global_vals).any():
                raise RuntimeError("Failed to evaluate FEM solution at some requested output points.")
        return global_vals.reshape(ny, nx)
    return None


def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD
    grid = case_spec["output"]["grid"]
    nx_out = int(grid["nx"])
    ny_out = int(grid["ny"])

    # Adaptive accuracy choice aimed at using available time wisely.
    base = max(48, int(max(nx_out, ny_out) * 1.5))
    candidates = [base, min(base + 16, 112)]
    degree = 2

    best = None
    wall0 = time.perf_counter()
    for i, n in enumerate(candidates):
        result = _solve_once(n=n, degree=degree, comm=comm)
        if best is None:
            best = result
        else:
            # Prefer finer mesh if affordable and residual improves.
            elapsed = time.perf_counter() - wall0
            if elapsed < 4.5 and result["residual_l2"] <= best["residual_l2"] * 1.2:
                best = result

    # Optional mesh-doubling verification if there is enough remaining budget
    verification = {}
    if len(candidates) >= 2 and best["solver_info"]["mesh_resolution"] == candidates[-1]:
        coarse = _solve_once(n=candidates[0], degree=degree, comm=comm)
        gspec = {"nx": min(64, nx_out), "ny": min(64, ny_out), "bbox": grid["bbox"]}
        uf = _sample_on_grid(best["domain"], best["u"], gspec)
        uc = _sample_on_grid(coarse["domain"], coarse["u"], gspec)
        if comm.rank == 0:
            diff = np.linalg.norm(uf - uc) / np.sqrt(uf.size)
            verification["mesh_doubling_rms_diff"] = float(diff)
            verification["coarse_residual_l2"] = float(coarse["residual_l2"])
        verification["fine_residual_l2"] = float(best["residual_l2"])

    u_grid = _sample_on_grid(best["domain"], best["u"], grid)

    if comm.rank == 0:
        solver_info = dict(best["solver_info"])
        solver_info["verification"] = verification
        return {"u": np.asarray(u_grid, dtype=np.float64), "solver_info": solver_info}
    else:
        return {"u": None, "solver_info": dict(best["solver_info"])}


if __name__ == "__main__":
    case_spec = {
        "output": {"grid": {"nx": 64, "ny": 64, "bbox": [0.0, 1.0, 0.0, 1.0]}},
        "pde": {"time": None},
    }
    out = solve(case_spec)
    if MPI.COMM_WORLD.rank == 0:
        print(out["u"].shape)
        print(out["solver_info"])
