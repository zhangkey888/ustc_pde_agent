import time
import math
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
import ufl
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
from basix.ufl import element as basix_element, mixed_element as basix_mixed_element

# ```DIAGNOSIS
# equation_type: stokes
# spatial_dim: 2
# domain_geometry: rectangle
# unknowns: vector+scalar
# coupling: saddle_point
# linearity: linear
# time_dependence: steady
# stiffness: N/A
# dominant_physics: diffusion
# peclet_or_reynolds: low
# solution_regularity: smooth
# bc_type: all_dirichlet
# special_notes: pressure_pinning, manufactured_solution
# ```
#
# ```METHOD
# spatial_method: fem
# element_or_basis: Taylor-Hood_P2P1
# stabilization: none
# time_method: none
# nonlinear_solver: none
# linear_solver: minres
# preconditioner: hypre
# special_treatment: pressure_pinning
# pde_skill: stokes
# ```

ScalarType = PETSc.ScalarType


def _u_exact_expr(x):
    return ufl.as_vector(
        [
            ufl.pi * ufl.cos(ufl.pi * x[1]) * ufl.sin(ufl.pi * x[0]),
            -ufl.pi * ufl.cos(ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1]),
        ]
    )


def _p_exact_expr(x):
    return ufl.cos(ufl.pi * x[0]) * ufl.cos(ufl.pi * x[1])


def _forcing_expr(msh, nu):
    x = ufl.SpatialCoordinate(msh)
    uex = _u_exact_expr(x)
    pex = _p_exact_expr(x)
    return -nu * ufl.div(ufl.grad(uex)) + ufl.grad(pex)


def _velocity_callable():
    def g(x):
        vals = np.zeros((2, x.shape[1]), dtype=np.float64)
        vals[0] = np.pi * np.cos(np.pi * x[1]) * np.sin(np.pi * x[0])
        vals[1] = -np.pi * np.cos(np.pi * x[0]) * np.sin(np.pi * x[1])
        return vals

    return g


def _sample_velocity_magnitude(u_fun, msh, grid):
    nx = int(grid["nx"])
    ny = int(grid["ny"])
    xmin, xmax, ymin, ymax = [float(v) for v in grid["bbox"]]

    xs = np.linspace(xmin, xmax, nx)
    ys = np.linspace(ymin, ymax, ny)
    XX, YY = np.meshgrid(xs, ys, indexing="xy")
    pts = np.column_stack([XX.ravel(), YY.ravel(), np.zeros(nx * ny, dtype=np.float64)])

    bb = geometry.bb_tree(msh, msh.topology.dim)
    candidates = geometry.compute_collisions_points(bb, pts)
    colliding = geometry.compute_colliding_cells(msh, candidates, pts)

    owned_ids = []
    owned_pts = []
    owned_cells = []
    for i in range(pts.shape[0]):
        links = colliding.links(i)
        if len(links) > 0:
            owned_ids.append(i)
            owned_pts.append(pts[i])
            owned_cells.append(links[0])

    local_mag = np.full(pts.shape[0], np.nan, dtype=np.float64)
    if owned_pts:
        vals = u_fun.eval(np.asarray(owned_pts, dtype=np.float64), np.asarray(owned_cells, dtype=np.int32))
        local_mag[np.asarray(owned_ids, dtype=np.int32)] = np.linalg.norm(vals, axis=1)

    comm = msh.comm
    gathered = comm.gather(local_mag, root=0)
    if comm.rank == 0:
        merged = np.full(pts.shape[0], np.nan, dtype=np.float64)
        for arr in gathered:
            mask = np.isfinite(arr)
            merged[mask] = arr[mask]
        merged[np.isnan(merged)] = 0.0
        out = merged.reshape(ny, nx)
    else:
        out = None
    return comm.bcast(out, root=0)


def _solve_with_resolution(mesh_resolution, nu, ksp_type="minres", pc_type="hypre", rtol=1e-9):
    comm = MPI.COMM_WORLD
    msh = mesh.create_unit_square(comm, mesh_resolution, mesh_resolution, cell_type=mesh.CellType.triangle)
    gdim = msh.geometry.dim
    cell_name = msh.topology.cell_name()

    vel_el = basix_element("Lagrange", cell_name, 2, shape=(gdim,))
    pre_el = basix_element("Lagrange", cell_name, 1)
    W = fem.functionspace(msh, basix_mixed_element([vel_el, pre_el]))
    V, _ = W.sub(0).collapse()
    Q, _ = W.sub(1).collapse()

    u, p = ufl.TrialFunctions(W)
    v, q = ufl.TestFunctions(W)

    f = _forcing_expr(msh, nu)
    a = (
        2.0 * nu * ufl.inner(ufl.sym(ufl.grad(u)), ufl.sym(ufl.grad(v))) * ufl.dx
        - p * ufl.div(v) * ufl.dx
        + q * ufl.div(u) * ufl.dx
    )
    L = ufl.inner(f, v) * ufl.dx

    fdim = msh.topology.dim - 1
    bfacets = mesh.locate_entities_boundary(msh, fdim, lambda x: np.ones(x.shape[1], dtype=bool))

    u_bc = fem.Function(V)
    u_bc.interpolate(_velocity_callable())
    dofs_u = fem.locate_dofs_topological((W.sub(0), V), fdim, bfacets)
    bc_u = fem.dirichletbc(u_bc, dofs_u, W.sub(0))

    p_dofs = fem.locate_dofs_geometrical(
        (W.sub(1), Q), lambda x: np.isclose(x[0], 0.0) & np.isclose(x[1], 0.0)
    )
    p0 = fem.Function(Q)
    p0.x.array[:] = 0.0
    bc_p = fem.dirichletbc(p0, p_dofs, W.sub(1))

    try:
        problem = petsc.LinearProblem(
            a,
            L,
            bcs=[bc_u, bc_p],
            petsc_options_prefix=f"stokes_{mesh_resolution}_",
            petsc_options={"ksp_type": ksp_type, "pc_type": pc_type, "ksp_rtol": rtol},
        )
        wh = problem.solve()
    except Exception:
        problem = petsc.LinearProblem(
            a,
            L,
            bcs=[bc_u, bc_p],
            petsc_options_prefix=f"stokes_fallback_{mesh_resolution}_",
            petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
        )
        wh = problem.solve()
        ksp_type = "preonly"
        pc_type = "lu"

    uh = wh.sub(0).collapse()
    ph = wh.sub(1).collapse()

    x = ufl.SpatialCoordinate(msh)
    u_exact = _u_exact_expr(x)
    p_exact = _p_exact_expr(x)

    err_u_sq = fem.assemble_scalar(fem.form(ufl.inner(uh - u_exact, uh - u_exact) * ufl.dx))
    err_p_sq = fem.assemble_scalar(fem.form((ph - p_exact) * (ph - p_exact) * ufl.dx))
    norm_u_sq = fem.assemble_scalar(fem.form(ufl.inner(u_exact, u_exact) * ufl.dx))
    norm_p_sq = fem.assemble_scalar(fem.form(p_exact * p_exact * ufl.dx))

    err_u = math.sqrt(comm.allreduce(err_u_sq, op=MPI.SUM))
    err_p = math.sqrt(comm.allreduce(err_p_sq, op=MPI.SUM))
    norm_u = math.sqrt(comm.allreduce(norm_u_sq, op=MPI.SUM))
    norm_p = math.sqrt(comm.allreduce(norm_p_sq, op=MPI.SUM))
    rel_err = err_u / (norm_u + 1e-16) + err_p / (norm_p + 1e-16)

    iterations = 0
    try:
        iterations = int(problem.solver.getIterationNumber())
    except Exception:
        iterations = 0

    return {
        "mesh": msh,
        "u": uh,
        "p": ph,
        "rel_error_metric": float(rel_err),
        "u_l2_error": float(err_u),
        "p_l2_error": float(err_p),
        "ksp_type": ksp_type,
        "pc_type": pc_type,
        "rtol": float(rtol),
        "iterations": int(iterations),
        "mesh_resolution": int(mesh_resolution),
        "element_degree": 2,
    }


def solve(case_spec: dict) -> dict:
    nu = float(case_spec.get("pde", {}).get("nu", case_spec.get("pde", {}).get("viscosity", 0.1)))
    grid = case_spec["output"]["grid"]

    time_budget = 6.844
    t0 = time.perf_counter()
    results = None

    for n in [20, 28, 36]:
        results = _solve_with_resolution(n, nu, ksp_type="minres", pc_type="hypre", rtol=1e-9)
        if time.perf_counter() - t0 > 0.78 * time_budget:
            break

    u_grid = _sample_velocity_magnitude(results["u"], results["mesh"], grid)
    solver_info = {
        "mesh_resolution": results["mesh_resolution"],
        "element_degree": results["element_degree"],
        "ksp_type": results["ksp_type"],
        "pc_type": results["pc_type"],
        "rtol": results["rtol"],
        "iterations": results["iterations"],
        "accuracy_verification": {
            "u_l2_error": results["u_l2_error"],
            "p_l2_error": results["p_l2_error"],
            "combined_relative_error_metric": results["rel_error_metric"],
        },
    }
    return {"u": u_grid, "solver_info": solver_info}


if __name__ == "__main__":
    case_spec = {
        "pde": {"nu": 0.1, "time": None},
        "output": {"grid": {"nx": 32, "ny": 32, "bbox": [0.0, 1.0, 0.0, 1.0]}},
    }
    out = solve(case_spec)
    if MPI.COMM_WORLD.rank == 0:
        print(out["u"].shape)
        print(out["solver_info"])
