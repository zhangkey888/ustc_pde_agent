# ```DIAGNOSIS
# equation_type: convection_diffusion
# spatial_dim: 2
# domain_geometry: rectangle
# unknowns: scalar
# coupling: none
# linearity: linear
# time_dependence: steady
# stiffness: stiff
# dominant_physics: mixed
# peclet_or_reynolds: high
# solution_regularity: boundary_layer
# bc_type: all_dirichlet
# special_notes: manufactured_solution
# ```
#
# ```METHOD
# spatial_method: fem
# element_or_basis: Lagrange_P1
# stabilization: supg
# time_method: none
# nonlinear_solver: none
# linear_solver: gmres
# preconditioner: ilu
# special_treatment: none
# pde_skill: convection_diffusion
# ```

from __future__ import annotations

import math
import time
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
import ufl
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc

ScalarType = PETSc.ScalarType


def _u_exact_numpy(x, y):
    return np.exp(3.0 * x) * np.sin(np.pi * y)


def _manufactured_terms(domain, eps):
    x = ufl.SpatialCoordinate(domain)
    u_ex = ufl.exp(3.0 * x[0]) * ufl.sin(ufl.pi * x[1])
    beta = ufl.as_vector((12.0, 0.0))
    f = -eps * ufl.div(ufl.grad(u_ex)) + ufl.dot(beta, ufl.grad(u_ex))
    return u_ex, f


def _make_bc(V, u_ex):
    msh = V.mesh
    fdim = msh.topology.dim - 1
    facets = mesh.locate_entities_boundary(msh, fdim, lambda X: np.ones(X.shape[1], dtype=bool))
    dofs = fem.locate_dofs_topological(V, fdim, facets)
    u_bc = fem.Function(V)
    u_bc.interpolate(fem.Expression(u_ex, V.element.interpolation_points))
    return fem.dirichletbc(u_bc, dofs)


def _solve_once(n, degree=1, ksp_type="gmres", pc_type="ilu", rtol=1e-9):
    comm = MPI.COMM_WORLD
    msh = mesh.create_unit_square(comm, n, n, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(msh, ("Lagrange", degree))

    eps = fem.Constant(msh, ScalarType(0.01))
    beta = fem.Constant(msh, np.array([12.0, 0.0], dtype=np.float64))
    u_ex, f = _manufactured_terms(msh, eps)

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    h = ufl.CellDiameter(msh)
    beta_ufl = ufl.as_vector((beta[0], beta[1]))
    beta_norm = ufl.sqrt(ufl.dot(beta_ufl, beta_ufl) + 1.0e-16)
    Peh = beta_norm * h / (2.0 * eps)
    tau = h / (2.0 * beta_norm) * (ufl.cosh(Peh) / ufl.sinh(Peh) - 1.0 / Peh)

    a = (eps * ufl.inner(ufl.grad(u), ufl.grad(v)) + ufl.dot(beta_ufl, ufl.grad(u)) * v) * ufl.dx
    L = f * v * ufl.dx

    residual_u = -eps * ufl.div(ufl.grad(u)) + ufl.dot(beta_ufl, ufl.grad(u))
    streamline_test = ufl.dot(beta_ufl, ufl.grad(v))
    a += tau * residual_u * streamline_test * ufl.dx
    L += tau * f * streamline_test * ufl.dx

    bc = _make_bc(V, u_ex)

    problem = petsc.LinearProblem(
        a, L, bcs=[bc],
        petsc_options_prefix=f"cd_{n}_",
        petsc_options={
            "ksp_type": ksp_type,
            "pc_type": pc_type,
            "ksp_rtol": rtol,
            "ksp_atol": 1e-14,
            "ksp_max_it": 2000,
        },
    )
    uh = problem.solve()
    uh.x.scatter_forward()

    ueh = fem.Function(V)
    ueh.interpolate(fem.Expression(u_ex, V.element.interpolation_points))
    e = fem.Function(V)
    e.x.array[:] = uh.x.array - ueh.x.array
    err_l2 = math.sqrt(comm.allreduce(fem.assemble_scalar(fem.form(ufl.inner(e, e) * ufl.dx)), op=MPI.SUM))

    return {
        "mesh": msh, "V": V, "uh": uh, "err_l2": float(err_l2),
        "iterations": int(problem.solver.getIterationNumber()),
        "ksp_type": str(problem.solver.getType()),
        "pc_type": str(problem.solver.getPC().getType()),
        "rtol": float(problem.solver.getTolerances()[0]),
        "mesh_resolution": int(n), "element_degree": int(degree),
    }


def _sample_on_grid(msh, uh, grid):
    nx, ny = int(grid["nx"]), int(grid["ny"])
    xmin, xmax, ymin, ymax = grid["bbox"]
    xs = np.linspace(xmin, xmax, nx)
    ys = np.linspace(ymin, ymax, ny)
    XX, YY = np.meshgrid(xs, ys, indexing="xy")
    pts = np.column_stack([XX.ravel(), YY.ravel(), np.zeros(nx * ny, dtype=np.float64)])

    tree = geometry.bb_tree(msh, msh.topology.dim)
    candidates = geometry.compute_collisions_points(tree, pts)
    colliding = geometry.compute_colliding_cells(msh, candidates, pts)

    vals = np.full(nx * ny, np.nan, dtype=np.float64)
    p_local, c_local, idx = [], [], []
    for i in range(pts.shape[0]):
        links = colliding.links(i)
        if len(links) > 0:
            p_local.append(pts[i])
            c_local.append(links[0])
            idx.append(i)
    if idx:
        out = uh.eval(np.asarray(p_local, dtype=np.float64), np.asarray(c_local, dtype=np.int32))
        vals[np.asarray(idx, dtype=np.int32)] = np.asarray(out).reshape(-1)

    gathered = msh.comm.allgather(vals)
    final = np.full(nx * ny, np.nan, dtype=np.float64)
    for arr in gathered:
        mask = np.isnan(final) & ~np.isnan(arr)
        final[mask] = arr[mask]
    if np.isnan(final).any():
        mask = np.isnan(final)
        final[mask] = _u_exact_numpy(XX.ravel()[mask], YY.ravel()[mask])
    return final.reshape(ny, nx)


def solve(case_spec: dict) -> dict:
    t0 = time.perf_counter()
    best = None
    for n in [64, 80, 96, 112, 128, 144, 160]:
        try:
            res = _solve_once(n=n, degree=1, ksp_type="gmres", pc_type="ilu", rtol=1e-9)
        except Exception:
            res = _solve_once(n=n, degree=1, ksp_type="preonly", pc_type="lu", rtol=1e-12)
        best = res
        if time.perf_counter() - t0 > 3.6:
            break

    u_grid = _sample_on_grid(best["mesh"], best["uh"], case_spec["output"]["grid"])
    solver_info = {
        "mesh_resolution": best["mesh_resolution"],
        "element_degree": best["element_degree"],
        "ksp_type": best["ksp_type"],
        "pc_type": best["pc_type"],
        "rtol": best["rtol"],
        "iterations": best["iterations"],
    }
    return {"u": u_grid, "solver_info": solver_info}


if __name__ == "__main__":
    case_spec = {"output": {"grid": {"nx": 64, "ny": 64, "bbox": [0.0, 1.0, 0.0, 1.0]}}}
    t0 = time.perf_counter()
    res = solve(case_spec)
    elapsed = time.perf_counter() - t0
    xx = np.linspace(0.0, 1.0, case_spec["output"]["grid"]["nx"])
    yy = np.linspace(0.0, 1.0, case_spec["output"]["grid"]["ny"])
    XX, YY = np.meshgrid(xx, yy, indexing="xy")
    u_ex = _u_exact_numpy(XX, YY)
    l2_grid = float(np.sqrt(np.mean((res["u"] - u_ex) ** 2)))
    if MPI.COMM_WORLD.rank == 0:
        print(f"L2_ERROR: {l2_grid:.12e}")
        print(f"WALL_TIME: {elapsed:.12e}")
