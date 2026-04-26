"""
DIAGNOSIS
equation_type: poisson
spatial_dim: 2
domain_geometry: rectangle
unknowns: scalar
coupling: none
linearity: linear
time_dependence: steady
stiffness: N/A
dominant_physics: diffusion
peclet_or_reynolds: N/A
solution_regularity: smooth
bc_type: all_dirichlet
special_notes: none

METHOD
spatial_method: fem
element_or_basis: Lagrange_P2
stabilization: none
time_method: none
nonlinear_solver: none
linear_solver: cg
preconditioner: hypre
special_treatment: none
pde_skill: poisson
"""

from __future__ import annotations
import time
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
import ufl
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc

ScalarType = PETSc.ScalarType


def _build_and_solve(n: int, degree: int, rtol: float):
    comm = MPI.COMM_WORLD
    msh = mesh.create_unit_square(comm, n, n, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(msh, ("Lagrange", degree))
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    x = ufl.SpatialCoordinate(msh)
    f = ufl.exp(-200.0 * ((x[0] - 0.25) ** 2 + (x[1] - 0.75) ** 2))
    a = ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = f * v * ufl.dx

    fdim = msh.topology.dim - 1
    facets = mesh.locate_entities_boundary(msh, fdim, lambda X: np.ones(X.shape[1], dtype=bool))
    dofs = fem.locate_dofs_topological(V, fdim, facets)
    bc = fem.dirichletbc(ScalarType(0.0), dofs, V)

    opts = {"ksp_type": "cg", "pc_type": "hypre", "ksp_rtol": rtol, "ksp_atol": 1e-14}
    try:
        problem = petsc.LinearProblem(a, L, bcs=[bc], petsc_options_prefix=f"poisson_{n}_", petsc_options=opts)
        uh = problem.solve()
        ksp = problem.solver
    except Exception:
        problem = petsc.LinearProblem(
            a, L, bcs=[bc], petsc_options_prefix=f"poissonlu_{n}_",
            petsc_options={"ksp_type": "preonly", "pc_type": "lu"}
        )
        uh = problem.solve()
        ksp = problem.solver

    uh.x.scatter_forward()
    energy = fem.assemble_scalar(fem.form(ufl.inner(ufl.grad(uh), ufl.grad(uh)) * ufl.dx))
    info = {
        "mesh_resolution": int(n),
        "element_degree": int(degree),
        "ksp_type": str(ksp.getType()),
        "pc_type": str(ksp.getPC().getType()),
        "rtol": float(rtol),
        "iterations": int(ksp.getIterationNumber()),
        "verification_energy": float(np.real(energy)),
    }
    return msh, uh, info


def _sample(msh, uh, grid):
    nx = int(grid["nx"])
    ny = int(grid["ny"])
    xmin, xmax, ymin, ymax = map(float, grid["bbox"])
    xs = np.linspace(xmin, xmax, nx)
    ys = np.linspace(ymin, ymax, ny)
    XX, YY = np.meshgrid(xs, ys, indexing="xy")
    pts = np.column_stack([XX.ravel(), YY.ravel(), np.zeros(nx * ny, dtype=np.float64)])

    tree = geometry.bb_tree(msh, msh.topology.dim)
    cand = geometry.compute_collisions_points(tree, pts)
    coll = geometry.compute_colliding_cells(msh, cand, pts)

    vals = np.full(nx * ny, np.nan, dtype=np.float64)
    pts_local = []
    cells = []
    ids = []
    for i in range(pts.shape[0]):
        links = coll.links(i)
        if len(links) > 0:
            pts_local.append(pts[i])
            cells.append(links[0])
            ids.append(i)
    if pts_local:
        ev = uh.eval(np.array(pts_local, dtype=np.float64), np.array(cells, dtype=np.int32))
        vals[np.array(ids, dtype=np.int32)] = np.asarray(ev).reshape(len(ids), -1)[:, 0]

    vals = msh.comm.allreduce(np.nan_to_num(vals, nan=0.0), op=MPI.SUM)
    return vals.reshape(ny, nx)


def solve(case_spec: dict) -> dict:
    grid = case_spec["output"]["grid"]
    degree = 2
    t0 = time.perf_counter()
    _, _, _ = None, None, None
    probe_n = 32
    msh0, uh0, _ = _build_and_solve(probe_n, degree, 1e-9)
    probe_time = time.perf_counter() - t0

    if probe_time < 0.3:
        n = 96
    elif probe_time < 0.7:
        n = 80
    else:
        n = 64

    msh, uh, info = _build_and_solve(n, degree, 1e-10)

    elapsed = time.perf_counter() - t0
    if elapsed < 1.6 and n >= 64:
        mshc, uhc, _ = _build_and_solve(n // 2, degree, 1e-9)
        uf = _sample(msh, uh, grid)
        uc = _sample(mshc, uhc, grid)
        info["verification_grid_difference_l2_per_point"] = float(np.linalg.norm(uf - uc) / np.sqrt(uf.size))
        u_grid = uf
    else:
        u_grid = _sample(msh, uh, grid)

    return {"u": u_grid, "solver_info": info}
