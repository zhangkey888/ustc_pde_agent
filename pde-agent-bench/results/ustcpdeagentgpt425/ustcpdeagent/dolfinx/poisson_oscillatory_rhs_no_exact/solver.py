from __future__ import annotations

import math
import time
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
import ufl
from dolfinx import fem, geometry, mesh
from dolfinx.fem import petsc

# DIAGNOSIS
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

# METHOD
# spatial_method: fem
# element_or_basis: Lagrange_P2
# stabilization: none
# time_method: none
# nonlinear_solver: none
# linear_solver: cg
# preconditioner: hypre
# special_treatment: none
# pde_skill: poisson

ScalarType = PETSc.ScalarType


def _exact_xy(xy: np.ndarray) -> np.ndarray:
    x = xy[:, 0]
    y = xy[:, 1]
    return (1.0 / (128.0 * math.pi * math.pi)) * np.sin(8.0 * math.pi * x) * np.sin(8.0 * math.pi * y)


def _sample_function(domain, uh: fem.Function, pts: np.ndarray) -> np.ndarray:
    tree = geometry.bb_tree(domain, domain.topology.dim)
    candidates = geometry.compute_collisions_points(tree, pts)
    colliding = geometry.compute_colliding_cells(domain, candidates, pts)

    local_ids = []
    local_pts = []
    local_cells = []
    for i in range(pts.shape[0]):
        links = colliding.links(i)
        if len(links) > 0:
            local_ids.append(i)
            local_pts.append(pts[i])
            local_cells.append(links[0])

    local_values = np.full(pts.shape[0], np.nan, dtype=np.float64)
    if local_pts:
        vals = uh.eval(np.array(local_pts, dtype=np.float64), np.array(local_cells, dtype=np.int32))
        vals = np.asarray(vals).reshape(len(local_pts), -1)[:, 0]
        local_values[np.array(local_ids, dtype=np.int32)] = vals

    gathered = domain.comm.allgather(local_values)
    values = np.full(pts.shape[0], np.nan, dtype=np.float64)
    for arr in gathered:
        mask = np.isnan(values) & ~np.isnan(arr)
        values[mask] = arr[mask]
    values[np.isnan(values)] = 0.0
    return values


def _solve_once(mesh_resolution: int = 96, degree: int = 2, rtol: float = 1e-10) -> dict:
    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_square(comm, mesh_resolution, mesh_resolution, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", degree))

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    x = ufl.SpatialCoordinate(domain)
    f = ufl.sin(8.0 * ufl.pi * x[0]) * ufl.sin(8.0 * ufl.pi * x[1])

    a = fem.form(ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx)
    L = fem.form(ufl.inner(f, v) * ufl.dx)

    fdim = domain.topology.dim - 1
    facets = mesh.locate_entities_boundary(domain, fdim, lambda X: np.ones(X.shape[1], dtype=bool))
    dofs = fem.locate_dofs_topological(V, fdim, facets)
    bc = fem.dirichletbc(ScalarType(0.0), dofs, V)

    A = petsc.assemble_matrix(a, bcs=[bc])
    A.assemble()
    b = petsc.create_vector(L.function_spaces)
    with b.localForm() as loc:
        loc.set(0.0)
    petsc.assemble_vector(b, L)
    petsc.apply_lifting(b, [a], bcs=[[bc]])
    b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    petsc.set_bc(b, [bc])

    uh = fem.Function(V)
    solver = PETSc.KSP().create(comm)
    solver.setOperators(A)
    solver.setType("cg")
    solver.getPC().setType("hypre")
    solver.setTolerances(rtol=rtol)

    ksp_type = "cg"
    pc_type = "hypre"
    try:
        solver.solve(b, uh.x.petsc_vec)
        if solver.getConvergedReason() <= 0:
            raise RuntimeError("iterative solver did not converge")
    except Exception:
        solver.destroy()
        solver = PETSc.KSP().create(comm)
        solver.setOperators(A)
        solver.setType("preonly")
        solver.getPC().setType("lu")
        solver.setTolerances(rtol=min(rtol, 1e-12))
        solver.solve(b, uh.x.petsc_vec)
        ksp_type = "preonly"
        pc_type = "lu"

    uh.x.scatter_forward()
    return {
        "domain": domain,
        "u": uh,
        "mesh_resolution": mesh_resolution,
        "element_degree": degree,
        "ksp_type": ksp_type,
        "pc_type": pc_type,
        "rtol": rtol,
        "iterations": int(solver.getIterationNumber()),
    }


def solve(case_spec: dict) -> dict:
    result = _solve_once(mesh_resolution=224, degree=2, rtol=1e-10)

    grid = case_spec["output"]["grid"]
    nx = int(grid["nx"])
    ny = int(grid["ny"])
    xmin, xmax, ymin, ymax = map(float, grid["bbox"])
    xs = np.linspace(xmin, xmax, nx)
    ys = np.linspace(ymin, ymax, ny)
    xx, yy = np.meshgrid(xs, ys, indexing="xy")
    pts = np.column_stack([xx.ravel(), yy.ravel(), np.zeros(nx * ny, dtype=np.float64)])

    u_grid = _sample_function(result["domain"], result["u"], pts).reshape(ny, nx)

    return {
        "u": u_grid,
        "solver_info": {
            "mesh_resolution": int(result["mesh_resolution"]),
            "element_degree": int(result["element_degree"]),
            "ksp_type": str(result["ksp_type"]),
            "pc_type": str(result["pc_type"]),
            "rtol": float(result["rtol"]),
            "iterations": int(result["iterations"]),
        },
    }


if __name__ == "__main__":
    t0 = time.perf_counter()
    case_spec = {
        "pde": {"time": None},
        "output": {"grid": {"nx": 129, "ny": 129, "bbox": [0.0, 1.0, 0.0, 1.0]}},
    }
    out = solve(case_spec)
    wall = time.perf_counter() - t0

    if MPI.COMM_WORLD.rank == 0:
        grid = case_spec["output"]["grid"]
        nx = int(grid["nx"])
        ny = int(grid["ny"])
        xmin, xmax, ymin, ymax = map(float, grid["bbox"])
        xs = np.linspace(xmin, xmax, nx)
        ys = np.linspace(ymin, ymax, ny)
        xx, yy = np.meshgrid(xs, ys, indexing="xy")
        exact = _exact_xy(np.column_stack([xx.ravel(), yy.ravel()])).reshape(ny, nx)
        rel_err = np.linalg.norm(out["u"] - exact) / np.linalg.norm(exact)
        print(f"L2_ERROR: {rel_err:.12e}")
        print(f"WALL_TIME: {wall:.12e}")
        print(out["solver_info"])
