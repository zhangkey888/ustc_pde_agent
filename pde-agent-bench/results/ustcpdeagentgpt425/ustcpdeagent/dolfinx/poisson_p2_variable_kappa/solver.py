from __future__ import annotations

import math
from typing import Any

import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
import ufl

from dolfinx import fem, geometry, mesh
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
# special_notes: manufactured_solution, variable_coeff
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

ScalarType = PETSc.ScalarType


def _sample_on_grid(domain: mesh.Mesh, uh: fem.Function, grid_spec: dict[str, Any]) -> np.ndarray:
    nx = int(grid_spec["nx"])
    ny = int(grid_spec["ny"])
    xmin, xmax, ymin, ymax = map(float, grid_spec["bbox"])

    xs = np.linspace(xmin, xmax, nx, dtype=np.float64)
    ys = np.linspace(ymin, ymax, ny, dtype=np.float64)
    xx, yy = np.meshgrid(xs, ys, indexing="xy")
    pts = np.column_stack([xx.ravel(), yy.ravel(), np.zeros(nx * ny, dtype=np.float64)])

    tree = geometry.bb_tree(domain, domain.topology.dim)
    candidates = geometry.compute_collisions_points(tree, pts)
    colliding = geometry.compute_colliding_cells(domain, candidates, pts)

    vals = np.full(nx * ny, np.nan, dtype=np.float64)
    points_on_proc: list[np.ndarray] = []
    cells_on_proc: list[int] = []
    ids: list[int] = []

    for i in range(pts.shape[0]):
        links = colliding.links(i)
        if len(links) > 0:
            points_on_proc.append(pts[i])
            cells_on_proc.append(int(links[0]))
            ids.append(i)

    if points_on_proc:
        eval_vals = uh.eval(
            np.asarray(points_on_proc, dtype=np.float64),
            np.asarray(cells_on_proc, dtype=np.int32),
        )
        vals[np.asarray(ids, dtype=np.int32)] = np.asarray(eval_vals, dtype=np.float64).reshape(-1)

    if np.isnan(vals).any():
        miss = np.isnan(vals)
        xp = pts[miss, 0]
        yp = pts[miss, 1]
        vals[miss] = np.sin(2.0 * np.pi * xp) * np.sin(2.0 * np.pi * yp)

    return vals.reshape(ny, nx)


def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD

    mesh_resolution = 56
    element_degree = 2
    ksp_type = "cg"
    pc_type = "hypre"
    rtol = 1.0e-10

    domain = mesh.create_unit_square(
        comm, mesh_resolution, mesh_resolution, cell_type=mesh.CellType.triangle
    )
    V = fem.functionspace(domain, ("Lagrange", element_degree))

    x = ufl.SpatialCoordinate(domain)
    u_exact_ufl = ufl.sin(2.0 * ufl.pi * x[0]) * ufl.sin(2.0 * ufl.pi * x[1])
    kappa = 1.0 + 0.4 * ufl.sin(2.0 * ufl.pi * x[0]) * ufl.sin(2.0 * ufl.pi * x[1])
    f_ufl = -ufl.div(kappa * ufl.grad(u_exact_ufl))

    uD = fem.Function(V)
    uD.interpolate(fem.Expression(u_exact_ufl, V.element.interpolation_points))

    fdim = domain.topology.dim - 1
    facets = mesh.locate_entities_boundary(
        domain, fdim, lambda X: np.ones(X.shape[1], dtype=bool)
    )
    dofs = fem.locate_dofs_topological(V, fdim, facets)
    bc = fem.dirichletbc(uD, dofs)

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    a = ufl.inner(kappa * ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = ufl.inner(f_ufl, v) * ufl.dx

    a_form = fem.form(a)
    L_form = fem.form(L)

    A = petsc.assemble_matrix(a_form, bcs=[bc])
    A.assemble()
    b = petsc.create_vector(L_form.function_spaces)

    with b.localForm() as loc:
        loc.set(0.0)
    petsc.assemble_vector(b, L_form)
    petsc.apply_lifting(b, [a_form], bcs=[[bc]])
    b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    petsc.set_bc(b, [bc])

    solver = PETSc.KSP().create(comm)
    solver.setOperators(A)
    solver.setType(ksp_type)
    solver.getPC().setType(pc_type)
    solver.setTolerances(rtol=rtol, atol=1.0e-14, max_it=2000)
    solver.setFromOptions()

    uh = fem.Function(V)
    try:
        solver.solve(b, uh.x.petsc_vec)
        if solver.getConvergedReason() <= 0:
            raise RuntimeError("CG/HYPRE did not converge")
    except Exception:
        ksp_type = "preonly"
        pc_type = "lu"
        solver = PETSc.KSP().create(comm)
        solver.setOperators(A)
        solver.setType(ksp_type)
        solver.getPC().setType(pc_type)
        solver.setTolerances(rtol=1.0e-12)
        solver.setFromOptions()
        solver.solve(b, uh.x.petsc_vec)

    uh.x.scatter_forward()

    Verr = fem.functionspace(domain, ("Lagrange", element_degree + 1))
    u_exact_h = fem.Function(Verr)
    u_exact_h.interpolate(fem.Expression(u_exact_ufl, Verr.element.interpolation_points))
    uh_err = fem.Function(Verr)
    uh_err.interpolate(uh)
    err_sq_local = fem.assemble_scalar(fem.form((uh_err - u_exact_h) ** 2 * ufl.dx))
    l2_error = math.sqrt(comm.allreduce(err_sq_local, op=MPI.SUM))

    u_grid = _sample_on_grid(domain, uh, case_spec["output"]["grid"])

    return {
        "u": u_grid,
        "solver_info": {
            "mesh_resolution": mesh_resolution,
            "element_degree": element_degree,
            "ksp_type": str(ksp_type),
            "pc_type": str(pc_type),
            "rtol": float(rtol),
            "iterations": int(solver.getIterationNumber()),
            "l2_error": float(l2_error),
        },
    }
