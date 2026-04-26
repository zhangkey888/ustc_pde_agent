import math
import time
from typing import Dict, Any

import numpy as np
from mpi4py import MPI
from petsc4py import PETSc

from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl


ScalarType = PETSc.ScalarType


def _u_exact_np(xy):
    x = xy[0]
    y = xy[1]
    return np.sin(2.0 * np.pi * (x + y)) * np.sin(np.pi * (x - y))


def _make_exact_and_rhs(domain, eps_value, beta_value):
    x = ufl.SpatialCoordinate(domain)
    eps = ScalarType(eps_value)
    beta = ufl.as_vector((ScalarType(beta_value[0]), ScalarType(beta_value[1])))

    u_exact = ufl.sin(2.0 * ufl.pi * (x[0] + x[1])) * ufl.sin(ufl.pi * (x[0] - x[1]))
    f = -eps * ufl.div(ufl.grad(u_exact)) + ufl.dot(beta, ufl.grad(u_exact))
    return u_exact, f, beta


def _sample_on_grid(domain, uh, grid_spec):
    nx = int(grid_spec["nx"])
    ny = int(grid_spec["ny"])
    bbox = grid_spec["bbox"]
    xmin, xmax, ymin, ymax = map(float, bbox)

    xs = np.linspace(xmin, xmax, nx)
    ys = np.linspace(ymin, ymax, ny)
    XX, YY = np.meshgrid(xs, ys, indexing="xy")
    pts2 = np.column_stack([XX.ravel(), YY.ravel(), np.zeros(nx * ny, dtype=np.float64)])

    tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(tree, pts2)
    colliding = geometry.compute_colliding_cells(domain, cell_candidates, pts2)

    local_vals = np.full(nx * ny, np.nan, dtype=np.float64)
    points_on_proc = []
    cells = []
    ids = []
    for i in range(nx * ny):
        links = colliding.links(i)
        if len(links) > 0:
            points_on_proc.append(pts2[i])
            cells.append(links[0])
            ids.append(i)

    if ids:
        vals = uh.eval(np.array(points_on_proc, dtype=np.float64), np.array(cells, dtype=np.int32))
        local_vals[np.array(ids, dtype=np.int32)] = np.asarray(vals, dtype=np.float64).reshape(-1)

    comm = domain.comm
    gathered = comm.gather(local_vals, root=0)

    if comm.rank == 0:
        final = np.full(nx * ny, np.nan, dtype=np.float64)
        for arr in gathered:
            mask = ~np.isnan(arr)
            final[mask] = arr[mask]
        if np.isnan(final).any():
            raise RuntimeError("Some output grid points could not be evaluated.")
        return final.reshape((ny, nx))
    return None


def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD
    t0 = time.perf_counter()

    eps_value = 0.05
    beta_value = np.array([3.0, 1.0], dtype=np.float64)

    target_time = 1.711
    degree = 1

    # Adaptive accuracy under the time budget:
    # P1 + SUPG is robust for this convection-diffusion case and efficient.
    # Use a moderately fine mesh aimed to fit the benchmark budget.
    n = 96
    if case_spec.get("performance_hint") == "fast":
        n = 80

    domain = mesh.create_unit_square(comm, n, n, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", degree))

    u_exact_ufl, f_ufl, beta_ufl = _make_exact_and_rhs(domain, eps_value, beta_value)

    u_bc = fem.Function(V)
    u_bc.interpolate(lambda x: _u_exact_np(x))
    fdim = domain.topology.dim - 1
    facets = mesh.locate_entities_boundary(domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool))
    bdofs = fem.locate_dofs_topological(V, fdim, facets)
    bc = fem.dirichletbc(u_bc, bdofs)

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    h = ufl.CellDiameter(domain)
    beta_norm = math.sqrt(beta_value[0] ** 2 + beta_value[1] ** 2)
    tau = h / (2.0 * beta_norm)

    residual_u = -eps_value * ufl.div(ufl.grad(u)) + ufl.dot(beta_ufl, ufl.grad(u))
    residual_rhs = f_ufl

    a = (
        eps_value * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
        + ufl.dot(beta_ufl, ufl.grad(u)) * v * ufl.dx
        + tau * ufl.dot(beta_ufl, ufl.grad(v)) * residual_u * ufl.dx
    )
    L = (
        f_ufl * v * ufl.dx
        + tau * ufl.dot(beta_ufl, ufl.grad(v)) * residual_rhs * ufl.dx
    )

    a_form = fem.form(a)
    L_form = fem.form(L)

    A = petsc.assemble_matrix(a_form, bcs=[bc])
    A.assemble()

    b = petsc.create_vector(L_form.function_spaces)

    ksp = PETSc.KSP().create(comm)
    ksp.setOperators(A)
    ksp.setType("gmres")
    pc = ksp.getPC()
    pc.setType("ilu")
    ksp.setTolerances(rtol=1e-9, atol=1e-12, max_it=1000)
    ksp.setFromOptions()

    uh = fem.Function(V)

    with b.localForm() as loc:
        loc.set(0)
    petsc.assemble_vector(b, L_form)
    petsc.apply_lifting(b, [a_form], bcs=[[bc]])
    b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    petsc.set_bc(b, [bc])

    try:
        ksp.solve(b, uh.x.petsc_vec)
    except Exception:
        ksp.setType("preonly")
        pc.setType("lu")
        ksp.solve(b, uh.x.petsc_vec)
    uh.x.scatter_forward()

    its = int(ksp.getIterationNumber())
    ksp_type = ksp.getType()
    pc_type = ksp.getPC().getType()

    # Accuracy verification
    u_ex = fem.Function(V)
    u_ex.interpolate(lambda x: _u_exact_np(x))
    err_fun = fem.Function(V)
    err_fun.x.array[:] = uh.x.array - u_ex.x.array
    err_fun.x.scatter_forward()
    l2_err = math.sqrt(comm.allreduce(fem.assemble_scalar(fem.form(err_fun * err_fun * ufl.dx)), op=MPI.SUM))
    l2_ref = math.sqrt(comm.allreduce(fem.assemble_scalar(fem.form(u_ex * u_ex * ufl.dx)), op=MPI.SUM))
    rel_l2_err = l2_err / l2_ref if l2_ref > 0 else l2_err

    grid = case_spec["output"]["grid"]
    u_grid = _sample_on_grid(domain, uh, grid)

    elapsed = time.perf_counter() - t0

    result = None
    if comm.rank == 0:
        result = {
            "u": u_grid,
            "solver_info": {
                "mesh_resolution": n,
                "element_degree": degree,
                "ksp_type": str(ksp_type),
                "pc_type": str(pc_type),
                "rtol": 1e-9,
                "iterations": its,
                "l2_error": float(l2_err),
                "relative_l2_error": float(rel_l2_err),
                "wall_time_sec": float(elapsed),
                "peclet_estimate": 63.2,
                "stabilization": "SUPG",
                "time_budget_target_sec": target_time,
            },
        }
    return result
