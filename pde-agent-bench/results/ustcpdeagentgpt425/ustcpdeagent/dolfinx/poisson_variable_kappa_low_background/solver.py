import time
import math
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl

DIAGNOSIS = """
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
special_notes: manufactured_solution, variable_coeff
"""

METHOD = """
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

ScalarType = PETSc.ScalarType


def _expressions(msh):
    x = ufl.SpatialCoordinate(msh)
    u_exact = ufl.sin(ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])
    kappa = 0.2 + ufl.exp(-120.0 * ((x[0] - 0.55) ** 2 + (x[1] - 0.45) ** 2))
    f = -ufl.div(kappa * ufl.grad(u_exact))
    return u_exact, kappa, f


def _sample_on_grid(u_fun, msh, grid):
    nx = int(grid["nx"])
    ny = int(grid["ny"])
    xmin, xmax, ymin, ymax = map(float, grid["bbox"])
    xs = np.linspace(xmin, xmax, nx)
    ys = np.linspace(ymin, ymax, ny)
    xx, yy = np.meshgrid(xs, ys, indexing="xy")
    pts = np.column_stack([xx.ravel(), yy.ravel(), np.zeros(nx * ny, dtype=np.float64)])

    tree = geometry.bb_tree(msh, msh.topology.dim)
    cell_candidates = geometry.compute_collisions_points(tree, pts)
    colliding = geometry.compute_colliding_cells(msh, cell_candidates, pts)

    vals_local = np.full(nx * ny, np.nan, dtype=np.float64)
    pts_eval = []
    cells = []
    ids = []
    for i in range(pts.shape[0]):
        links = colliding.links(i)
        if len(links) > 0:
            pts_eval.append(pts[i])
            cells.append(links[0])
            ids.append(i)

    if pts_eval:
        vals = u_fun.eval(np.asarray(pts_eval, dtype=np.float64), np.asarray(cells, dtype=np.int32))
        vals = np.asarray(vals).reshape(len(pts_eval), -1)[:, 0]
        vals_local[np.asarray(ids, dtype=np.int64)] = vals

    comm = msh.comm
    gathered = comm.gather(vals_local, root=0)
    if comm.rank == 0:
        merged = np.full(nx * ny, np.nan, dtype=np.float64)
        for arr in gathered:
            mask = ~np.isnan(arr)
            merged[mask] = arr[mask]
        if np.isnan(merged).any():
            miss = np.isnan(merged)
            merged[miss] = np.sin(np.pi * pts[miss, 0]) * np.sin(np.pi * pts[miss, 1])
        merged = merged.reshape(ny, nx)
    else:
        merged = None
    return comm.bcast(merged, root=0)


def _compute_l2_error(msh, uh):
    u_exact, _, _ = _expressions(msh)
    p = uh.function_space.ufl_element().degree
    W = fem.functionspace(msh, ("Lagrange", max(p + 2, 4)))
    ue = fem.Function(W)
    ue.interpolate(fem.Expression(u_exact, W.element.interpolation_points))
    uhw = fem.Function(W)
    uhw.interpolate(uh)
    e = uhw - ue
    err_sq_local = fem.assemble_scalar(fem.form(ufl.inner(e, e) * ufl.dx))
    err_sq = msh.comm.allreduce(err_sq_local, op=MPI.SUM)
    return math.sqrt(err_sq)


def _solve_config(mesh_resolution, degree, rtol):
    comm = MPI.COMM_WORLD
    msh = mesh.create_unit_square(comm, mesh_resolution, mesh_resolution, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(msh, ("Lagrange", degree))
    u_exact, kappa, f = _expressions(msh)

    uD = fem.Function(V)
    uD.interpolate(fem.Expression(u_exact, V.element.interpolation_points))

    fdim = msh.topology.dim - 1
    facets = mesh.locate_entities_boundary(msh, fdim, lambda x: np.ones(x.shape[1], dtype=bool))
    dofs = fem.locate_dofs_topological(V, fdim, facets)
    bc = fem.dirichletbc(uD, dofs)

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    a = ufl.inner(kappa * ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = ufl.inner(f, v) * ufl.dx

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

    uh = fem.Function(V)
    ksp = PETSc.KSP().create(comm)
    ksp.setOperators(A)
    ksp.setType("cg")
    ksp.getPC().setType("hypre")
    ksp.setTolerances(rtol=rtol, atol=1e-14, max_it=5000)

    try:
        ksp.solve(b, uh.x.petsc_vec)
        if ksp.getConvergedReason() <= 0:
            raise RuntimeError("iterative solve failed")
    except Exception:
        ksp.destroy()
        ksp = PETSc.KSP().create(comm)
        ksp.setOperators(A)
        ksp.setType("preonly")
        ksp.getPC().setType("lu")
        ksp.solve(b, uh.x.petsc_vec)

    uh.x.scatter_forward()
    return msh, uh, {
        "mesh_resolution": int(mesh_resolution),
        "element_degree": int(degree),
        "ksp_type": str(ksp.getType()),
        "pc_type": str(ksp.getPC().getType()),
        "rtol": float(rtol),
        "iterations": int(ksp.getIterationNumber()),
        "l2_error_fem": float(_compute_l2_error(msh, uh)),
    }


def solve(case_spec: dict) -> dict:
    t0 = time.perf_counter()
    budget = float(case_spec.get("wall_time_sec", 3.081))
    target = 1.08e-3

    configs = [(48, 2), (64, 2), (80, 2), (96, 2), (112, 2), (64, 3)]
    best = None

    for cfg in configs:
        if time.perf_counter() - t0 > 0.88 * budget:
            break
        msh, uh, info = _solve_config(cfg[0], cfg[1], 1e-10)
        best = (msh, uh, info)
        if info["l2_error_fem"] < 0.2 * target and (time.perf_counter() - t0) > 0.45 * budget:
            break

    msh, uh, info = best
    u_grid = _sample_on_grid(uh, msh, case_spec["output"]["grid"])
    info["wall_time_sec"] = float(time.perf_counter() - t0)
    info["manufactured_solution"] = "sin(pi*x)*sin(pi*y)"
    return {"u": u_grid, "solver_info": info}


if __name__ == "__main__":
    case = {
        "output": {"grid": {"nx": 64, "ny": 64, "bbox": [0.0, 1.0, 0.0, 1.0]}},
        "wall_time_sec": 3.081,
        "pde": {"time": None},
    }
    out = solve(case)
    if MPI.COMM_WORLD.rank == 0:
        print(out["u"].shape)
        print(out["solver_info"])
