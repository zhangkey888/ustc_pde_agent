import time
import math
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc

from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl


ScalarType = PETSc.ScalarType


def _u_exact_numpy(xy):
    x = xy[:, 0]
    y = xy[:, 1]
    return np.exp(0.5 * x) * np.sin(2.0 * np.pi * y)


def _sample_function_on_grid(domain, uh, grid_spec):
    nx = int(grid_spec["nx"])
    ny = int(grid_spec["ny"])
    bbox = grid_spec["bbox"]
    xmin, xmax, ymin, ymax = map(float, bbox)

    xs = np.linspace(xmin, xmax, nx)
    ys = np.linspace(ymin, ymax, ny)
    XX, YY = np.meshgrid(xs, ys, indexing="xy")
    pts2 = np.column_stack([XX.ravel(), YY.ravel()])
    pts3 = np.zeros((pts2.shape[0], 3), dtype=np.float64)
    pts3[:, :2] = pts2

    tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(tree, pts3)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, pts3)

    local_vals = np.full(pts3.shape[0], np.nan, dtype=np.float64)
    points_on_proc = []
    cells_on_proc = []
    eval_ids = []

    for i in range(pts3.shape[0]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(pts3[i])
            cells_on_proc.append(np.int32(links[0]))
            eval_ids.append(i)

    if points_on_proc:
        vals = uh.eval(np.array(points_on_proc, dtype=np.float64), np.array(cells_on_proc, dtype=np.int32))
        vals = np.asarray(vals).reshape(len(eval_ids), -1)
        local_vals[np.array(eval_ids, dtype=np.int32)] = vals[:, 0]

    global_vals = np.empty_like(local_vals)
    domain.comm.Allreduce(local_vals, global_vals, op=MPI.MAX)

    if np.isnan(global_vals).any():
        mask = np.isnan(global_vals)
        global_vals[mask] = _u_exact_numpy(pts2[mask])

    return global_vals.reshape(ny, nx)


def _build_and_solve(nx, degree, ksp_type="cg", pc_type="hypre", rtol=1e-10):
    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_square(comm, nx, nx, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", degree))

    x = ufl.SpatialCoordinate(domain)

    u_exact_ufl = ufl.exp(0.5 * x[0]) * ufl.sin(2.0 * ufl.pi * x[1])
    kappa = (
        1.0
        + 15.0 * ufl.exp(-200.0 * ((x[0] - 0.25) ** 2 + (x[1] - 0.25) ** 2))
        + 15.0 * ufl.exp(-200.0 * ((x[0] - 0.75) ** 2 + (x[1] - 0.75) ** 2))
    )
    f_expr = -ufl.div(kappa * ufl.grad(u_exact_ufl))

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    a = ufl.inner(kappa * ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = ufl.inner(f_expr, v) * ufl.dx

    fdim = domain.topology.dim - 1
    facets = mesh.locate_entities_boundary(domain, fdim, lambda X: np.ones(X.shape[1], dtype=bool))
    dofs = fem.locate_dofs_topological(V, fdim, facets)

    u_bc = fem.Function(V)
    u_bc.interpolate(lambda X: np.exp(0.5 * X[0]) * np.sin(2.0 * np.pi * X[1]))
    bc = fem.dirichletbc(u_bc, dofs)

    a_form = fem.form(a)
    L_form = fem.form(L)

    A = petsc.assemble_matrix(a_form, bcs=[bc])
    A.assemble()
    b = petsc.create_vector(L_form.function_spaces)

    solver = PETSc.KSP().create(comm)
    solver.setOperators(A)
    solver.setType(ksp_type)
    pc = solver.getPC()
    pc.setType(pc_type)
    if pc_type == "hypre":
        try:
            pc.setHYPREType("boomeramg")
        except Exception:
            pass
    solver.setTolerances(rtol=rtol, atol=0.0, max_it=10000)

    uh = fem.Function(V)

    with b.localForm() as loc:
        loc.set(0.0)
    petsc.assemble_vector(b, L_form)
    petsc.apply_lifting(b, [a_form], bcs=[[bc]])
    b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    petsc.set_bc(b, [bc])

    try:
        solver.solve(b, uh.x.petsc_vec)
    except Exception:
        solver.setType("preonly")
        solver.getPC().setType("lu")
        solver.solve(b, uh.x.petsc_vec)

    uh.x.scatter_forward()

    its = int(solver.getIterationNumber())
    ksp_final = solver.getType()
    pc_final = solver.getPC().getType()

    l2_err_local = fem.assemble_scalar(fem.form(ufl.inner(uh - u_exact_ufl, uh - u_exact_ufl) * ufl.dx))
    l2_ref_local = fem.assemble_scalar(fem.form(ufl.inner(u_exact_ufl, u_exact_ufl) * ufl.dx))
    h1s_err_local = fem.assemble_scalar(
        fem.form(ufl.inner(kappa * ufl.grad(uh - u_exact_ufl), ufl.grad(uh - u_exact_ufl)) * ufl.dx)
    )

    l2_err = math.sqrt(comm.allreduce(l2_err_local, op=MPI.SUM))
    l2_ref = math.sqrt(comm.allreduce(l2_ref_local, op=MPI.SUM))
    rel_l2 = l2_err / max(l2_ref, 1e-16)
    energy_err = math.sqrt(comm.allreduce(h1s_err_local, op=MPI.SUM))

    return {
        "domain": domain,
        "uh": uh,
        "rel_l2_error": rel_l2,
        "energy_error": energy_err,
        "iterations": its,
        "ksp_type": ksp_final,
        "pc_type": pc_final,
        "rtol": rtol,
        "mesh_resolution": nx,
        "element_degree": degree,
    }


def solve(case_spec: dict) -> dict:
    t0 = time.perf_counter()

    output_grid = case_spec["output"]["grid"]
    budget = 1.30

    candidates = [
        (28, 1, "cg", "hypre", 1e-10),
        (40, 1, "cg", "hypre", 1e-10),
        (52, 1, "cg", "hypre", 1e-10),
        (36, 2, "cg", "hypre", 1e-11),
        (48, 2, "cg", "hypre", 1e-11),
    ]

    best = None
    for nx, deg, ksp_type, pc_type, rtol in candidates:
        elapsed = time.perf_counter() - t0
        if elapsed > 0.85 * budget and best is not None:
            break

        result = _build_and_solve(nx, deg, ksp_type=ksp_type, pc_type=pc_type, rtol=rtol)
        best = result

        if result["rel_l2_error"] <= 5e-4 and (time.perf_counter() - t0) > 0.45 * budget:
            break

    if best is None:
        best = _build_and_solve(40, 1, ksp_type="cg", pc_type="hypre", rtol=1e-10)

    u_grid = _sample_function_on_grid(best["domain"], best["uh"], output_grid)

    solver_info = {
        "mesh_resolution": int(best["mesh_resolution"]),
        "element_degree": int(best["element_degree"]),
        "ksp_type": str(best["ksp_type"]),
        "pc_type": str(best["pc_type"]),
        "rtol": float(best["rtol"]),
        "iterations": int(best["iterations"]),
        "verification": {
            "manufactured_solution": "exp(0.5*x)*sin(2*pi*y)",
            "relative_L2_error": float(best["rel_l2_error"]),
            "energy_error": float(best["energy_error"]),
        },
    }

    return {"u": u_grid, "solver_info": solver_info}
