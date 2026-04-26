import time
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl

ScalarType = PETSc.ScalarType


def _sample_function_on_grid(u_func, msh, grid_spec):
    nx = int(grid_spec["nx"])
    ny = int(grid_spec["ny"])
    xmin, xmax, ymin, ymax = grid_spec["bbox"]

    xs = np.linspace(xmin, xmax, nx, dtype=np.float64)
    ys = np.linspace(ymin, ymax, ny, dtype=np.float64)
    xx, yy = np.meshgrid(xs, ys, indexing="xy")
    pts = np.column_stack([xx.ravel(), yy.ravel(), np.zeros(nx * ny, dtype=np.float64)])

    tree = geometry.bb_tree(msh, msh.topology.dim)
    cell_candidates = geometry.compute_collisions_points(tree, pts)
    colliding_cells = geometry.compute_colliding_cells(msh, cell_candidates, pts)

    local_ids = []
    local_points = []
    local_cells = []
    for i in range(pts.shape[0]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            local_ids.append(i)
            local_points.append(pts[i])
            local_cells.append(links[0])

    values = np.full(pts.shape[0], np.nan, dtype=np.float64)
    if local_points:
        vals = u_func.eval(
            np.asarray(local_points, dtype=np.float64),
            np.asarray(local_cells, dtype=np.int32),
        )
        values[np.asarray(local_ids, dtype=np.int32)] = np.asarray(vals, dtype=np.float64).reshape(-1)

    comm = msh.comm
    gathered = comm.gather(values, root=0)

    if comm.rank == 0:
        out = np.full_like(values, np.nan)
        for arr in gathered:
            mask = np.isnan(out) & ~np.isnan(arr)
            out[mask] = arr[mask]
        if np.isnan(out).any():
            raise RuntimeError("Failed to evaluate solution at some output grid points.")
        return out.reshape(ny, nx)
    return None


def _build_and_solve(n, degree, ksp_type="cg", pc_type="hypre", rtol=1e-10):
    comm = MPI.COMM_WORLD
    msh = mesh.create_unit_square(comm, n, n, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(msh, ("Lagrange", degree))

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    x = ufl.SpatialCoordinate(msh)

    kappa = fem.Constant(msh, ScalarType(1.0))
    f = fem.Constant(msh, ScalarType(1.0))

    a = ufl.inner(kappa * ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = ufl.inner(f, v) * ufl.dx

    g_fun = fem.Function(V)
    g_expr = ufl.sin(ufl.pi * x[0]) + ufl.cos(ufl.pi * x[1])
    g_fun.interpolate(fem.Expression(g_expr, V.element.interpolation_points))

    fdim = msh.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(
        msh, fdim, lambda X: np.ones(X.shape[1], dtype=bool)
    )
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc = fem.dirichletbc(g_fun, boundary_dofs)

    a_form = fem.form(a)
    L_form = fem.form(L)

    A = petsc.assemble_matrix(a_form, bcs=[bc])
    A.assemble()
    b = petsc.create_vector(L_form.function_spaces)

    uh = fem.Function(V)

    solver = PETSc.KSP().create(comm)
    solver.setOperators(A)
    solver.setType(ksp_type)
    solver.getPC().setType(pc_type)
    solver.setTolerances(rtol=rtol)
    solver.setFromOptions()

    with b.localForm() as loc:
        loc.set(0)
    petsc.assemble_vector(b, L_form)
    petsc.apply_lifting(b, [a_form], bcs=[[bc]])
    b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    petsc.set_bc(b, [bc])

    solver.solve(b, uh.x.petsc_vec)
    if solver.getConvergedReason() <= 0:
        solver.destroy()
        solver = PETSc.KSP().create(comm)
        solver.setOperators(A)
        solver.setType("preonly")
        solver.getPC().setType("lu")
        solver.setTolerances(rtol=rtol)
        solver.setFromOptions()
        solver.solve(b, uh.x.petsc_vec)

    uh.x.scatter_forward()

    w = ufl.TestFunction(V)
    residual_form = fem.form(
        (ufl.inner(kappa * ufl.grad(uh), ufl.grad(w)) - ufl.inner(f, w)) * ufl.dx
    )
    r = petsc.create_vector(residual_form.function_spaces)
    with r.localForm() as loc:
        loc.set(0)
    petsc.assemble_vector(r, residual_form)
    petsc.apply_lifting(r, [a_form], bcs=[[bc]])
    r.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    petsc.set_bc(r, [bc])
    residual_norm = float(r.norm())

    return {
        "mesh": msh,
        "u": uh,
        "mesh_resolution": int(n),
        "element_degree": int(degree),
        "ksp_type": str(solver.getType()),
        "pc_type": str(solver.getPC().getType()),
        "rtol": float(rtol),
        "iterations": int(solver.getIterationNumber()),
        "residual_norm": residual_norm,
    }


def solve(case_spec: dict) -> dict:
    degree = 2
    candidate_meshes = [48, 64, 80, 96, 112]
    time_budget = 2.676
    t0 = time.perf_counter()

    result = None
    for n in candidate_meshes:
        trial = _build_and_solve(n, degree, ksp_type="cg", pc_type="hypre", rtol=1e-10)
        result = trial
        if time.perf_counter() - t0 > 0.8 * time_budget:
            break

    verification = {}
    if result["mesh_resolution"] < candidate_meshes[-1] and (time.perf_counter() - t0) < 0.6 * time_budget:
        idx = candidate_meshes.index(result["mesh_resolution"])
        refined = _build_and_solve(candidate_meshes[idx + 1], degree, ksp_type="cg", pc_type="hypre", rtol=1e-10)
        grid = case_spec["output"]["grid"]
        u0 = _sample_function_on_grid(result["u"], result["mesh"], grid)
        u1 = _sample_function_on_grid(refined["u"], refined["mesh"], grid)
        if MPI.COMM_WORLD.rank == 0:
            verification["grid_refinement_linf"] = float(np.max(np.abs(u1 - u0)))
            verification["grid_refinement_l2"] = float(np.sqrt(np.mean((u1 - u0) ** 2)))
        result = refined

    u_grid = _sample_function_on_grid(result["u"], result["mesh"], case_spec["output"]["grid"])

    if MPI.COMM_WORLD.rank == 0:
        return {
            "u": u_grid,
            "solver_info": {
                "mesh_resolution": result["mesh_resolution"],
                "element_degree": result["element_degree"],
                "ksp_type": result["ksp_type"],
                "pc_type": result["pc_type"],
                "rtol": result["rtol"],
                "iterations": result["iterations"],
                "residual_norm": result["residual_norm"],
                **verification,
            },
        }
    return {"u": None, "solver_info": {}}
