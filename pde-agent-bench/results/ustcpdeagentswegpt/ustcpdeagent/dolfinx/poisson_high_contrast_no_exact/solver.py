import time
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
import ufl
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc

ScalarType = PETSc.ScalarType


def _build_mesh_and_solve(n: int, degree: int = 1):
    comm = MPI.COMM_WORLD
    msh = mesh.create_unit_square(comm, n, n, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(msh, ("Lagrange", degree))

    fdim = msh.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(
        msh, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
    )
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc = fem.dirichletbc(ScalarType(0.0), boundary_dofs, V)

    x = ufl.SpatialCoordinate(msh)
    kappa = 1.0 + 1000.0 * ufl.exp(-100.0 * (x[0] - 0.5) ** 2)
    f = fem.Constant(msh, ScalarType(1.0))

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    a = ufl.inner(kappa * ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = ufl.inner(f, v) * ufl.dx

    problem = petsc.LinearProblem(
        a,
        L,
        bcs=[bc],
        petsc_options_prefix=f"poisson_{n}_{degree}_",
        petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
    )
    uh = problem.solve()
    uh.x.scatter_forward()

    ksp_type = "preonly"
    pc_type = "lu"
    iterations = 1
    try:
        solver = problem.solver
        ksp_type = solver.getType()
        pc_type = solver.getPC().getType()
        iterations = int(solver.getIterationNumber())
    except Exception:
        pass

    residual_norm = np.nan
    try:
        vr = ufl.TestFunction(V)
        res_form = fem.form((ufl.inner(kappa * ufl.grad(uh), ufl.grad(vr)) - ufl.inner(f, vr)) * ufl.dx)
        res_vec = petsc.create_vector(res_form.function_spaces)
        with res_vec.localForm() as loc:
            loc.set(0.0)
        petsc.assemble_vector(res_vec, res_form)
        petsc.apply_lifting(res_vec, [fem.form(a)], bcs=[[bc]])
        res_vec.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        petsc.set_bc(res_vec, [bc])
        residual_norm = float(res_vec.norm())
    except Exception:
        pass

    return msh, uh, {
        "mesh_resolution": int(n),
        "element_degree": int(degree),
        "ksp_type": str(ksp_type),
        "pc_type": str(pc_type),
        "rtol": 1e-8,
        "iterations": int(iterations),
        "verification_residual_l2": residual_norm,
    }


def _sample_on_grid(msh, uh, grid_spec):
    nx = int(grid_spec["nx"])
    ny = int(grid_spec["ny"])
    xmin, xmax, ymin, ymax = map(float, grid_spec["bbox"])

    xs = np.linspace(xmin, xmax, nx)
    ys = np.linspace(ymin, ymax, ny)
    XX, YY = np.meshgrid(xs, ys, indexing="xy")
    pts = np.column_stack([XX.ravel(), YY.ravel(), np.zeros(nx * ny, dtype=np.float64)])

    tree = geometry.bb_tree(msh, msh.topology.dim)
    cell_candidates = geometry.compute_collisions_points(tree, pts)
    colliding_cells = geometry.compute_colliding_cells(msh, cell_candidates, pts)

    local_vals = np.full(nx * ny, np.nan, dtype=np.float64)
    points_on_proc = []
    cells_on_proc = []
    ids = []

    for i in range(pts.shape[0]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(pts[i])
            cells_on_proc.append(links[0])
            ids.append(i)

    if points_on_proc:
        vals = uh.eval(
            np.array(points_on_proc, dtype=np.float64),
            np.array(cells_on_proc, dtype=np.int32),
        )
        vals = np.asarray(vals).reshape(len(points_on_proc), -1)[:, 0]
        local_vals[np.array(ids, dtype=np.int32)] = vals

    comm = msh.comm
    gathered = comm.gather(local_vals, root=0)
    if comm.rank == 0:
        merged = np.full_like(local_vals, np.nan)
        for arr in gathered:
            mask = ~np.isnan(arr)
            merged[mask] = arr[mask]
        merged[np.isnan(merged)] = 0.0
        out = merged.reshape(ny, nx)
    else:
        out = None
    return comm.bcast(out, root=0)


def solve(case_spec: dict) -> dict:
    t0 = time.perf_counter()

    mesh_resolution = 96
    degree = 1

    try:
        g = case_spec["output"]["grid"]
        target = max(int(g["nx"]), int(g["ny"]))
        if target >= 192:
            mesh_resolution = 112
        elif target <= 64:
            mesh_resolution = 80
    except Exception:
        pass

    msh, uh, solver_info = _build_mesh_and_solve(mesh_resolution, degree)

    if time.perf_counter() - t0 < 0.9:
        try:
            refined_n = min(mesh_resolution + 16, 128)
            msh2, uh2, _ = _build_mesh_and_solve(refined_n, degree)
            cmp_grid = {"nx": 41, "ny": 41, "bbox": [0.0, 1.0, 0.0, 1.0]}
            u1 = _sample_on_grid(msh, uh, cmp_grid)
            u2 = _sample_on_grid(msh2, uh2, cmp_grid)
            diff = u2 - u1
            solver_info["mesh_comparison_linf"] = float(np.max(np.abs(diff)))
            solver_info["mesh_comparison_l2_grid"] = float(np.sqrt(np.mean(diff**2)))
            if time.perf_counter() - t0 < 1.6:
                msh, uh = msh2, uh2
                solver_info["mesh_resolution"] = refined_n
        except Exception as e:
            solver_info["mesh_comparison_error"] = str(e)

    u_grid = _sample_on_grid(msh, uh, case_spec["output"]["grid"])
    solver_info["wall_time_sec_estimate"] = float(time.perf_counter() - t0)

    return {"u": u_grid, "solver_info": solver_info}
