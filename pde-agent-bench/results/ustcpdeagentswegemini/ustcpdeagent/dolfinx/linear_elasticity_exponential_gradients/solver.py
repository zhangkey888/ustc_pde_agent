import time
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl


ScalarType = PETSc.ScalarType


def _build_exact_and_rhs(msh, E=1.0, nu=0.33):
    gdim = msh.geometry.dim
    x = ufl.SpatialCoordinate(msh)
    mu = E / (2.0 * (1.0 + nu))
    lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))

    u_exact = ufl.as_vector(
        [
            ufl.exp(2.0 * x[0]) * ufl.sin(ufl.pi * x[1]),
            -ufl.exp(2.0 * x[1]) * ufl.sin(ufl.pi * x[0]),
        ]
    )

    def eps(w):
        return ufl.sym(ufl.grad(w))

    def sigma(w):
        return 2.0 * mu * eps(w) + lam * ufl.tr(eps(w)) * ufl.Identity(gdim)

    f = -ufl.div(sigma(u_exact))
    return u_exact, f, mu, lam


def _sample_function_on_grid(u_fun, nx, ny, bbox):
    msh = u_fun.function_space.mesh
    xs = np.linspace(bbox[0], bbox[1], nx)
    ys = np.linspace(bbox[2], bbox[3], ny)
    XX, YY = np.meshgrid(xs, ys, indexing="xy")
    pts2 = np.column_stack([XX.ravel(), YY.ravel()])
    pts3 = np.zeros((pts2.shape[0], 3), dtype=np.float64)
    pts3[:, :2] = pts2

    tree = geometry.bb_tree(msh, msh.topology.dim)
    candidates = geometry.compute_collisions_points(tree, pts3)
    colliding = geometry.compute_colliding_cells(msh, candidates, pts3)

    local_vals = np.full((pts3.shape[0], msh.geometry.dim), np.nan, dtype=np.float64)
    points_on_proc = []
    cells = []
    mapping = []
    for i in range(pts3.shape[0]):
        links = colliding.links(i)
        if len(links) > 0:
            points_on_proc.append(pts3[i])
            cells.append(links[0])
            mapping.append(i)

    if len(points_on_proc) > 0:
        vals = u_fun.eval(np.array(points_on_proc, dtype=np.float64), np.array(cells, dtype=np.int32))
        local_vals[np.array(mapping, dtype=np.int32), :] = np.asarray(vals, dtype=np.float64)

    gathered = msh.comm.gather(local_vals, root=0)
    if msh.comm.rank == 0:
        vals = np.full_like(local_vals, np.nan)
        for arr in gathered:
            mask = np.isnan(vals[:, 0]) & (~np.isnan(arr[:, 0]))
            vals[mask] = arr[mask]
        mag = np.linalg.norm(vals, axis=1).reshape(ny, nx)
    else:
        mag = None
    mag = msh.comm.bcast(mag, root=0)
    return mag


def _solve_once(n, degree, ksp_type="cg", pc_type="hypre", rtol=1.0e-10):
    comm = MPI.COMM_WORLD
    msh = mesh.create_unit_square(comm, n, n, cell_type=mesh.CellType.triangle)
    gdim = msh.geometry.dim
    V = fem.functionspace(msh, ("Lagrange", degree, (gdim,)))

    u_exact_ufl, f_ufl, mu, lam = _build_exact_and_rhs(msh)

    def eps(w):
        return ufl.sym(ufl.grad(w))

    def sigma(w):
        return 2.0 * mu * eps(w) + lam * ufl.tr(eps(w)) * ufl.Identity(gdim)

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    a = ufl.inner(sigma(u), eps(v)) * ufl.dx
    L = ufl.inner(f_ufl, v) * ufl.dx

    fdim = msh.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(
        msh, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
    )
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)

    u_bc = fem.Function(V)
    expr = fem.Expression(u_exact_ufl, V.element.interpolation_points)
    u_bc.interpolate(expr)
    bc = fem.dirichletbc(u_bc, boundary_dofs)

    A = petsc.assemble_matrix(fem.form(a), bcs=[bc])
    A.assemble()
    b = petsc.create_vector(fem.form(L).function_spaces)
    with b.localForm() as loc:
        loc.set(0.0)
    petsc.assemble_vector(b, fem.form(L))
    petsc.apply_lifting(b, [fem.form(a)], bcs=[[bc]])
    b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    petsc.set_bc(b, [bc])

    uh = fem.Function(V)
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
    solver.setTolerances(rtol=rtol)
    solver.setFromOptions()

    t0 = time.perf_counter()
    try:
        solver.solve(b, uh.x.petsc_vec)
        uh.x.scatter_forward()
    except Exception:
        solver.destroy()
        solver = PETSc.KSP().create(comm)
        solver.setOperators(A)
        solver.setType("preonly")
        solver.getPC().setType("lu")
        solver.setTolerances(rtol=min(rtol, 1.0e-12))
        solver.solve(b, uh.x.petsc_vec)
        uh.x.scatter_forward()
        ksp_used = "preonly"
        pc_used = "lu"
    else:
        ksp_used = solver.getType()
        pc_used = solver.getPC().getType()
        reason = solver.getConvergedReason()
        if reason <= 0:
            solver.destroy()
            solver = PETSc.KSP().create(comm)
            solver.setOperators(A)
            solver.setType("preonly")
            solver.getPC().setType("lu")
            solver.setTolerances(rtol=min(rtol, 1.0e-12))
            solver.solve(b, uh.x.petsc_vec)
            uh.x.scatter_forward()
            ksp_used = "preonly"
            pc_used = "lu"
    solve_time = time.perf_counter() - t0

    Vex = fem.functionspace(msh, ("Lagrange", degree + 2, (gdim,)))
    uex = fem.Function(Vex)
    expr_ex = fem.Expression(u_exact_ufl, Vex.element.interpolation_points)
    uex.interpolate(expr_ex)

    uh_ex = fem.Function(Vex)
    uh_ex.interpolate(uh)

    err_form = fem.form(ufl.inner(uh_ex - uex, uh_ex - uex) * ufl.dx)
    norm_form = fem.form(ufl.inner(uex, uex) * ufl.dx)
    l2_err_sq = fem.assemble_scalar(err_form)
    l2_ref_sq = fem.assemble_scalar(norm_form)
    l2_err_sq = comm.allreduce(l2_err_sq, op=MPI.SUM)
    l2_ref_sq = comm.allreduce(l2_ref_sq, op=MPI.SUM)
    l2_error = np.sqrt(l2_err_sq)
    rel_l2_error = l2_error / np.sqrt(l2_ref_sq)

    iterations = int(solver.getIterationNumber())

    return {
        "mesh": msh,
        "uh": uh,
        "l2_error": float(l2_error),
        "rel_l2_error": float(rel_l2_error),
        "solve_time": float(solve_time),
        "iterations": iterations,
        "ksp_type": str(ksp_used),
        "pc_type": str(pc_used),
        "rtol": float(rtol),
        "mesh_resolution": int(n),
        "element_degree": int(degree),
    }


def solve(case_spec: dict) -> dict:
    t_start = time.perf_counter()
    time_limit = 16.420
    target_err = 3.39e-4

    out_grid = case_spec["output"]["grid"]
    nx = int(out_grid["nx"])
    ny = int(out_grid["ny"])
    bbox = out_grid["bbox"]

    pde = case_spec.get("pde", {})
    material = case_spec.get("material", {})
    E = float(material.get("E", 1.0))
    nu = float(material.get("nu", 0.33))

    degree = 2 if nu > 0.4 else 1

    candidates = [48, 64, 80, 96, 112, 128]
    best = None
    elapsed = 0.0

    for n in candidates:
        remaining = time_limit - (time.perf_counter() - t_start)
        if remaining < 2.0 and best is not None:
            break
        result = _solve_once(n=n, degree=degree, ksp_type="cg", pc_type="hypre", rtol=1.0e-10)
        best = result
        elapsed = time.perf_counter() - t_start

        if result["l2_error"] <= target_err:
            if elapsed < 0.55 * time_limit:
                continue
            break

    if best is None:
        best = _solve_once(n=64, degree=degree, ksp_type="cg", pc_type="hypre", rtol=1.0e-10)

    u_grid = _sample_function_on_grid(best["uh"], nx=nx, ny=ny, bbox=bbox)

    solver_info = {
        "mesh_resolution": int(best["mesh_resolution"]),
        "element_degree": int(best["element_degree"]),
        "ksp_type": best["ksp_type"],
        "pc_type": best["pc_type"],
        "rtol": float(best["rtol"]),
        "iterations": int(best["iterations"]),
        "l2_error": float(best["l2_error"]),
        "relative_l2_error": float(best["rel_l2_error"]),
        "wall_time_sec_est": float(time.perf_counter() - t_start),
    }

    if pde.get("time") is not None:
        solver_info["dt"] = float(pde.get("dt", 0.0) or 0.0)
        solver_info["n_steps"] = int(pde.get("n_steps", 0) or 0)
        solver_info["time_scheme"] = str(pde.get("time_scheme", "none"))

    return {
        "u": np.asarray(u_grid, dtype=np.float64).reshape(ny, nx),
        "solver_info": solver_info,
    }
