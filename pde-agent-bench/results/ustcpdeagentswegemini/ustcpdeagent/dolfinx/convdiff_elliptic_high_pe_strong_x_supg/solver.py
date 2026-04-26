import time
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl


ScalarType = PETSc.ScalarType


def _probe_function(u_func, pts):
    msh = u_func.function_space.mesh
    tree = geometry.bb_tree(msh, msh.topology.dim)
    candidates = geometry.compute_collisions_points(tree, pts)
    colliding = geometry.compute_colliding_cells(msh, candidates, pts)

    points_local = []
    cells_local = []
    ids_local = []
    for i in range(pts.shape[0]):
        links = colliding.links(i)
        if len(links) > 0:
            points_local.append(pts[i])
            cells_local.append(links[0])
            ids_local.append(i)

    values = np.full((pts.shape[0],), np.nan, dtype=np.float64)
    if points_local:
        vals = u_func.eval(np.array(points_local, dtype=np.float64),
                           np.array(cells_local, dtype=np.int32))
        vals = np.asarray(vals).reshape(len(points_local), -1)[:, 0]
        values[np.array(ids_local, dtype=np.int32)] = vals

    comm = msh.comm
    gathered = comm.allgather(values)
    out = gathered[0].copy()
    for arr in gathered[1:]:
        mask = np.isnan(out) & ~np.isnan(arr)
        out[mask] = arr[mask]
    return out


def _sample_on_grid(u_func, grid_spec):
    nx = int(grid_spec["nx"])
    ny = int(grid_spec["ny"])
    xmin, xmax, ymin, ymax = grid_spec["bbox"]
    xs = np.linspace(xmin, xmax, nx)
    ys = np.linspace(ymin, ymax, ny)
    XX, YY = np.meshgrid(xs, ys, indexing="xy")
    pts = np.column_stack([XX.ravel(), YY.ravel(), np.zeros(nx * ny, dtype=np.float64)])
    vals = _probe_function(u_func, pts)
    return vals.reshape(ny, nx)


def _solve_once(n, degree, eps_val, beta_vec, ksp_type="gmres", pc_type="ilu", rtol=1e-9):
    comm = MPI.COMM_WORLD
    msh = mesh.create_unit_square(comm, n, n, cell_type=mesh.CellType.triangle)

    V = fem.functionspace(msh, ("Lagrange", degree))
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    x = ufl.SpatialCoordinate(msh)

    pi = np.pi
    u_exact_ufl = ufl.sin(pi * x[0]) * ufl.sin(pi * x[1])
    beta = fem.Constant(msh, np.array(beta_vec, dtype=np.float64))
    eps_c = fem.Constant(msh, ScalarType(eps_val))

    grad_u_exact = ufl.grad(u_exact_ufl)
    lap_u_exact = ufl.div(grad_u_exact)
    f_expr = -eps_c * lap_u_exact + ufl.dot(beta, grad_u_exact)

    # SUPG parameter
    h = ufl.CellDiameter(msh)
    beta_norm = ufl.sqrt(ufl.dot(beta, beta) + 1.0e-16)
    tau = h / (2.0 * beta_norm)
    residual_u = -eps_c * ufl.div(ufl.grad(u)) + ufl.dot(beta, ufl.grad(u))
    residual_rhs = f_expr

    a = (
        eps_c * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
        + ufl.dot(beta, ufl.grad(u)) * v * ufl.dx
        + tau * residual_u * ufl.dot(beta, ufl.grad(v)) * ufl.dx
    )
    L = (
        f_expr * v * ufl.dx
        + tau * residual_rhs * ufl.dot(beta, ufl.grad(v)) * ufl.dx
    )

    u_bc = fem.Function(V)
    u_bc.interpolate(fem.Expression(u_exact_ufl, V.element.interpolation_points))
    fdim = msh.topology.dim - 1
    facets = mesh.locate_entities_boundary(msh, fdim, lambda X: np.ones(X.shape[1], dtype=bool))
    dofs = fem.locate_dofs_topological(V, fdim, facets)
    bc = fem.dirichletbc(u_bc, dofs)

    a_form = fem.form(a)
    L_form = fem.form(L)
    A = petsc.assemble_matrix(a_form, bcs=[bc])
    A.assemble()
    b = petsc.create_vector(L_form.function_spaces)

    solver = PETSc.KSP().create(comm)
    solver.setOperators(A)
    solver.setType(ksp_type)
    solver.getPC().setType(pc_type)
    solver.setTolerances(rtol=rtol)

    # Fallback if setup/solve fails for iterative + ilu
    uh = fem.Function(V)
    iterations = 0
    used_ksp = ksp_type
    used_pc = pc_type
    try:
        with b.localForm() as loc:
            loc.set(0)
        petsc.assemble_vector(b, L_form)
        petsc.apply_lifting(b, [a_form], bcs=[[bc]])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        petsc.set_bc(b, [bc])
        solver.solve(b, uh.x.petsc_vec)
        uh.x.scatter_forward()
        reason = solver.getConvergedReason()
        iterations = solver.getIterationNumber()
        if reason <= 0 or np.isnan(np.linalg.norm(uh.x.array)):
            raise RuntimeError(f"KSP failed with reason {reason}")
    except Exception:
        solver.destroy()
        solver = PETSc.KSP().create(comm)
        solver.setOperators(A)
        solver.setType("preonly")
        solver.getPC().setType("lu")
        used_ksp = "preonly"
        used_pc = "lu"
        with b.localForm() as loc:
            loc.set(0)
        petsc.assemble_vector(b, L_form)
        petsc.apply_lifting(b, [a_form], bcs=[[bc]])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        petsc.set_bc(b, [bc])
        solver.solve(b, uh.x.petsc_vec)
        uh.x.scatter_forward()
        iterations = solver.getIterationNumber()

    # Accuracy verification via discrete L2 error against analytical solution
    u_exact = fem.Function(V)
    u_exact.interpolate(fem.Expression(u_exact_ufl, V.element.interpolation_points))
    err_form = fem.form((uh - u_exact) ** 2 * ufl.dx)
    l2_sq_local = fem.assemble_scalar(err_form)
    l2_sq = comm.allreduce(l2_sq_local, op=MPI.SUM)
    l2_error = float(np.sqrt(max(l2_sq, 0.0)))

    return {
        "mesh": msh,
        "space": V,
        "solution": uh,
        "l2_error": l2_error,
        "iterations": int(iterations),
        "ksp_type": used_ksp,
        "pc_type": used_pc,
        "rtol": float(rtol),
        "mesh_resolution": int(n),
        "element_degree": int(degree),
    }


def solve(case_spec: dict) -> dict:
    start = time.perf_counter()

    eps_val = float(case_spec.get("pde", {}).get("epsilon", 0.01))
    beta_vec = case_spec.get("pde", {}).get("beta", [15.0, 0.0])
    grid_spec = case_spec["output"]["grid"]

    # Adaptive time-accuracy tradeoff under approx 1.103s target
    candidates = [(40, 1), (56, 1), (72, 1), (56, 2), (72, 2)]
    budget = 0.92
    best = None

    for n, degree in candidates:
        now = time.perf_counter()
        if now - start > budget and best is not None:
            break
        result = _solve_once(n, degree, eps_val, beta_vec)
        best = result
        # stop early if highly accurate and some budget already used
        if result["l2_error"] < 2.5e-4 and (time.perf_counter() - start) > 0.35:
            break

    u_grid = _sample_on_grid(best["solution"], grid_spec)

    solver_info = {
        "mesh_resolution": best["mesh_resolution"],
        "element_degree": best["element_degree"],
        "ksp_type": best["ksp_type"],
        "pc_type": best["pc_type"],
        "rtol": best["rtol"],
        "iterations": best["iterations"],
        "l2_error_verify": best["l2_error"],
    }

    return {"u": u_grid, "solver_info": solver_info}
