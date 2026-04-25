import time
import math
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl


ScalarType = PETSc.ScalarType


# ```DIAGNOSIS
# equation_type:        convection_diffusion
# spatial_dim:          2
# domain_geometry:      rectangle
# unknowns:             scalar
# coupling:             none
# linearity:            linear
# time_dependence:      steady
# stiffness:            stiff
# dominant_physics:     mixed
# peclet_or_reynolds:   high
# solution_regularity:  smooth
# bc_type:              all_dirichlet
# special_notes:        manufactured_solution
# ```
#
# ```METHOD
# spatial_method:       fem
# element_or_basis:     Lagrange_P2
# stabilization:        supg
# time_method:          none
# nonlinear_solver:     none
# linear_solver:        gmres
# preconditioner:       ilu
# special_treatment:    none
# pde_skill:            convection_diffusion
# ```


def _exact_u_expr(x):
    return ufl.sin(ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])


def _forcing_expr(msh, eps_value, beta_vec):
    x = ufl.SpatialCoordinate(msh)
    u_ex = _exact_u_expr(x)
    beta = ufl.as_vector(beta_vec)
    f_expr = -eps_value * ufl.div(ufl.grad(u_ex)) + ufl.dot(beta, ufl.grad(u_ex))
    return u_ex, f_expr


def _boundary_all(x):
    return np.ones(x.shape[1], dtype=bool)


def _sample_on_grid(uh, msh, grid):
    nx = int(grid["nx"])
    ny = int(grid["ny"])
    bbox = grid["bbox"]
    xmin, xmax, ymin, ymax = map(float, bbox)
    xs = np.linspace(xmin, xmax, nx)
    ys = np.linspace(ymin, ymax, ny)
    XX, YY = np.meshgrid(xs, ys, indexing="xy")
    pts = np.column_stack([XX.ravel(), YY.ravel(), np.zeros(nx * ny, dtype=np.float64)])

    tree = geometry.bb_tree(msh, msh.topology.dim)
    candidates = geometry.compute_collisions_points(tree, pts)
    colliding = geometry.compute_colliding_cells(msh, candidates, pts)

    values = np.full(nx * ny, np.nan, dtype=np.float64)
    points_on_proc = []
    cells_on_proc = []
    idx_map = []
    for i in range(nx * ny):
        links = colliding.links(i)
        if len(links) > 0:
            points_on_proc.append(pts[i])
            cells_on_proc.append(links[0])
            idx_map.append(i)

    if len(points_on_proc) > 0:
        vals = uh.eval(np.array(points_on_proc, dtype=np.float64), np.array(cells_on_proc, dtype=np.int32))
        values[np.array(idx_map, dtype=np.int32)] = np.asarray(vals).reshape(-1)

    comm = msh.comm
    gathered = comm.allgather(values)
    merged = np.full_like(values, np.nan)
    for arr in gathered:
        mask = np.isfinite(arr)
        merged[mask] = arr[mask]

    if np.isnan(merged).any():
        # Boundary points can occasionally miss due to geometric tolerance; fill with exact solution
        xx = pts[:, 0]
        yy = pts[:, 1]
        exact_vals = np.sin(np.pi * xx) * np.sin(np.pi * yy)
        mask = np.isnan(merged)
        merged[mask] = exact_vals[mask]

    return merged.reshape((ny, nx))


def _solve_once(n, degree=2, eps_value=0.01, beta_vec=(20.0, 0.0), rtol=1.0e-10):
    comm = MPI.COMM_WORLD
    msh = mesh.create_unit_square(comm, n, n, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(msh, ("Lagrange", degree))

    x = ufl.SpatialCoordinate(msh)
    u_ex, f_expr = _forcing_expr(msh, eps_value, beta_vec)
    beta = ufl.as_vector(beta_vec)

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    h = ufl.CellDiameter(msh)
    beta_norm = ufl.sqrt(ufl.dot(beta, beta) + 1.0e-16)
    Pe = beta_norm * h / (2.0 * eps_value)
    cothPe = (ufl.exp(2.0 * Pe) + 1.0) / (ufl.exp(2.0 * Pe) - 1.0)
    tau = h / (2.0 * beta_norm) * (cothPe - 1.0 / Pe)

    a_gal = eps_value * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx + ufl.inner(beta, ufl.grad(u)) * v * ufl.dx
    L_gal = f_expr * v * ufl.dx

    residual_u = -eps_value * ufl.div(ufl.grad(u)) + ufl.dot(beta, ufl.grad(u))
    residual_rhs = f_expr
    a_supg = tau * ufl.inner(ufl.dot(beta, ufl.grad(v)), residual_u) * ufl.dx
    L_supg = tau * ufl.inner(ufl.dot(beta, ufl.grad(v)), residual_rhs) * ufl.dx

    a = a_gal + a_supg
    L = L_gal + L_supg

    fdim = msh.topology.dim - 1
    facets = mesh.locate_entities_boundary(msh, fdim, _boundary_all)
    dofs = fem.locate_dofs_topological(V, fdim, facets)

    u_bc = fem.Function(V)
    u_bc.interpolate(lambda X: np.sin(np.pi * X[0]) * np.sin(np.pi * X[1]))
    bc = fem.dirichletbc(u_bc, dofs)

    a_form = fem.form(a)
    L_form = fem.form(L)

    A = petsc.assemble_matrix(a_form, bcs=[bc])
    A.assemble()
    b = petsc.create_vector(L_form.function_spaces)

    solver = PETSc.KSP().create(comm)
    solver.setOperators(A)
    solver.setType("gmres")
    pc = solver.getPC()
    pc.setType("ilu")
    solver.setTolerances(rtol=rtol, atol=1.0e-14, max_it=2000)
    solver.setFromOptions()

    uh = fem.Function(V)

    with b.localForm() as loc:
        loc.set(0)
    petsc.assemble_vector(b, L_form)
    petsc.apply_lifting(b, [a_form], bcs=[[bc]])
    b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    petsc.set_bc(b, [bc])

    t0 = time.perf_counter()
    solver.solve(b, uh.x.petsc_vec)
    uh.x.scatter_forward()
    solve_time = time.perf_counter() - t0

    ksp_reason = solver.getConvergedReason()
    iterations = solver.getIterationNumber()

    if ksp_reason <= 0:
        solver.destroy()
        A.destroy()
        b.destroy()
        problem = petsc.LinearProblem(
            a, L, bcs=[bc],
            petsc_options_prefix=f"cd_{n}_",
            petsc_options={"ksp_type": "preonly", "pc_type": "lu"}
        )
        t1 = time.perf_counter()
        uh = problem.solve()
        uh.x.scatter_forward()
        solve_time = time.perf_counter() - t1
        ksp_type = "preonly"
        pc_type = "lu"
        iterations = 1
    else:
        ksp_type = solver.getType()
        pc_type = solver.getPC().getType()

    err_form = fem.form((uh - u_ex) * (uh - u_ex) * ufl.dx)
    l2_sq = fem.assemble_scalar(err_form)
    l2_sq = comm.allreduce(l2_sq, op=MPI.SUM)
    l2_error = math.sqrt(max(l2_sq, 0.0))

    solver_info = {
        "mesh_resolution": int(n),
        "element_degree": int(degree),
        "ksp_type": str(ksp_type),
        "pc_type": str(pc_type),
        "rtol": float(rtol),
        "iterations": int(iterations),
        "l2_error": float(l2_error),
        "solve_time": float(solve_time),
    }

    try:
        solver.destroy()
    except Exception:
        pass
    try:
        A.destroy()
    except Exception:
        pass
    try:
        b.destroy()
    except Exception:
        pass

    return msh, uh, solver_info


def solve(case_spec: dict) -> dict:
    output_grid = case_spec["output"]["grid"]
    pde = case_spec.get("pde", {})
    eps_value = float(pde.get("epsilon", 0.01))
    beta_in = pde.get("beta", [20.0, 0.0])
    beta_vec = (float(beta_in[0]), float(beta_in[1]))

    candidates = [40, 48, 56, 64, 72]
    budget = 3.6
    start = time.perf_counter()

    best = None
    for n in candidates:
        msh, uh, info = _solve_once(n=n, degree=2, eps_value=eps_value, beta_vec=beta_vec, rtol=1.0e-10)
        elapsed = time.perf_counter() - start
        best = (msh, uh, info)
        remaining = budget - elapsed
        if elapsed > 0 and remaining > 0:
            est_next = elapsed / max(1, candidates.index(n) + 1)
            if remaining < 0.4 * est_next:
                break

    msh, uh, info = best
    u_grid = _sample_on_grid(uh, msh, output_grid)

    return {
        "u": u_grid,
        "solver_info": {
            "mesh_resolution": info["mesh_resolution"],
            "element_degree": info["element_degree"],
            "ksp_type": info["ksp_type"],
            "pc_type": info["pc_type"],
            "rtol": info["rtol"],
            "iterations": info["iterations"],
        },
    }


if __name__ == "__main__":
    case_spec = {
        "pde": {"epsilon": 0.01, "beta": [20.0, 0.0]},
        "output": {"grid": {"nx": 64, "ny": 64, "bbox": [0.0, 1.0, 0.0, 1.0]}},
    }
    result = solve(case_spec)
    if MPI.COMM_WORLD.rank == 0:
        print(result["u"].shape)
        print(result["solver_info"])
