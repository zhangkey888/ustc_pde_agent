import time
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl

# ```DIAGNOSIS
# equation_type: convection_diffusion
# spatial_dim: 2
# domain_geometry: rectangle
# unknowns: scalar
# coupling: none
# linearity: linear
# time_dependence: steady
# stiffness: stiff
# dominant_physics: mixed
# peclet_or_reynolds: high
# solution_regularity: smooth
# bc_type: all_dirichlet
# special_notes: manufactured_solution
# ```

# ```METHOD
# spatial_method: fem
# element_or_basis: Lagrange_P2
# stabilization: supg
# time_method: none
# nonlinear_solver: none
# linear_solver: gmres
# preconditioner: ilu
# special_treatment: none
# pde_skill: convection_diffusion / reaction_diffusion
# ```

ScalarType = PETSc.ScalarType


def _sample_on_grid(domain, uh, grid_spec):
    nx = int(grid_spec["nx"])
    ny = int(grid_spec["ny"])
    xmin, xmax, ymin, ymax = grid_spec["bbox"]

    xs = np.linspace(xmin, xmax, nx, dtype=np.float64)
    ys = np.linspace(ymin, ymax, ny, dtype=np.float64)
    XX, YY = np.meshgrid(xs, ys, indexing="xy")
    pts = np.zeros((nx * ny, 3), dtype=np.float64)
    pts[:, 0] = XX.ravel()
    pts[:, 1] = YY.ravel()

    tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(tree, pts)
    colliding = geometry.compute_colliding_cells(domain, cell_candidates, pts)

    local_vals = np.full(nx * ny, np.nan, dtype=np.float64)
    points_on_proc = []
    cells_on_proc = []
    eval_map = []

    for i in range(pts.shape[0]):
        links = colliding.links(i)
        if len(links) > 0:
            points_on_proc.append(pts[i])
            cells_on_proc.append(links[0])
            eval_map.append(i)

    if points_on_proc:
        vals = uh.eval(np.array(points_on_proc, dtype=np.float64),
                       np.array(cells_on_proc, dtype=np.int32))
        local_vals[np.array(eval_map, dtype=np.int32)] = np.asarray(vals).reshape(-1).real

    gathered = domain.comm.gather(local_vals, root=0)
    if domain.comm.rank == 0:
        merged = np.full(nx * ny, np.nan, dtype=np.float64)
        for arr in gathered:
            mask = ~np.isnan(arr)
            merged[mask] = arr[mask]
        if np.isnan(merged).any():
            raise RuntimeError("Failed to evaluate FEM solution at some output grid points.")
        out = merged.reshape((ny, nx))
    else:
        out = None
    return domain.comm.bcast(out, root=0)


def _build_exact_data(domain):
    x = ufl.SpatialCoordinate(domain)
    pi = np.pi
    eps = 0.03
    beta_np = np.array([5.0, 2.0], dtype=np.float64)
    beta = fem.Constant(domain, beta_np.astype(ScalarType))
    eps_c = fem.Constant(domain, ScalarType(eps))

    u_exact = ufl.sin(pi * x[0]) * ufl.sin(2.0 * pi * x[1])
    f = -eps_c * ufl.div(ufl.grad(u_exact)) + ufl.dot(beta, ufl.grad(u_exact))
    return eps_c, beta, u_exact, f


def _solve_single(n, degree=2, ksp_type="gmres", pc_type="ilu", rtol=1e-10):
    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_square(comm, n, n, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", degree))

    eps_c, beta, u_exact_ufl, f_ufl = _build_exact_data(domain)

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    h = ufl.CellDiameter(domain)
    beta_norm = ufl.sqrt(ufl.dot(beta, beta))
    tau = h / (2.0 * beta_norm + 4.0 * eps_c / h)

    a = (
        eps_c * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
        + ufl.dot(beta, ufl.grad(u)) * v * ufl.dx
        + tau * (-eps_c * ufl.div(ufl.grad(u)) + ufl.dot(beta, ufl.grad(u))) * ufl.dot(beta, ufl.grad(v)) * ufl.dx
    )
    L = (
        f_ufl * v * ufl.dx
        + tau * f_ufl * ufl.dot(beta, ufl.grad(v)) * ufl.dx
    )

    u_bc = fem.Function(V)
    u_bc.interpolate(fem.Expression(u_exact_ufl, V.element.interpolation_points))

    fdim = domain.topology.dim - 1
    facets = mesh.locate_entities_boundary(domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool))
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
    solver.setTolerances(rtol=rtol, atol=1e-14, max_it=5000)

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
        solver = PETSc.KSP().create(comm)
        solver.setOperators(A)
        solver.setType("preonly")
        solver.getPC().setType("lu")
        solver.setTolerances(rtol=1e-12, atol=1e-14, max_it=1)
        solver.solve(b, uh.x.petsc_vec)

    uh.x.scatter_forward()

    e = fem.Function(V)
    e.x.array[:] = uh.x.array - u_bc.x.array

    l2_sq = fem.assemble_scalar(fem.form(ufl.inner(e, e) * ufl.dx))
    l2_sq = comm.allreduce(l2_sq, op=MPI.SUM)
    l2_err = float(np.sqrt(max(l2_sq, 0.0)))

    h1_sq = fem.assemble_scalar(fem.form((ufl.inner(e, e) + ufl.inner(ufl.grad(e), ufl.grad(e))) * ufl.dx))
    h1_sq = comm.allreduce(h1_sq, op=MPI.SUM)
    h1_err = float(np.sqrt(max(h1_sq, 0.0)))

    info = {
        "mesh_resolution": int(n),
        "element_degree": int(degree),
        "ksp_type": solver.getType(),
        "pc_type": solver.getPC().getType(),
        "rtol": float(rtol),
        "iterations": int(solver.getIterationNumber()),
        "l2_error": l2_err,
        "h1_error": h1_err,
    }
    return domain, uh, info


def solve(case_spec: dict) -> dict:
    t0 = time.perf_counter()
    time_limit = 8.788
    target_error = 2.28e-05

    candidates = [64, 96, 128, 160]
    best = None

    for i, n in enumerate(candidates):
        t_start = time.perf_counter()
        domain, uh, info = _solve_single(n=n, degree=2, ksp_type="gmres", pc_type="ilu", rtol=1e-10)
        elapsed_total = time.perf_counter() - t0
        elapsed_this = time.perf_counter() - t_start
        best = (domain, uh, info)

        if info["l2_error"] <= target_error:
            if i + 1 < len(candidates) and (elapsed_total + 1.6 * elapsed_this) < 0.85 * time_limit:
                continue
            break

        if elapsed_total > 0.9 * time_limit:
            break

    domain, uh, info = best
    u_grid = _sample_on_grid(domain, uh, case_spec["output"]["grid"])

    solver_info = {
        "mesh_resolution": info["mesh_resolution"],
        "element_degree": info["element_degree"],
        "ksp_type": info["ksp_type"],
        "pc_type": info["pc_type"],
        "rtol": info["rtol"],
        "iterations": info["iterations"],
        "l2_error": info["l2_error"],
        "h1_error": info["h1_error"],
    }

    return {
        "u": u_grid,
        "solver_info": solver_info,
    }


if __name__ == "__main__":
    case_spec = {
        "output": {
            "grid": {"nx": 64, "ny": 64, "bbox": [0.0, 1.0, 0.0, 1.0]}
        },
        "pde": {"time": None},
    }
    result = solve(case_spec)
    if MPI.COMM_WORLD.rank == 0:
        print(result["u"].shape)
        print(result["solver_info"])
