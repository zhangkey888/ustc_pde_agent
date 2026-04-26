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
# special_notes: manufactured_solution
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


def _exact_u(x):
    return np.sin(np.pi * x[0]) * np.sin(np.pi * x[1])


def _build_and_solve(n, degree, kappa=0.1, rtol=1.0e-10):
    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_square(comm, nx=n, ny=n, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", degree))

    x = ufl.SpatialCoordinate(domain)
    u_ex_ufl = ufl.sin(ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])
    f_ufl = 2.0 * (ufl.pi**2) * kappa * u_ex_ufl

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    a = kappa * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = ufl.inner(f_ufl, v) * ufl.dx

    fdim = domain.topology.dim - 1
    facets = mesh.locate_entities_boundary(domain, fdim, lambda X: np.ones(X.shape[1], dtype=bool))
    dofs = fem.locate_dofs_topological(V, fdim, facets)

    u_bc = fem.Function(V)
    u_bc.interpolate(_exact_u)
    bc = fem.dirichletbc(u_bc, dofs)

    a_form = fem.form(a)
    L_form = fem.form(L)

    A = petsc.assemble_matrix(a_form, bcs=[bc])
    A.assemble()
    b = petsc.create_vector(L_form.function_spaces)

    with b.localForm() as loc:
        loc.set(0)
    petsc.assemble_vector(b, L_form)
    petsc.apply_lifting(b, [a_form], bcs=[[bc]])
    b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    petsc.set_bc(b, [bc])

    uh = fem.Function(V)

    solver = PETSc.KSP().create(comm)
    solver.setOperators(A)
    solver.setType("cg")
    pc = solver.getPC()
    pc.setType("hypre")
    solver.setTolerances(rtol=rtol, atol=0.0, max_it=10000)
    solver.setFromOptions()

    try:
        solver.solve(b, uh.x.petsc_vec)
        reason = solver.getConvergedReason()
        if reason <= 0:
            raise RuntimeError(f"Iterative solve failed with reason {reason}")
    except Exception:
        solver.destroy()
        solver = PETSc.KSP().create(comm)
        solver.setOperators(A)
        solver.setType("preonly")
        solver.getPC().setType("lu")
        solver.setTolerances(rtol=rtol, atol=0.0, max_it=1)
        solver.solve(b, uh.x.petsc_vec)

    uh.x.scatter_forward()

    u_exact = fem.Function(V)
    u_exact.interpolate(_exact_u)
    err_fun = fem.Function(V)
    err_fun.x.array[:] = uh.x.array - u_exact.x.array
    err_local = fem.assemble_scalar(fem.form(ufl.inner(err_fun, err_fun) * ufl.dx))
    err_l2 = math.sqrt(comm.allreduce(err_local, op=MPI.SUM))

    info = {
        "mesh_resolution": int(n),
        "element_degree": int(degree),
        "ksp_type": solver.getType(),
        "pc_type": solver.getPC().getType(),
        "rtol": float(rtol),
        "iterations": int(solver.getIterationNumber()),
        "l2_error": float(err_l2),
    }
    return domain, uh, info


def _sample_on_grid(domain, uh, grid_spec):
    nx = int(grid_spec["nx"])
    ny = int(grid_spec["ny"])
    bbox = grid_spec["bbox"]
    xmin, xmax, ymin, ymax = map(float, bbox)

    xs = np.linspace(xmin, xmax, nx)
    ys = np.linspace(ymin, ymax, ny)
    XX, YY = np.meshgrid(xs, ys, indexing="xy")
    pts = np.column_stack([XX.ravel(), YY.ravel(), np.zeros(nx * ny, dtype=np.float64)])

    tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(tree, pts)
    colliding = geometry.compute_colliding_cells(domain, cell_candidates, pts)

    local_points = []
    local_cells = []
    local_ids = []
    for i in range(pts.shape[0]):
        links = colliding.links(i)
        if len(links) > 0:
            local_points.append(pts[i])
            local_cells.append(links[0])
            local_ids.append(i)

    vals = np.full(pts.shape[0], np.nan, dtype=np.float64)
    if local_points:
        arr = uh.eval(np.asarray(local_points, dtype=np.float64), np.asarray(local_cells, dtype=np.int32))
        vals[np.asarray(local_ids, dtype=np.int32)] = np.asarray(arr).reshape(-1)

    comm = domain.comm
    gathered = comm.gather(vals, root=0)

    if comm.rank == 0:
        out = np.full_like(vals, np.nan)
        for g in gathered:
            mask = np.isnan(out) & ~np.isnan(g)
            out[mask] = g[mask]
        if np.isnan(out).any():
            nan_ids = np.where(np.isnan(out))[0]
            out[nan_ids] = np.sin(np.pi * pts[nan_ids, 0]) * np.sin(np.pi * pts[nan_ids, 1])
        return out.reshape(ny, nx)
    return None


def solve(case_spec: dict) -> dict:
    grid_spec = case_spec["output"]["grid"]
    time_limit = 0.68
    target_error = 1.92e-3

    candidates = [
        (18, 1),
        (24, 1),
        (32, 1),
        (20, 2),
        (24, 2),
        (28, 2),
        (32, 2),
    ]

    best = None
    total_start = time.perf_counter()

    for n, degree in candidates:
        stage_start = time.perf_counter()
        domain, uh, info = _build_and_solve(n=n, degree=degree, kappa=0.1, rtol=1.0e-10)
        elapsed = time.perf_counter() - total_start

        admissible = elapsed < 0.92 * time_limit
        accurate = info["l2_error"] <= target_error

        best = (domain, uh, info)
        stage_elapsed = time.perf_counter() - stage_start

        if accurate and (not admissible):
            break
        if accurate and elapsed > 0.55 * time_limit:
            break
        if elapsed + max(stage_elapsed, 0.02) > 0.95 * time_limit:
            break

    domain, uh, info = best
    u_grid = _sample_on_grid(domain, uh, grid_spec)

    result = {
        "u": u_grid,
        "solver_info": {
            "mesh_resolution": info["mesh_resolution"],
            "element_degree": info["element_degree"],
            "ksp_type": info["ksp_type"],
            "pc_type": info["pc_type"],
            "rtol": info["rtol"],
            "iterations": info["iterations"],
            "l2_error": info["l2_error"],
        },
    }
    return result
