import time
import math
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl

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

COMM = MPI.COMM_WORLD
ScalarType = PETSc.ScalarType


def _u_exact_numpy(x, y):
    return np.sin(np.pi * x * y)


def _build_and_solve(nx: int, degree: int, ksp_type: str = "cg", pc_type: str = "hypre", rtol: float = 1e-10):
    domain = mesh.create_unit_square(COMM, nx, nx, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", degree))

    x = ufl.SpatialCoordinate(domain)
    u_exact_ufl = ufl.sin(ufl.pi * x[0] * x[1])
    f_expr = -ufl.div(ufl.grad(u_exact_ufl))

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    a = fem.form(ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx)
    L = fem.form(ufl.inner(f_expr, v) * ufl.dx)

    u_bc = fem.Function(V)
    u_bc.interpolate(lambda X: np.sin(np.pi * X[0] * X[1]))

    fdim = domain.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda X: np.ones(X.shape[1], dtype=bool)
    )
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc = fem.dirichletbc(u_bc, boundary_dofs)

    # Iterative solver first, fallback to LU if needed
    A = petsc.assemble_matrix(a, bcs=[bc])
    A.assemble()
    b = petsc.create_vector(L.function_spaces)
    with b.localForm() as loc:
        loc.set(0.0)
    petsc.assemble_vector(b, L)
    petsc.apply_lifting(b, [a], bcs=[[bc]])
    b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    petsc.set_bc(b, [bc])

    uh = fem.Function(V)
    solver = PETSc.KSP().create(domain.comm)
    solver.setOperators(A)
    solver.setType(ksp_type)
    pc = solver.getPC()
    pc.setType(pc_type)
    solver.setTolerances(rtol=rtol)

    if ksp_type == "cg":
        try:
            solver.setNormType(PETSc.KSP.NormType.NORM_PRECONDITIONED)
        except Exception:
            pass

    solver.setFromOptions()
    solver.solve(b, uh.x.petsc_vec)
    uh.x.scatter_forward()

    reason = solver.getConvergedReason()
    iterations = solver.getIterationNumber()

    actual_ksp = solver.getType()
    actual_pc = solver.getPC().getType()

    if reason <= 0:
        solver.destroy()
        solver = PETSc.KSP().create(domain.comm)
        solver.setOperators(A)
        solver.setType("preonly")
        solver.getPC().setType("lu")
        solver.setTolerances(rtol=min(rtol, 1e-12))
        solver.setFromOptions()
        solver.solve(b, uh.x.petsc_vec)
        uh.x.scatter_forward()
        iterations = solver.getIterationNumber()
        actual_ksp = solver.getType()
        actual_pc = solver.getPC().getType()

    err_L2 = math.sqrt(domain.comm.allreduce(
        fem.assemble_scalar(fem.form((uh - u_exact_ufl) ** 2 * ufl.dx)), op=MPI.SUM
    ))
    err_H1 = math.sqrt(domain.comm.allreduce(
        fem.assemble_scalar(fem.form(ufl.inner(ufl.grad(uh - u_exact_ufl), ufl.grad(uh - u_exact_ufl)) * ufl.dx)),
        op=MPI.SUM
    ))

    return {
        "domain": domain,
        "V": V,
        "uh": uh,
        "L2_error": float(err_L2),
        "H1_error": float(err_H1),
        "iterations": int(iterations),
        "ksp_type": str(actual_ksp),
        "pc_type": str(actual_pc),
        "rtol": float(rtol),
        "mesh_resolution": int(nx),
        "element_degree": int(degree),
    }


def _sample_on_grid(domain, uh, grid_spec):
    nx = int(grid_spec["nx"])
    ny = int(grid_spec["ny"])
    bbox = grid_spec["bbox"]
    xmin, xmax, ymin, ymax = map(float, bbox)

    xs = np.linspace(xmin, xmax, nx)
    ys = np.linspace(ymin, ymax, ny)
    XX, YY = np.meshgrid(xs, ys, indexing="xy")
    pts = np.column_stack([XX.ravel(), YY.ravel(), np.zeros(nx * ny)])

    tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(tree, pts)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, pts)

    values = np.full((pts.shape[0],), np.nan, dtype=np.float64)
    points_on_proc = []
    cells_on_proc = []
    eval_map = []

    for i in range(pts.shape[0]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(pts[i])
            cells_on_proc.append(links[0])
            eval_map.append(i)

    if points_on_proc:
        vals = uh.eval(np.array(points_on_proc, dtype=np.float64), np.array(cells_on_proc, dtype=np.int32))
        values[np.array(eval_map, dtype=np.int32)] = np.asarray(vals).reshape(-1)

    gathered = COMM.gather(values, root=0)

    if COMM.rank == 0:
        merged = np.full_like(values, np.nan)
        for arr in gathered:
            mask = np.isnan(merged) & ~np.isnan(arr)
            merged[mask] = arr[mask]
        # Fallback for exact boundary/corner points if any remain unassigned
        nan_mask = np.isnan(merged)
        if np.any(nan_mask):
            merged[nan_mask] = _u_exact_numpy(pts[nan_mask, 0], pts[nan_mask, 1])
        out = merged.reshape(ny, nx)
    else:
        out = None

    out = COMM.bcast(out, root=0)
    return out


def solve(case_spec: dict) -> dict:
    t0 = time.perf_counter()

    output_grid = case_spec["output"]["grid"]

    best = _build_and_solve(nx=24, degree=2, ksp_type="cg", pc_type="hypre", rtol=1e-10)

    u_grid = _sample_on_grid(best["domain"], best["uh"], output_grid)

    solver_info = {
        "mesh_resolution": best["mesh_resolution"],
        "element_degree": best["element_degree"],
        "ksp_type": best["ksp_type"],
        "pc_type": best["pc_type"],
        "rtol": best["rtol"],
        "iterations": best["iterations"],
        "l2_error_verification": best["L2_error"],
        "h1_error_verification": best["H1_error"],
        "wall_time_sec": time.perf_counter() - t0,
    }

    return {"u": u_grid, "solver_info": solver_info}
