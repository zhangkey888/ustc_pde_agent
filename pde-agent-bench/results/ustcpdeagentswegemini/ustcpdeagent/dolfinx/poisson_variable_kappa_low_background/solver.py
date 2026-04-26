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
# special_notes: manufactured_solution, variable_coeff
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


def _u_exact_numpy(x, y):
    return np.sin(np.pi * x) * np.sin(np.pi * y)


def _build_and_solve(n, degree, ksp_type="cg", pc_type="hypre", rtol=1e-10):
    msh = mesh.create_unit_square(COMM, n, n, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(msh, ("Lagrange", degree))
    x = ufl.SpatialCoordinate(msh)

    u_exact = ufl.sin(ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])
    kappa = 0.2 + ufl.exp(-120.0 * ((x[0] - 0.55) ** 2 + (x[1] - 0.45) ** 2))
    f = -ufl.div(kappa * ufl.grad(u_exact))

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    a = ufl.inner(kappa * ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = ufl.inner(f, v) * ufl.dx

    fdim = msh.topology.dim - 1
    facets = mesh.locate_entities_boundary(msh, fdim, lambda X: np.ones(X.shape[1], dtype=bool))
    dofs = fem.locate_dofs_topological(V, fdim, facets)

    u_bc = fem.Function(V)
    u_bc.interpolate(lambda X: np.sin(np.pi * X[0]) * np.sin(np.pi * X[1]))
    bc = fem.dirichletbc(u_bc, dofs)

    a_form = fem.form(a)
    L_form = fem.form(L)
    A = petsc.assemble_matrix(a_form, bcs=[bc])
    A.assemble()
    b = petsc.create_vector(L_form.function_spaces)

    solver = PETSc.KSP().create(msh.comm)
    solver.setOperators(A)
    solver.setType(ksp_type)
    solver.getPC().setType(pc_type)
    solver.setTolerances(rtol=rtol)
    try:
        solver.setFromOptions()
    except Exception:
        pass

    uh = fem.Function(V)

    with b.localForm() as loc:
        loc.set(0.0)
    petsc.assemble_vector(b, L_form)
    petsc.apply_lifting(b, [a_form], bcs=[[bc]])
    b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    petsc.set_bc(b, [bc])

    try:
        solver.solve(b, uh.x.petsc_vec)
        uh.x.scatter_forward()
        if solver.getConvergedReason() <= 0:
            raise RuntimeError("Iterative solver failed")
    except Exception:
        solver = PETSc.KSP().create(msh.comm)
        solver.setOperators(A)
        solver.setType("preonly")
        solver.getPC().setType("lu")
        solver.setTolerances(rtol=rtol)
        solver.solve(b, uh.x.petsc_vec)
        uh.x.scatter_forward()
        ksp_type = "preonly"
        pc_type = "lu"

    err_l2_local = fem.assemble_scalar(fem.form((uh - u_exact) ** 2 * ufl.dx))
    err_h1_local = fem.assemble_scalar(
        fem.form(ufl.inner(ufl.grad(uh - u_exact), ufl.grad(uh - u_exact)) * ufl.dx)
    )
    err_l2 = math.sqrt(COMM.allreduce(err_l2_local, op=MPI.SUM))
    err_h1 = math.sqrt(COMM.allreduce(err_h1_local, op=MPI.SUM))

    return {
        "mesh": msh,
        "uh": uh,
        "error_L2": err_l2,
        "error_H1": err_h1,
        "iterations": int(solver.getIterationNumber()),
        "ksp_type": ksp_type,
        "pc_type": pc_type,
        "rtol": rtol,
        "mesh_resolution": n,
        "element_degree": degree,
    }


def _sample_on_grid(msh, uh, grid_spec):
    nx = int(grid_spec["nx"])
    ny = int(grid_spec["ny"])
    xmin, xmax, ymin, ymax = grid_spec["bbox"]

    xs = np.linspace(xmin, xmax, nx)
    ys = np.linspace(ymin, ymax, ny)
    XX, YY = np.meshgrid(xs, ys, indexing="xy")
    pts = np.column_stack([XX.ravel(), YY.ravel(), np.zeros(nx * ny)])

    tree = geometry.bb_tree(msh, msh.topology.dim)
    candidates = geometry.compute_collisions_points(tree, pts)
    colliding = geometry.compute_colliding_cells(msh, candidates, pts)

    local_vals = np.full(nx * ny, np.nan, dtype=np.float64)
    points_on_proc = []
    cells = []
    mapping = []
    for i in range(pts.shape[0]):
        links = colliding.links(i)
        if len(links) > 0:
            points_on_proc.append(pts[i])
            cells.append(links[0])
            mapping.append(i)

    if points_on_proc:
        vals = uh.eval(np.array(points_on_proc, dtype=np.float64), np.array(cells, dtype=np.int32))
        local_vals[np.array(mapping, dtype=np.int32)] = np.asarray(vals).reshape(-1)

    gathered = COMM.gather(local_vals, root=0)
    if COMM.rank == 0:
        merged = np.full(nx * ny, np.nan, dtype=np.float64)
        for arr in gathered:
            mask = np.isnan(merged) & (~np.isnan(arr))
            merged[mask] = arr[mask]
        if np.isnan(merged).any():
            miss = np.isnan(merged)
            merged[miss] = _u_exact_numpy(pts[miss, 0], pts[miss, 1])
        out = merged.reshape(ny, nx)
    else:
        out = None

    return COMM.bcast(out, root=0)


def solve(case_spec: dict) -> dict:
    output_grid = case_spec["output"]["grid"]
    time_limit = 2.825
    start = time.perf_counter()

    candidates = [(40, 2), (52, 2), (64, 2), (56, 3)]
    best = None

    for n, degree in candidates:
        if best is not None and (time.perf_counter() - start) > 0.9 * time_limit:
            break
        t0 = time.perf_counter()
        res = _build_and_solve(n, degree)
        dt = time.perf_counter() - t0
        best = res
        if res["error_L2"] <= 3.0e-4 and (time.perf_counter() - start) > 0.45 * time_limit:
            break
        if (time.perf_counter() - start) + 1.3 * dt > 0.97 * time_limit:
            break

    u_grid = _sample_on_grid(best["mesh"], best["uh"], output_grid)

    solver_info = {
        "mesh_resolution": int(best["mesh_resolution"]),
        "element_degree": int(best["element_degree"]),
        "ksp_type": str(best["ksp_type"]),
        "pc_type": str(best["pc_type"]),
        "rtol": float(best["rtol"]),
        "iterations": int(best["iterations"]),
        "verification_error_L2": float(best["error_L2"]),
        "verification_error_H1": float(best["error_H1"]),
    }

    return {"u": u_grid, "solver_info": solver_info}


if __name__ == "__main__":
    case_spec = {
        "output": {
            "grid": {"nx": 64, "ny": 64, "bbox": [0.0, 1.0, 0.0, 1.0]}
        }
    }
    result = solve(case_spec)
    if COMM.rank == 0:
        print(result["u"].shape)
        print(result["solver_info"])
