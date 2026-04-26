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
# special_notes: none
# ```
#
# ```METHOD
# spatial_method: fem
# element_or_basis: Lagrange_P1_or_P2
# stabilization: supg
# time_method: none
# nonlinear_solver: none
# linear_solver: gmres
# preconditioner: ilu
# special_treatment: none
# pde_skill: convection_diffusion
# ```


def _build_problem(n, degree=1, ksp_type="gmres", pc_type="ilu", rtol=1e-9):
    comm = MPI.COMM_WORLD
    msh = mesh.create_unit_square(comm, n, n, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(msh, ("Lagrange", degree))

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    x = ufl.SpatialCoordinate(msh)

    eps = 0.005
    beta_np = np.array([15.0, 7.0], dtype=np.float64)
    beta = fem.Constant(msh, ScalarType(beta_np))
    f_expr = ufl.sin(10.0 * ufl.pi * x[0]) * ufl.sin(8.0 * ufl.pi * x[1])
    f = fem.Expression(f_expr, V.element.interpolation_points)
    f_fun = fem.Function(V)
    f_fun.interpolate(f)

    h = ufl.CellDiameter(msh)
    beta_norm = float(np.linalg.norm(beta_np))
    # Classical linear SUPG parameter for steady convection-diffusion
    tau = 1.0 / ufl.sqrt((2.0 * beta_norm / h) ** 2 + (9.0 * (4.0 * eps / h**2) ** 2))
    r_u = -eps * ufl.div(ufl.grad(u)) + ufl.dot(beta, ufl.grad(u))
    strong_test = ufl.dot(beta, ufl.grad(v))

    a = (
        eps * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
        + ufl.dot(beta, ufl.grad(u)) * v * ufl.dx
        + tau * r_u * strong_test * ufl.dx
    )
    L = (
        f_fun * v * ufl.dx
        + tau * f_fun * strong_test * ufl.dx
    )

    fdim = msh.topology.dim - 1
    facets = mesh.locate_entities_boundary(msh, fdim, lambda X: np.ones(X.shape[1], dtype=bool))
    dofs = fem.locate_dofs_topological(V, fdim, facets)
    bc = fem.dirichletbc(ScalarType(0.0), dofs, V)

    a_form = fem.form(a)
    L_form = fem.form(L)
    A = petsc.assemble_matrix(a_form, bcs=[bc])
    A.assemble()
    b = petsc.create_vector(L_form.function_spaces)
    with b.localForm() as loc:
        loc.set(0.0)
    petsc.assemble_vector(b, L_form)
    petsc.apply_lifting(b, [a_form], bcs=[[bc]])
    b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    petsc.set_bc(b, [bc])

    uh = fem.Function(V)
    solver = PETSc.KSP().create(comm)
    solver.setOperators(A)
    solver.setType(ksp_type)
    pc = solver.getPC()
    try:
        pc.setType(pc_type)
    except Exception:
        pc.setType("jacobi")
        pc_type = "jacobi"
    solver.setTolerances(rtol=rtol, atol=1e-14, max_it=2000)
    if ksp_type.lower() == "gmres":
        solver.setGMRESRestart(200)
    solver.setFromOptions()

    try:
        solver.solve(b, uh.x.petsc_vec)
    except Exception:
        solver.setType("preonly")
        solver.getPC().setType("lu")
        solver.solve(b, uh.x.petsc_vec)
        ksp_type, pc_type = "preonly", "lu"
    uh.x.scatter_forward()

    info = {
        "mesh_resolution": int(n),
        "element_degree": int(degree),
        "ksp_type": solver.getType(),
        "pc_type": solver.getPC().getType(),
        "rtol": float(rtol),
        "iterations": int(solver.getIterationNumber()),
    }
    return msh, V, uh, info


def _sample_on_grid(msh, uh, grid_spec):
    nx = int(grid_spec["nx"])
    ny = int(grid_spec["ny"])
    xmin, xmax, ymin, ymax = grid_spec["bbox"]
    xs = np.linspace(xmin, xmax, nx)
    ys = np.linspace(ymin, ymax, ny)
    XX, YY = np.meshgrid(xs, ys, indexing="xy")
    pts = np.column_stack([XX.ravel(), YY.ravel(), np.zeros(nx * ny)])

    tree = geometry.bb_tree(msh, msh.topology.dim)
    cell_candidates = geometry.compute_collisions_points(tree, pts)
    colliding = geometry.compute_colliding_cells(msh, cell_candidates, pts)

    local_points = []
    local_cells = []
    local_ids = []
    for i in range(pts.shape[0]):
        links = colliding.links(i)
        if len(links) > 0:
            local_points.append(pts[i])
            local_cells.append(links[0])
            local_ids.append(i)

    local_vals = np.full(pts.shape[0], np.nan, dtype=np.float64)
    if local_points:
        vals = uh.eval(np.array(local_points, dtype=np.float64), np.array(local_cells, dtype=np.int32))
        local_vals[np.array(local_ids, dtype=np.int32)] = np.asarray(vals).reshape(-1)

    comm = msh.comm
    gathered = comm.gather(local_vals, root=0)
    if comm.rank == 0:
        out = np.full(pts.shape[0], np.nan, dtype=np.float64)
        for arr in gathered:
            mask = np.isnan(out) & ~np.isnan(arr)
            out[mask] = arr[mask]
        out = np.nan_to_num(out, nan=0.0)
        return out.reshape(ny, nx)
    return None


def _compute_refinement_indicator(grid_coarse, grid_fine):
    if grid_coarse.shape != grid_fine.shape:
        return np.inf
    diff = grid_fine - grid_coarse
    return float(np.linalg.norm(diff.ravel()) / max(np.linalg.norm(grid_fine.ravel()), 1e-14))


def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD
    grid = case_spec["output"]["grid"]
    t0 = time.perf_counter()
    time_budget = 56.394
    safety_budget = 0.88 * time_budget

    candidates = [(128, 1), (160, 1), (192, 1), (160, 2), (192, 2), (224, 2)]
    accepted = None
    accepted_grid = None
    accepted_indicator = None

    last_runtime = 0.0
    for idx, (n, degree) in enumerate(candidates):
        start = time.perf_counter()
        msh, V, uh, info = _build_problem(n=n, degree=degree, ksp_type="gmres", pc_type="ilu", rtol=1e-9)
        u_grid = _sample_on_grid(msh, uh, grid)
        if comm.rank == 0:
            runtime = time.perf_counter() - start
            last_runtime = runtime
            indicator = None
            if accepted_grid is not None:
                indicator = _compute_refinement_indicator(accepted_grid, u_grid)
            accepted = {"msh": msh, "V": V, "uh": uh, "info": info}
            accepted_grid = u_grid
            accepted_indicator = indicator

            elapsed_total = time.perf_counter() - t0
            if idx + 1 < len(candidates):
                remaining = safety_budget - elapsed_total
                est_next = max(runtime * 1.8, 2.0)
                if remaining < est_next:
                    break
        else:
            accepted = {"msh": msh, "V": V, "uh": uh, "info": info}

    result_grid = accepted_grid if comm.rank == 0 else None
    solver_info = accepted["info"].copy()
    if comm.rank == 0:
        solver_info["verification_refinement_indicator"] = (
            None if accepted_indicator is None else float(accepted_indicator)
        )
        solver_info["verification_walltime_last_solve"] = float(last_runtime)
        solver_info["verification_note"] = "SUPG stabilized FEM; refinement indicator compares sampled grids."
    return {"u": result_grid, "solver_info": solver_info}


if __name__ == "__main__":
    case_spec = {
        "output": {
            "grid": {"nx": 64, "ny": 64, "bbox": [0.0, 1.0, 0.0, 1.0]}
        },
        "pde": {"time": None},
    }
    out = solve(case_spec)
    if MPI.COMM_WORLD.rank == 0:
        print(out["u"].shape)
        print(out["solver_info"])
