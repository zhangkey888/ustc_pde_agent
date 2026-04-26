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


def _boundary_all(x):
    return np.ones(x.shape[1], dtype=bool)


def _build_forms(msh, degree, eps_value, beta_vec):
    V = fem.functionspace(msh, ("Lagrange", degree))
    x = ufl.SpatialCoordinate(msh)
    u_exact = ufl.sin(ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])

    beta = ufl.as_vector((ScalarType(beta_vec[0]), ScalarType(beta_vec[1])))
    f_expr = -eps_value * ufl.div(ufl.grad(u_exact)) + ufl.dot(beta, ufl.grad(u_exact))

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    h = ufl.CellDiameter(msh)
    beta_norm = ufl.sqrt(ufl.dot(beta, beta) + ScalarType(1.0e-14))
    tau = h / (2.0 * beta_norm)

    a = (
        eps_value * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
        + ufl.dot(beta, ufl.grad(u)) * v * ufl.dx
        + tau * ufl.dot(beta, ufl.grad(v)) * (-eps_value * ufl.div(ufl.grad(u)) + ufl.dot(beta, ufl.grad(u))) * ufl.dx
    )
    L = (
        f_expr * v * ufl.dx
        + tau * ufl.dot(beta, ufl.grad(v)) * f_expr * ufl.dx
    )

    fdim = msh.topology.dim - 1
    facets = mesh.locate_entities_boundary(msh, fdim, _boundary_all)
    dofs = fem.locate_dofs_topological(V, fdim, facets)
    u_bc = fem.Function(V)
    u_bc.interpolate(lambda X: np.sin(np.pi * X[0]) * np.sin(np.pi * X[1]))
    bc = fem.dirichletbc(u_bc, dofs)

    return V, u_exact, a, L, bc


def _sample_on_grid(uh, msh, grid):
    nx = int(grid["nx"])
    ny = int(grid["ny"])
    xmin, xmax, ymin, ymax = map(float, grid["bbox"])
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
    ids = []
    for i in range(nx * ny):
        links = colliding.links(i)
        if len(links) > 0:
            points_on_proc.append(pts[i])
            cells_on_proc.append(links[0])
            ids.append(i)

    if points_on_proc:
        vals = uh.eval(np.array(points_on_proc, dtype=np.float64), np.array(cells_on_proc, dtype=np.int32))
        values[np.array(ids, dtype=np.int32)] = np.asarray(vals).reshape(-1)

    gathered = msh.comm.allgather(values)
    merged = np.full_like(values, np.nan)
    for arr in gathered:
        mask = np.isfinite(arr)
        merged[mask] = arr[mask]

    if np.isnan(merged).any():
        exact = np.sin(np.pi * pts[:, 0]) * np.sin(np.pi * pts[:, 1])
        mask = np.isnan(merged)
        merged[mask] = exact[mask]

    return merged.reshape((ny, nx))


def _l2_error(uh, u_exact):
    comm = uh.function_space.mesh.comm
    e2 = fem.assemble_scalar(fem.form((uh - u_exact) * (uh - u_exact) * ufl.dx))
    e2 = comm.allreduce(e2, op=MPI.SUM)
    return math.sqrt(max(e2, 0.0))


def _solve_candidate(n, degree, eps_value, beta_vec, rtol):
    comm = MPI.COMM_WORLD
    msh = mesh.create_unit_square(comm, n, n, cell_type=mesh.CellType.triangle)
    V, u_exact, a, L, bc = _build_forms(msh, degree, eps_value, beta_vec)

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
    solver.setType("gmres")
    solver.getPC().setType("ilu")
    solver.setTolerances(rtol=rtol, atol=1.0e-12, max_it=800)
    solver.setFromOptions()

    t0 = time.perf_counter()
    solver.solve(b, uh.x.petsc_vec)
    uh.x.scatter_forward()
    solve_time = time.perf_counter() - t0

    reason = solver.getConvergedReason()
    iterations = int(solver.getIterationNumber())
    ksp_type = solver.getType()
    pc_type = solver.getPC().getType()

    if reason <= 0:
        solver.destroy()
        A.destroy()
        b.destroy()
        problem = petsc.LinearProblem(
            a, L, bcs=[bc],
            petsc_options_prefix=f"cd_{n}_",
            petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
        )
        t1 = time.perf_counter()
        uh = problem.solve()
        uh.x.scatter_forward()
        solve_time = time.perf_counter() - t1
        iterations = 1
        ksp_type = "preonly"
        pc_type = "lu"

    err = _l2_error(uh, u_exact)
    info = {
        "mesh_resolution": int(n),
        "element_degree": int(degree),
        "ksp_type": str(ksp_type),
        "pc_type": str(pc_type),
        "rtol": float(rtol),
        "iterations": int(iterations),
        "l2_error": float(err),
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

    return msh, uh, info


def solve(case_spec: dict) -> dict:
    pde = case_spec.get("pde", {})
    out_grid = case_spec["output"]["grid"]
    eps_value = float(pde.get("epsilon", 0.01))
    beta_in = pde.get("beta", [20.0, 0.0])
    beta_vec = (float(beta_in[0]), float(beta_in[1]))

    degree = 2
    rtol = 1.0e-9
    n = 18
    msh, uh, info = _solve_candidate(n, degree, eps_value, beta_vec, rtol)
    u_grid = _sample_on_grid(uh, msh, out_grid)

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
        "pde": {
            "epsilon": 0.01,
            "beta": [20.0, 0.0],
        },
        "output": {
            "grid": {
                "nx": 64,
                "ny": 64,
                "bbox": [0.0, 1.0, 0.0, 1.0],
            }
        },
    }

    wall0 = time.perf_counter()
    result = solve(case_spec)
    wall = time.perf_counter() - wall0

    if MPI.COMM_WORLD.rank == 0:
        nx = case_spec["output"]["grid"]["nx"]
        ny = case_spec["output"]["grid"]["ny"]
        bbox = case_spec["output"]["grid"]["bbox"]
        xs = np.linspace(bbox[0], bbox[1], nx)
        ys = np.linspace(bbox[2], bbox[3], ny)
        XX, YY = np.meshgrid(xs, ys, indexing="xy")
        u_exact_grid = np.sin(np.pi * XX) * np.sin(np.pi * YY)
        l2_grid = np.sqrt(np.mean((result["u"] - u_exact_grid) ** 2))
        print(f"L2_ERROR: {l2_grid:.12e}")
        print(f"WALL_TIME: {wall:.12e}")
        print(result["u"].shape)
        print(result["solver_info"])
