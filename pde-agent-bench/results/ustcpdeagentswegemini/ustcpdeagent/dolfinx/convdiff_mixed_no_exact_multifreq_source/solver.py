import time
import math
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
import ufl
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc

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
# special_notes: multifrequency_source
# ```
#
# ```METHOD
# spatial_method: fem
# element_or_basis: Lagrange_P1
# stabilization: supg
# time_method: none
# nonlinear_solver: none
# linear_solver: gmres
# preconditioner: ilu
# special_treatment: none
# pde_skill: convection_diffusion / reaction_diffusion / biharmonic
# ```

ScalarType = PETSc.ScalarType
COMM = MPI.COMM_WORLD


def _build_problem(n, degree=1, eps=0.01, beta_vals=(12.0, 6.0), tau_scale=1.0):
    msh = mesh.create_unit_square(COMM, n, n, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(msh, ("Lagrange", degree))

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    x = ufl.SpatialCoordinate(msh)

    beta = fem.Constant(msh, np.array(beta_vals, dtype=np.float64))
    eps_c = fem.Constant(msh, ScalarType(eps))
    f_expr = ufl.sin(8.0 * ufl.pi * x[0]) * ufl.sin(6.0 * ufl.pi * x[1]) + 0.3 * ufl.sin(12.0 * ufl.pi * x[0]) * ufl.sin(10.0 * ufl.pi * x[1])

    fdim = msh.topology.dim - 1
    facets = mesh.locate_entities_boundary(msh, fdim, lambda X: np.ones(X.shape[1], dtype=bool))
    dofs = fem.locate_dofs_topological(V, fdim, facets)
    bc = fem.dirichletbc(ScalarType(0.0), dofs, V)

    h = ufl.CellDiameter(msh)
    beta_norm = ufl.sqrt(ufl.inner(beta, beta) + 1.0e-16)
    Pe = beta_norm * h / (2.0 * eps_c + 1.0e-16)
    cothPe = (ufl.exp(2.0 * Pe) + 1.0) / (ufl.exp(2.0 * Pe) - 1.0 + 1.0e-16)
    tau = tau_scale * h / (2.0 * beta_norm) * (cothPe - 1.0 / (Pe + 1.0e-16))

    a = (
        eps_c * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
        + ufl.inner(beta, ufl.grad(u)) * v * ufl.dx
        + tau * (ufl.inner(beta, ufl.grad(u)) - eps_c * ufl.div(ufl.grad(u))) * ufl.inner(beta, ufl.grad(v)) * ufl.dx
    )
    L = f_expr * v * ufl.dx + tau * f_expr * ufl.inner(beta, ufl.grad(v)) * ufl.dx

    return msh, V, a, L, [bc]


def _solve_once(n, degree=1, eps=0.01, beta_vals=(12.0, 6.0), rtol=1e-9, tau_scale=1.0):
    msh, V, a, L, bcs = _build_problem(n=n, degree=degree, eps=eps, beta_vals=beta_vals, tau_scale=tau_scale)
    opts = {
        "ksp_type": "gmres",
        "pc_type": "ilu",
        "ksp_rtol": rtol,
        "ksp_atol": 1.0e-12,
        "ksp_max_it": 5000,
    }

    uh = fem.Function(V)
    try:
        problem = petsc.LinearProblem(a, L, u=uh, bcs=bcs, petsc_options=opts, petsc_options_prefix=f"cd_{n}_{degree}_")
        uh = problem.solve()
    except Exception:
        opts = {"ksp_type": "preonly", "pc_type": "lu"}
        problem = petsc.LinearProblem(a, L, u=uh, bcs=bcs, petsc_options=opts, petsc_options_prefix=f"cdlu_{n}_{degree}_")
        uh = problem.solve()

    uh.x.scatter_forward()
    ksp = problem.solver
    return msh, uh, {
        "mesh_resolution": int(n),
        "element_degree": int(degree),
        "ksp_type": str(ksp.getType()),
        "pc_type": str(ksp.getPC().getType()),
        "rtol": float(rtol),
        "iterations": int(ksp.getIterationNumber()),
    }


def _sample_on_grid(msh, uh, grid_spec):
    nx = int(grid_spec["nx"])
    ny = int(grid_spec["ny"])
    bbox = grid_spec["bbox"]
    xs = np.linspace(float(bbox[0]), float(bbox[1]), nx)
    ys = np.linspace(float(bbox[2]), float(bbox[3]), ny)
    XX, YY = np.meshgrid(xs, ys, indexing="xy")
    pts = np.column_stack([XX.ravel(), YY.ravel(), np.zeros(nx * ny, dtype=np.float64)])

    tree = geometry.bb_tree(msh, msh.topology.dim)
    cell_candidates = geometry.compute_collisions_points(tree, pts)
    colliding = geometry.compute_colliding_cells(msh, cell_candidates, pts)

    local_points, local_cells, mapping = [], [], []
    for i in range(pts.shape[0]):
        links = colliding.links(i)
        if len(links) > 0:
            local_points.append(pts[i])
            local_cells.append(links[0])
            mapping.append(i)

    vals_local = np.full(pts.shape[0], np.nan, dtype=np.float64)
    if local_points:
        vals = uh.eval(np.array(local_points, dtype=np.float64), np.array(local_cells, dtype=np.int32))
        vals_local[np.array(mapping, dtype=np.int32)] = np.asarray(vals).reshape(-1)

    gathered = COMM.gather(vals_local, root=0)
    if COMM.rank == 0:
        vals = gathered[0].copy()
        for arr in gathered[1:]:
            mask = np.isnan(vals) & np.isfinite(arr)
            vals[mask] = arr[mask]
        vals = np.where(np.isnan(vals), 0.0, vals)
        grid = vals.reshape(ny, nx)
    else:
        grid = None
    return COMM.bcast(grid, root=0)


def _coarse_fine_indicator(uh_coarse, uh_fine):
    Vc = uh_coarse.function_space
    uf_on_c = fem.Function(Vc)
    expr = fem.Expression(uh_fine, Vc.element.interpolation_points)
    uf_on_c.interpolate(expr)
    form = fem.form((uh_coarse - uf_on_c) ** 2 * ufl.dx)
    local = fem.assemble_scalar(form)
    return math.sqrt(max(Vc.mesh.comm.allreduce(local, op=MPI.SUM), 0.0))


def solve(case_spec: dict) -> dict:
    pde = case_spec.get("pde", {})
    output = case_spec.get("output", {})
    grid_spec = output.get("grid", {"nx": 64, "ny": 64, "bbox": [0.0, 1.0, 0.0, 1.0]})
    eps = float(pde.get("epsilon", 0.01))
    beta_raw = pde.get("beta", [12.0, 6.0])
    beta_vals = (float(beta_raw[0]), float(beta_raw[1]))
    time_limit = float(case_spec.get("time_limit", 30.410))
    budget = min(27.5, 0.9 * time_limit)

    candidate_meshes = [48, 64, 80, 96, 112, 128, 144, 160]
    chosen_msh = None
    chosen_uh = None
    chosen_info = None
    prev_uh = None
    verification = {}

    t0 = time.perf_counter()
    for n in candidate_meshes:
        msh, uh, info = _solve_once(n=n, degree=1, eps=eps, beta_vals=beta_vals, rtol=1e-9, tau_scale=1.0)
        elapsed = time.perf_counter() - t0
        chosen_msh, chosen_uh, chosen_info = msh, uh, info
        if prev_uh is not None:
            verification = {
                "mesh_pair": [candidate_meshes[max(0, candidate_meshes.index(n)-1)], n],
                "coarse_fine_l2_difference": float(_coarse_fine_indicator(prev_uh, uh)),
                "elapsed_so_far": float(elapsed),
            }
        else:
            verification = {"mesh_pair": [n], "coarse_fine_l2_difference": None, "elapsed_so_far": float(elapsed)}
        prev_uh = uh
        if elapsed > 0.6 * budget and n >= 96:
            break

    u_grid = _sample_on_grid(chosen_msh, chosen_uh, grid_spec)
    solver_info = dict(chosen_info)
    solver_info["verification"] = verification
    return {"u": u_grid, "solver_info": solver_info}


if __name__ == "__main__":
    case_spec = {
        "pde": {"epsilon": 0.01, "beta": [12.0, 6.0]},
        "output": {"grid": {"nx": 64, "ny": 64, "bbox": [0.0, 1.0, 0.0, 1.0]}},
        "time_limit": 30.410,
    }
    result = solve(case_spec)
    if COMM.rank == 0:
        print(result["u"].shape)
        print(result["solver_info"])
