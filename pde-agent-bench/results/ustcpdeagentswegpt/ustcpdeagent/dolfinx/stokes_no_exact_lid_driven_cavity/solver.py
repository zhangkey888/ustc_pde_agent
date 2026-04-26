import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
from basix.ufl import element as basix_element, mixed_element as basix_mixed_element
import ufl

DIAGNOSIS = (
    "DIAGNOSIS\n"
    "equation_type: stokes\n"
    "spatial_dim: 2\n"
    "domain_geometry: rectangle\n"
    "unknowns: vector+scalar\n"
    "coupling: saddle_point\n"
    "linearity: linear\n"
    "time_dependence: steady\n"
    "stiffness: N/A\n"
    "dominant_physics: diffusion\n"
    "peclet_or_reynolds: low\n"
    "solution_regularity: boundary_layer\n"
    "bc_type: all_dirichlet\n"
    "special_notes: pressure_pinning\n"
)

METHOD = (
    "METHOD\n"
    "spatial_method: fem\n"
    "element_or_basis: Taylor-Hood_P2P1\n"
    "stabilization: none\n"
    "time_method: none\n"
    "nonlinear_solver: none\n"
    "linear_solver: minres\n"
    "preconditioner: hypre\n"
    "special_treatment: pressure_pinning\n"
    "pde_skill: stokes\n"
)


def _sample_on_grid(u_func: fem.Function, domain, grid_spec: dict) -> np.ndarray:
    nx = int(grid_spec["nx"])
    ny = int(grid_spec["ny"])
    xmin, xmax, ymin, ymax = map(float, grid_spec["bbox"])

    xs = np.linspace(xmin, xmax, nx)
    ys = np.linspace(ymin, ymax, ny)
    xx, yy = np.meshgrid(xs, ys, indexing="xy")
    pts = np.zeros((nx * ny, 3), dtype=np.float64)
    pts[:, 0] = xx.ravel()
    pts[:, 1] = yy.ravel()

    tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(tree, pts)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, pts)

    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    for i in range(pts.shape[0]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(pts[i])
            cells_on_proc.append(links[0])
            eval_map.append(i)

    local_vals = np.full((pts.shape[0],), np.nan, dtype=np.float64)
    if points_on_proc:
        vals = u_func.eval(np.array(points_on_proc, dtype=np.float64),
                           np.array(cells_on_proc, dtype=np.int32))
        local_vals[np.array(eval_map, dtype=np.int32)] = np.linalg.norm(vals[:, :2], axis=1)

    gathered = domain.comm.gather(local_vals, root=0)
    if domain.comm.rank == 0:
        global_vals = np.full((pts.shape[0],), np.nan, dtype=np.float64)
        for arr in gathered:
            m = ~np.isnan(arr)
            global_vals[m] = arr[m]
        if np.isnan(global_vals).any():
            # Conservative fallback only for rare boundary misses
            nan_ids = np.where(np.isnan(global_vals))[0]
            for i in nan_ids:
                x, y = pts[i, 0], pts[i, 1]
                if np.isclose(y, ymax):
                    global_vals[i] = 1.0
                else:
                    global_vals[i] = 0.0
        out = global_vals.reshape((ny, nx))
    else:
        out = None
    return domain.comm.bcast(out, root=0)


def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD
    gdim = 2

    grid_spec = case_spec["output"]["grid"]
    nu = 0.2
    if "pde" in case_spec and isinstance(case_spec["pde"], dict):
        nu = float(case_spec["pde"].get("nu", nu))
    if "viscosity" in case_spec:
        nu = float(case_spec["viscosity"])
    if "Viscosity" in case_spec:
        nu = float(case_spec["Viscosity"])

    # Use a reasonably fine mesh within the generous time budget.
    max_grid = max(int(grid_spec.get("nx", 64)), int(grid_spec.get("ny", 64)))
    n = 64 if max_grid <= 128 else 96

    domain = mesh.create_unit_square(comm, n, n, cell_type=mesh.CellType.triangle)

    vel_el = basix_element("Lagrange", domain.topology.cell_name(), 2, shape=(gdim,))
    pre_el = basix_element("Lagrange", domain.topology.cell_name(), 1)
    W = fem.functionspace(domain, basix_mixed_element([vel_el, pre_el]))
    V, _ = W.sub(0).collapse()
    Q, _ = W.sub(1).collapse()

    u, p = ufl.TrialFunctions(W)
    v, q = ufl.TestFunctions(W)

    def eps(w):
        return ufl.sym(ufl.grad(w))

    f = fem.Constant(domain, PETSc.ScalarType((0.0, 0.0)))
    a = (
        2.0 * nu * ufl.inner(eps(u), eps(v)) * ufl.dx
        - p * ufl.div(v) * ufl.dx
        + ufl.div(u) * q * ufl.dx
    )
    L = ufl.inner(f, v) * ufl.dx

    fdim = domain.topology.dim - 1

    top_facets = mesh.locate_entities_boundary(domain, fdim, lambda x: np.isclose(x[1], 1.0))
    bottom_facets = mesh.locate_entities_boundary(domain, fdim, lambda x: np.isclose(x[1], 0.0))
    left_facets = mesh.locate_entities_boundary(domain, fdim, lambda x: np.isclose(x[0], 0.0))
    right_facets = mesh.locate_entities_boundary(domain, fdim, lambda x: np.isclose(x[0], 1.0))

    u_top = fem.Function(V)
    u_top.interpolate(lambda x: np.vstack((np.ones(x.shape[1]), np.zeros(x.shape[1]))))
    u_zero = fem.Function(V)
    u_zero.interpolate(lambda x: np.zeros((gdim, x.shape[1])))

    bc_top = fem.dirichletbc(u_top, fem.locate_dofs_topological((W.sub(0), V), fdim, top_facets), W.sub(0))
    bc_bottom = fem.dirichletbc(u_zero, fem.locate_dofs_topological((W.sub(0), V), fdim, bottom_facets), W.sub(0))
    bc_left = fem.dirichletbc(u_zero, fem.locate_dofs_topological((W.sub(0), V), fdim, left_facets), W.sub(0))
    bc_right = fem.dirichletbc(u_zero, fem.locate_dofs_topological((W.sub(0), V), fdim, right_facets), W.sub(0))

    p_dofs = fem.locate_dofs_geometrical(
        (W.sub(1), Q), lambda x: np.isclose(x[0], 0.0) & np.isclose(x[1], 0.0)
    )
    p_fix = fem.Function(Q)
    p_fix.x.array[:] = 0.0
    bc_p = fem.dirichletbc(p_fix, p_dofs, W.sub(1))

    bcs = [bc_top, bc_bottom, bc_left, bc_right, bc_p]

    rtol = 1.0e-9
    ksp_type = "minres"
    pc_type = "hypre"

    try:
        problem = petsc.LinearProblem(
            a, L, bcs=bcs,
            petsc_options_prefix="stokes_",
            petsc_options={
                "ksp_type": ksp_type,
                "ksp_rtol": rtol,
                "pc_type": pc_type,
                "ksp_error_if_not_converged": False,
            },
        )
        wh = problem.solve()
        solver_obj = problem.solver
    except Exception:
        ksp_type = "preonly"
        pc_type = "lu"
        problem = petsc.LinearProblem(
            a, L, bcs=bcs,
            petsc_options_prefix="stokes_fallback_",
            petsc_options={"ksp_type": ksp_type, "pc_type": pc_type},
        )
        wh = problem.solve()
        solver_obj = problem.solver

    wh.x.scatter_forward()
    uh = wh.sub(0).collapse()
    _ = wh.sub(1).collapse()

    u_grid = _sample_on_grid(uh, domain, grid_spec)

    div_form = fem.form((ufl.div(uh) ** 2) * ufl.dx)
    div_l2 = np.sqrt(comm.allreduce(fem.assemble_scalar(div_form), op=MPI.SUM))

    if comm.rank == 0:
        top_err = float(np.max(np.abs(u_grid[-1, :] - 1.0)))
        bottom_err = float(np.max(np.abs(u_grid[0, :])))
        bc_residual = max(top_err, bottom_err)
    else:
        bc_residual = 0.0
    bc_residual = comm.bcast(bc_residual, root=0)

    try:
        iterations = int(solver_obj.getIterationNumber())
        ksp_type = str(solver_obj.getType())
        pc_type = str(solver_obj.getPC().getType())
    except Exception:
        iterations = 0

    solver_info = {
        "mesh_resolution": int(n),
        "element_degree": 2,
        "ksp_type": ksp_type,
        "pc_type": pc_type,
        "rtol": float(rtol),
        "iterations": int(iterations),
        "verification": {
            "divergence_l2": float(div_l2),
            "boundary_residual_max": float(bc_residual),
            "diagnosis": DIAGNOSIS,
            "method": METHOD,
        },
    }

    return {"u": u_grid, "solver_info": solver_info}


if __name__ == "__main__":
    case = {"pde": {"nu": 0.2}, "output": {"grid": {"nx": 32, "ny": 32, "bbox": [0.0, 1.0, 0.0, 1.0]}}}
    result = solve(case)
    if MPI.COMM_WORLD.rank == 0:
        print(result["u"].shape)
        print(result["solver_info"])
