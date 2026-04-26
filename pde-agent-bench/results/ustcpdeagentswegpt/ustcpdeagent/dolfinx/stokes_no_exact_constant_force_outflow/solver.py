import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
import ufl
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc as fpetsc
from basix.ufl import element as basix_element, mixed_element as basix_mixed_element

# ```DIAGNOSIS
# equation_type: stokes
# spatial_dim: 2
# domain_geometry: rectangle
# unknowns: vector+scalar
# coupling: saddle_point
# linearity: linear
# time_dependence: steady
# stiffness: N/A
# dominant_physics: diffusion
# peclet_or_reynolds: low
# solution_regularity: smooth
# bc_type: mixed
# special_notes: pressure_pinning
# ```

# ```METHOD
# spatial_method: fem
# element_or_basis: Taylor-Hood_P2P1
# stabilization: none
# time_method: none
# nonlinear_solver: none
# linear_solver: minres
# preconditioner: hypre
# special_treatment: pressure_pinning
# pde_skill: stokes
# ```

ScalarType = PETSc.ScalarType


def _sample_function_on_grid(u_func, bbox, nx, ny):
    msh = u_func.function_space.mesh
    xs = np.linspace(float(bbox[0]), float(bbox[1]), int(nx))
    ys = np.linspace(float(bbox[2]), float(bbox[3]), int(ny))
    XX, YY = np.meshgrid(xs, ys, indexing="xy")
    pts2 = np.column_stack([XX.ravel(), YY.ravel()])
    pts3 = np.zeros((pts2.shape[0], 3), dtype=np.float64)
    pts3[:, :2] = pts2

    tree = geometry.bb_tree(msh, msh.topology.dim)
    candidates = geometry.compute_collisions_points(tree, pts3)
    colliding = geometry.compute_colliding_cells(msh, candidates, pts3)

    local_vals = np.full((pts3.shape[0], msh.geometry.dim), np.nan, dtype=np.float64)
    eval_points = []
    eval_cells = []
    eval_ids = []
    for i in range(pts3.shape[0]):
        links = colliding.links(i)
        if len(links) > 0:
            eval_points.append(pts3[i])
            eval_cells.append(links[0])
            eval_ids.append(i)

    if eval_points:
        vals = u_func.eval(np.array(eval_points, dtype=np.float64), np.array(eval_cells, dtype=np.int32))
        vals = np.asarray(vals, dtype=np.float64)
        if vals.ndim == 1:
            vals = vals.reshape(-1, 1)
        local_vals[np.array(eval_ids, dtype=np.int32), : vals.shape[1]] = vals

    comm = msh.comm
    global_vals = np.empty_like(local_vals)
    comm.Allreduce(local_vals, global_vals, op=MPI.MAX)

    if np.isnan(global_vals).any():
        nan_mask = np.isnan(global_vals[:, 0])
        global_vals[nan_mask, :] = 0.0

    mag = np.linalg.norm(global_vals[:, :2], axis=1).reshape((ny, nx))
    return mag


def _build_solver(n):
    comm = MPI.COMM_WORLD
    msh = mesh.create_unit_square(comm, n, n, cell_type=mesh.CellType.triangle)
    gdim = msh.geometry.dim
    cell_name = msh.topology.cell_name()

    vel_el = basix_element("Lagrange", cell_name, 2, shape=(gdim,))
    pre_el = basix_element("Lagrange", cell_name, 1)
    W = fem.functionspace(msh, basix_mixed_element([vel_el, pre_el]))
    V, _ = W.sub(0).collapse()
    Q, _ = W.sub(1).collapse()

    nu = ScalarType(0.4)
    f = fem.Constant(msh, np.array([1.0, 0.0], dtype=ScalarType))
    (u, p) = ufl.TrialFunctions(W)
    (v, q) = ufl.TestFunctions(W)

    def eps(w):
        return ufl.sym(ufl.grad(w))

    a = (
        2.0 * nu * ufl.inner(eps(u), eps(v)) * ufl.dx
        - p * ufl.div(v) * ufl.dx
        + q * ufl.div(u) * ufl.dx
    )
    L = ufl.inner(f, v) * ufl.dx

    fdim = msh.topology.dim - 1

    def _facets(marker):
        return mesh.locate_entities_boundary(msh, fdim, marker)

    left = _facets(lambda x: np.isclose(x[0], 0.0))
    bottom = _facets(lambda x: np.isclose(x[1], 0.0))
    top = _facets(lambda x: np.isclose(x[1], 1.0))
    wall_facets = np.unique(np.concatenate([left, bottom, top]).astype(np.int32))

    u_bc = fem.Function(V)
    u_bc.x.array[:] = 0.0
    dofs_u = fem.locate_dofs_topological((W.sub(0), V), fdim, wall_facets)
    bc_u = fem.dirichletbc(u_bc, dofs_u, W.sub(0))

    p_dofs = fem.locate_dofs_geometrical(
        (W.sub(1), Q), lambda x: np.isclose(x[0], 0.0) & np.isclose(x[1], 0.0)
    )
    p0 = fem.Function(Q)
    p0.x.array[:] = 0.0
    bc_p = fem.dirichletbc(p0, p_dofs, W.sub(1))

    bcs = [bc_u, bc_p]

    opts = {
        "ksp_type": "minres",
        "pc_type": "hypre",
        "ksp_rtol": 1.0e-9,
    }

    problem = fpetsc.LinearProblem(
        a,
        L,
        bcs=bcs,
        petsc_options=opts,
        petsc_options_prefix=f"stokes_{n}_",
    )
    wh = problem.solve()
    wh.x.scatter_forward()

    uh = wh.sub(0).collapse()
    ph = wh.sub(1).collapse()

    # Accuracy verification module
    div_form = fem.form(ufl.inner(ufl.div(uh), ufl.div(uh)) * ufl.dx)
    div_l2_sq_local = fem.assemble_scalar(div_form)
    div_l2_sq = comm.allreduce(div_l2_sq_local, op=MPI.SUM)
    div_l2 = float(np.sqrt(max(div_l2_sq, 0.0)))

    wall_test_points = np.array(
        [
            [0.0, 0.5, 0.0],
            [0.5, 0.0, 0.0],
            [0.5, 1.0, 0.0],
        ],
        dtype=np.float64,
    )
    tree = geometry.bb_tree(msh, msh.topology.dim)
    cand = geometry.compute_collisions_points(tree, wall_test_points)
    coll = geometry.compute_colliding_cells(msh, cand, wall_test_points)
    eval_pts, eval_cells = [], []
    for i in range(wall_test_points.shape[0]):
        links = coll.links(i)
        if len(links) > 0:
            eval_pts.append(wall_test_points[i])
            eval_cells.append(links[0])
    if eval_pts:
        wall_vals = np.asarray(uh.eval(np.array(eval_pts, dtype=np.float64), np.array(eval_cells, dtype=np.int32)))
        bc_res = float(np.max(np.linalg.norm(wall_vals[:, :2], axis=1)))
    else:
        bc_res = 0.0
    bc_res = comm.allreduce(bc_res, op=MPI.MAX)

    ksp = problem.solver
    its = int(ksp.getIterationNumber())

    return {
        "mesh": msh,
        "W": W,
        "V": V,
        "Q": Q,
        "u": uh,
        "p": ph,
        "iterations": its,
        "ksp_type": ksp.getType(),
        "pc_type": ksp.getPC().getType(),
        "rtol": float(ksp.getTolerances()[0]),
        "verification": {
            "divergence_l2": div_l2,
            "boundary_bc_max_residual": bc_res,
        },
    }


def solve(case_spec: dict) -> dict:
    output = case_spec.get("output", {})
    grid = output.get("grid", {})
    nx = int(grid.get("nx", 64))
    ny = int(grid.get("ny", 64))
    bbox = grid.get("bbox", [0.0, 1.0, 0.0, 1.0])

    # Adaptive time-accuracy trade-off: choose a mesh at least as fine as output grid scale,
    # but bounded for robustness/runtime.
    target = max(32, min(96, int(max(nx, ny))))
    if target < 48:
        target = 48

    data = _build_solver(target)
    u_grid = _sample_function_on_grid(data["u"], bbox, nx, ny)

    solver_info = {
        "mesh_resolution": int(target),
        "element_degree": 2,
        "ksp_type": str(data["ksp_type"]),
        "pc_type": str(data["pc_type"]),
        "rtol": float(data["rtol"]),
        "iterations": int(data["iterations"]),
        "verification": data["verification"],
    }

    return {"u": u_grid, "solver_info": solver_info}


if __name__ == "__main__":
    case_spec = {
        "pde": {"type": "stokes", "time": None},
        "output": {"grid": {"nx": 64, "ny": 64, "bbox": [0.0, 1.0, 0.0, 1.0]}},
    }
    out = solve(case_spec)
    if MPI.COMM_WORLD.rank == 0:
        print(out["u"].shape)
        print(out["solver_info"])
