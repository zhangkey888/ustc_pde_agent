import time
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
import ufl
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
from basix.ufl import element as basix_element, mixed_element as basix_mixed_element

ScalarType = PETSc.ScalarType


def _sample_function_on_grid(uh, msh, grid_spec):
    nx = int(grid_spec["nx"])
    ny = int(grid_spec["ny"])
    xmin, xmax, ymin, ymax = grid_spec["bbox"]
    xs = np.linspace(xmin, xmax, nx)
    ys = np.linspace(ymin, ymax, ny)
    XX, YY = np.meshgrid(xs, ys)
    pts = np.column_stack([XX.ravel(), YY.ravel(), np.zeros(nx * ny, dtype=np.float64)])

    tree = geometry.bb_tree(msh, msh.topology.dim)
    cell_candidates = geometry.compute_collisions_points(tree, pts)
    colliding_cells = geometry.compute_colliding_cells(msh, cell_candidates, pts)

    local_vals = np.full(nx * ny, np.nan, dtype=np.float64)
    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    for i in range(pts.shape[0]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(pts[i])
            cells_on_proc.append(links[0])
            eval_map.append(i)

    if len(points_on_proc) > 0:
        vals = uh.eval(np.array(points_on_proc, dtype=np.float64), np.array(cells_on_proc, dtype=np.int32))
        vals = np.asarray(vals).reshape(len(points_on_proc), -1)[:, 0]
        local_vals[np.array(eval_map, dtype=np.int32)] = vals

    comm = msh.comm
    gathered = comm.gather(local_vals, root=0)
    if comm.rank == 0:
        out = np.full(nx * ny, np.nan, dtype=np.float64)
        for arr in gathered:
            mask = ~np.isnan(arr)
            out[mask] = arr[mask]
        if np.isnan(out).any():
            out = np.nan_to_num(out, nan=0.0)
        out = out.reshape(ny, nx)
    else:
        out = None
    return out


def _solve_once(n, degree=2, ksp_type="preonly", pc_type="lu", rtol=1e-10):
    comm = MPI.COMM_WORLD
    msh = mesh.create_unit_square(comm, n, n, cell_type=mesh.CellType.triangle)
    cell = msh.topology.cell_name()

    scalar_el = basix_element("Lagrange", cell, degree)
    W = fem.functionspace(msh, mixed_element([scalar_el, scalar_el]))
    V_u, _ = W.sub(0).collapse()

    (u, vaux) = ufl.TrialFunctions(W)
    (phi, psi) = ufl.TestFunctions(W)

    x = ufl.SpatialCoordinate(msh)
    u_exact_expr = x[0] * (1.0 - x[0]) * x[1] * (1.0 - x[1])
    f_expr = ufl.div(ufl.grad(ufl.div(ufl.grad(u_exact_expr))))

    a = (
        ufl.inner(ufl.grad(u), ufl.grad(phi)) * ufl.dx
        - ufl.inner(vaux, phi) * ufl.dx
        + ufl.inner(ufl.grad(vaux), ufl.grad(psi)) * ufl.dx
    )
    L = ufl.inner(f_expr, psi) * ufl.dx

    fdim = msh.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(msh, fdim, lambda X: np.ones(X.shape[1], dtype=bool))

    u_bc_fun = fem.Function(V_u)
    u_bc_fun.interpolate(lambda X: X[0] * (1.0 - X[0]) * X[1] * (1.0 - X[1]))
    u_dofs = fem.locate_dofs_topological((W.sub(0), V_u), fdim, boundary_facets)
    bc_u = fem.dirichletbc(u_bc_fun, u_dofs, W.sub(0))

    v_bc_fun = fem.Function(V_u)
    v_bc_fun.interpolate(lambda X: -2.0 * (X[0] * (1.0 - X[0]) + X[1] * (1.0 - X[1])))
    v_dofs = fem.locate_dofs_topological((W.sub(1), V_u), fdim, boundary_facets)
    bc_v = fem.dirichletbc(v_bc_fun, v_dofs, W.sub(1))

    problem = petsc.LinearProblem(
        a,
        L,
        bcs=[bc_u, bc_v],
        petsc_options_prefix=f"biharm_{n}_",
        petsc_options={
            "ksp_type": ksp_type,
            "pc_type": pc_type,
            "ksp_rtol": rtol,
            "ksp_atol": 1e-14,
            "ksp_max_it": 1000,
        },
    )

    t0 = time.perf_counter()
    wh = problem.solve()
    solve_time = time.perf_counter() - t0
    wh.x.scatter_forward()

    uh = wh.sub(0).collapse()

    u_ex = fem.Function(V_u)
    u_ex.interpolate(lambda X: X[0] * (1.0 - X[0]) * X[1] * (1.0 - X[1]))

    err_form = fem.form((uh - u_ex) ** 2 * ufl.dx)
    l2_err = np.sqrt(comm.allreduce(fem.assemble_scalar(err_form), op=MPI.SUM))

    ksp = problem.solver
    its = int(ksp.getIterationNumber())

    return {
        "mesh": msh,
        "uh": uh,
        "l2_error": float(l2_err),
        "solve_time": float(solve_time),
        "iterations": its,
        "ksp_type": ksp.getType(),
        "pc_type": ksp.getPC().getType(),
        "rtol": float(rtol),
        "mesh_resolution": int(n),
        "element_degree": int(degree),
    }


def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD
    target_time = 5.048
    safety_time = 4.6

    candidates = [20, 28, 36, 48, 64]
    best = None

    for n in candidates:
        try:
            result = _solve_once(n=n, degree=2, ksp_type="preonly", pc_type="lu", rtol=1e-10)
        except Exception:
            result = _solve_once(n=n, degree=2, ksp_type="gmres", pc_type="ilu", rtol=1e-10)

        best = result
        if result["solve_time"] > safety_time or result["l2_error"] <= 5e-4:
            break

    grid = case_spec["output"]["grid"]
    u_grid = _sample_function_on_grid(best["uh"], best["mesh"], grid)

    if comm.rank == 0:
        solver_info = {
            "mesh_resolution": best["mesh_resolution"],
            "element_degree": best["element_degree"],
            "ksp_type": best["ksp_type"],
            "pc_type": best["pc_type"],
            "rtol": best["rtol"],
            "iterations": int(best["iterations"]),
            "verification_l2_error": float(best["l2_error"]),
            "wall_time_solve": float(best["solve_time"]),
        }
        return {"u": u_grid, "solver_info": solver_info}
    else:
        return {"u": None, "solver_info": {
            "mesh_resolution": best["mesh_resolution"],
            "element_degree": best["element_degree"],
            "ksp_type": best["ksp_type"],
            "pc_type": best["pc_type"],
            "rtol": best["rtol"],
            "iterations": int(best["iterations"]),
            "verification_l2_error": float(best["l2_error"]),
            "wall_time_solve": float(best["solve_time"]),
        }}


if __name__ == "__main__":
    case_spec = {
        "output": {"grid": {"nx": 64, "ny": 64, "bbox": [0.0, 1.0, 0.0, 1.0]}},
        "pde": {"time": None},
    }
    out = solve(case_spec)
    if MPI.COMM_WORLD.rank == 0:
        print(out["u"].shape)
        print(out["solver_info"])
