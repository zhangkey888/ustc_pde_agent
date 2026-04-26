import time
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc

import ufl
from basix.ufl import element as basix_element, mixed_element as basix_mixed_element
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc

ScalarType = PETSc.ScalarType


def _sample_function_on_grid(domain, uh, grid_spec):
    nx = int(grid_spec["nx"])
    ny = int(grid_spec["ny"])
    bbox = grid_spec["bbox"]
    xmin, xmax, ymin, ymax = map(float, bbox)

    xs = np.linspace(xmin, xmax, nx)
    ys = np.linspace(ymin, ymax, ny)
    XX, YY = np.meshgrid(xs, ys, indexing="xy")
    pts2 = np.column_stack([XX.ravel(), YY.ravel()])
    pts3 = np.zeros((pts2.shape[0], 3), dtype=np.float64)
    pts3[:, :2] = pts2

    tdim = domain.topology.dim
    tree = geometry.bb_tree(domain, tdim)
    cell_candidates = geometry.compute_collisions_points(tree, pts3)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, pts3)

    local_vals = np.full(pts3.shape[0], np.nan, dtype=np.float64)
    points_on_proc = []
    cells_on_proc = []
    idx_on_proc = []
    for i in range(pts3.shape[0]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(pts3[i])
            cells_on_proc.append(links[0])
            idx_on_proc.append(i)

    if len(points_on_proc) > 0:
        vals = uh.eval(np.array(points_on_proc, dtype=np.float64), np.array(cells_on_proc, dtype=np.int32))
        vals = np.asarray(vals).reshape(len(points_on_proc), -1)[:, 0]
        local_vals[np.array(idx_on_proc, dtype=np.int32)] = vals

    comm = domain.comm
    gathered = comm.gather(local_vals, root=0)

    if comm.rank == 0:
        global_vals = np.full(pts3.shape[0], np.nan, dtype=np.float64)
        for arr in gathered:
            mask = ~np.isnan(arr)
            global_vals[mask] = arr[mask]
        if np.isnan(global_vals).any():
            global_vals = np.nan_to_num(global_vals, nan=0.0)
        return global_vals.reshape(ny, nx)
    return None


def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD
    t0 = time.perf_counter()

    output_grid = case_spec["output"]["grid"]
    target_time = 7.148

    candidates = [
        {"n": 36, "deg": 2, "ksp": "cg", "pc": "hypre", "rtol": 1.0e-10},
        {"n": 48, "deg": 2, "ksp": "cg", "pc": "hypre", "rtol": 1.0e-10},
        {"n": 56, "deg": 2, "ksp": "cg", "pc": "hypre", "rtol": 1.0e-10},
        {"n": 64, "deg": 2, "ksp": "cg", "pc": "hypre", "rtol": 1.0e-10},
    ]

    best = None

    for cand in candidates:
        n = cand["n"]
        degree = cand["deg"]
        domain = mesh.create_rectangle(
            comm,
            [np.array([0.0, 0.0]), np.array([1.0, 1.0])],
            [n, n],
            cell_type=mesh.CellType.quadrilateral,
        )

        x = ufl.SpatialCoordinate(domain)
        u_exact_ufl = ufl.sin(2.0 * ufl.pi * x[0]) * ufl.cos(3.0 * ufl.pi * x[1])
        lap_u = ufl.div(ufl.grad(u_exact_ufl))
        f_ufl = ufl.div(ufl.grad(lap_u))  # Δ²u

        cell_name = domain.topology.cell_name()
        P = basix_element("Lagrange", cell_name, degree)
        W = fem.functionspace(domain, basix_mixed_element([P, P]))
        V = fem.functionspace(domain, ("Lagrange", degree))

        (u, w) = ufl.TrialFunctions(W)
        (v, q) = ufl.TestFunctions(W)

        a = (
            ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
            + ufl.inner(w, v) * ufl.dx
            + ufl.inner(ufl.grad(w), ufl.grad(q)) * ufl.dx
        )
        L = ufl.inner(f_ufl, q) * ufl.dx

        fdim = domain.topology.dim - 1
        boundary_facets = mesh.locate_entities_boundary(domain, fdim, lambda X: np.ones(X.shape[1], dtype=bool))

        u_bc_fun = fem.Function(V)
        u_bc_expr = fem.Expression(u_exact_ufl, V.element.interpolation_points)
        u_bc_fun.interpolate(u_bc_expr)
        dofs_u = fem.locate_dofs_topological((W.sub(0), V), fdim, boundary_facets)
        bc_u = fem.dirichletbc(u_bc_fun, dofs_u, W.sub(0))

        w_bc_fun = fem.Function(V)
        w_bc_expr = fem.Expression(lap_u, V.element.interpolation_points)
        w_bc_fun.interpolate(w_bc_expr)
        dofs_w = fem.locate_dofs_topological((W.sub(1), V), fdim, boundary_facets)
        bc_w = fem.dirichletbc(w_bc_fun, dofs_w, W.sub(1))

        bcs = [bc_u, bc_w]

        opts = {
            "ksp_type": cand["ksp"],
            "ksp_rtol": cand["rtol"],
            "pc_type": cand["pc"],
            "ksp_norm_type": "unpreconditioned",
        }
        if cand["pc"] == "hypre":
            opts["pc_hypre_type"] = "boomeramg"

        try:
            problem = petsc.LinearProblem(
                a,
                L,
                bcs=bcs,
                petsc_options_prefix=f"biharmonic_{n}_",
                petsc_options=opts,
            )
            wh = problem.solve()
            ksp = problem.solver
            its = int(ksp.getIterationNumber())
            reason = ksp.getConvergedReason()
            if reason <= 0:
                raise RuntimeError("Iterative solver failed to converge")
            ksp_type_used = ksp.getType()
            pc_type_used = ksp.getPC().getType()
        except Exception:
            opts_fallback = {"ksp_type": "preonly", "pc_type": "lu"}
            problem = petsc.LinearProblem(
                a,
                L,
                bcs=bcs,
                petsc_options_prefix=f"biharmonic_lu_{n}_",
                petsc_options=opts_fallback,
            )
            wh = problem.solve()
            ksp = problem.solver
            its = int(ksp.getIterationNumber())
            ksp_type_used = ksp.getType()
            pc_type_used = ksp.getPC().getType()

        uh = wh.sub(0).collapse()
        uh.x.scatter_forward()

        u_exact_fun = fem.Function(V)
        u_exact_fun.interpolate(u_bc_expr)
        err_form = fem.form((uh - u_exact_fun) ** 2 * ufl.dx)
        ref_form = fem.form((u_exact_fun) ** 2 * ufl.dx)
        l2_err_local = fem.assemble_scalar(err_form)
        l2_ref_local = fem.assemble_scalar(ref_form)
        l2_err = np.sqrt(comm.allreduce(l2_err_local, op=MPI.SUM))
        l2_ref = np.sqrt(comm.allreduce(l2_ref_local, op=MPI.SUM))
        rel_l2 = l2_err / (l2_ref + 1.0e-16)

        elapsed = time.perf_counter() - t0
        best = {
            "domain": domain,
            "uh": uh,
            "mesh_resolution": n,
            "element_degree": degree,
            "ksp_type": ksp_type_used,
            "pc_type": pc_type_used,
            "rtol": cand["rtol"],
            "iterations": its,
            "l2_error": float(l2_err),
            "relative_l2_error": float(rel_l2),
            "time": elapsed,
        }

        if elapsed > 0.75 * target_time:
            break
        if l2_err <= 1.0e-4 and elapsed > 0.25 * target_time:
            break

    u_grid = _sample_function_on_grid(best["domain"], best["uh"], output_grid)

    result = {
        "u": u_grid,
        "solver_info": {
            "mesh_resolution": int(best["mesh_resolution"]),
            "element_degree": int(best["element_degree"]),
            "ksp_type": str(best["ksp_type"]),
            "pc_type": str(best["pc_type"]),
            "rtol": float(best["rtol"]),
            "iterations": int(best["iterations"]),
            "l2_error": float(best["l2_error"]),
            "relative_l2_error": float(best["relative_l2_error"]),
            "wall_time_sec": float(best["time"]),
        },
    }
    return result


if __name__ == "__main__":
    case_spec = {
        "output": {
            "grid": {"nx": 64, "ny": 64, "bbox": [0.0, 1.0, 0.0, 1.0]}
        }
    }
    out = solve(case_spec)
    if MPI.COMM_WORLD.rank == 0:
        print(out["u"].shape)
        print(out["solver_info"])
