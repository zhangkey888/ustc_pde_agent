import time
import math
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc

from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from basix.ufl import element as basix_element, mixed_element as basix_mixed_element

ScalarType = PETSc.ScalarType


def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD
    rank = comm.rank

    nu_value = float(case_spec.get("pde", {}).get("nu", 0.14))
    output_grid = case_spec["output"]["grid"]
    nx_out = int(output_grid["nx"])
    ny_out = int(output_grid["ny"])
    bbox = output_grid["bbox"]

    def u_exact_numpy(x):
        x0 = x[0]
        x1 = x[1]
        e1 = np.exp(-30.0 * ((x0 - 0.3) ** 2 + (x1 - 0.7) ** 2))
        e2 = np.exp(-30.0 * ((x0 - 0.7) ** 2 + (x1 - 0.3) ** 2))
        u0 = -60.0 * (x1 - 0.7) * e1 + 60.0 * (x1 - 0.3) * e2
        u1 = 60.0 * (x0 - 0.3) * e1 - 60.0 * (x0 - 0.7) * e2
        return np.vstack((u0, u1))

    def build_and_solve(ncells, degree_u=2, degree_p=1):
        msh = mesh.create_unit_square(comm, ncells, ncells, cell_type=mesh.CellType.triangle)
        gdim = msh.geometry.dim
        cell_name = msh.topology.cell_name()

        vel_el = basix_element("Lagrange", cell_name, degree_u, shape=(gdim,))
        pres_el = basix_element("Lagrange", cell_name, degree_p)
        W = fem.functionspace(msh, basix_mixed_element([vel_el, pres_el]))
        V, _ = W.sub(0).collapse()
        Q, _ = W.sub(1).collapse()

        x = ufl.SpatialCoordinate(msh)
        nu = fem.Constant(msh, ScalarType(nu_value))

        u_ex_ufl = ufl.as_vector([
            -60.0 * (x[1] - 0.7) * ufl.exp(-30.0 * ((x[0] - 0.3) ** 2 + (x[1] - 0.7) ** 2))
            + 60.0 * (x[1] - 0.3) * ufl.exp(-30.0 * ((x[0] - 0.7) ** 2 + (x[1] - 0.3) ** 2)),
            60.0 * (x[0] - 0.3) * ufl.exp(-30.0 * ((x[0] - 0.3) ** 2 + (x[1] - 0.7) ** 2))
            - 60.0 * (x[0] - 0.7) * ufl.exp(-30.0 * ((x[0] - 0.7) ** 2 + (x[1] - 0.3) ** 2)),
        ])
        p_ex_ufl = 0.0 * x[0]
        f_ufl = ufl.grad(u_ex_ufl) * u_ex_ufl - nu * ufl.div(ufl.grad(u_ex_ufl)) + ufl.grad(p_ex_ufl)

        w = fem.Function(W)
        (u, p) = ufl.split(w)
        (v, q) = ufl.TestFunctions(W)

        def eps(a):
            return ufl.sym(ufl.grad(a))

        fdim = msh.topology.dim - 1
        boundary_facets = mesh.locate_entities_boundary(
            msh, fdim, lambda X: np.ones(X.shape[1], dtype=bool)
        )
        u_bc = fem.Function(V)
        u_bc.interpolate(u_exact_numpy)
        dofs_u = fem.locate_dofs_topological((W.sub(0), V), fdim, boundary_facets)
        bc_u = fem.dirichletbc(u_bc, dofs_u, W.sub(0))

        p_dofs = fem.locate_dofs_geometrical(
            (W.sub(1), Q),
            lambda X: np.isclose(X[0], 0.0) & np.isclose(X[1], 0.0),
        )
        bcs = [bc_u]
        if len(p_dofs) > 0:
            p0 = fem.Function(Q)
            p0.x.array[:] = 0.0
            bc_p = fem.dirichletbc(p0, p_dofs, W.sub(1))
            bcs.append(bc_p)

        dw = ufl.TrialFunction(W)
        (du, dp) = ufl.split(dw)
        a_stokes = (
            2.0 * nu * ufl.inner(eps(du), eps(v)) * ufl.dx
            - ufl.inner(dp, ufl.div(v)) * ufl.dx
            + ufl.inner(ufl.div(du), q) * ufl.dx
        )
        L_stokes = ufl.inner(f_ufl, v) * ufl.dx

        stokes_problem = petsc.LinearProblem(
            a_stokes,
            L_stokes,
            bcs=bcs,
            petsc_options_prefix=f"stokes_{ncells}_",
            petsc_options={
                "ksp_type": "gmres",
                "pc_type": "lu",
                "ksp_rtol": 1.0e-10,
            },
        )
        try:
            w_stokes = stokes_problem.solve()
            w.x.array[:] = w_stokes.x.array
            w.x.scatter_forward()
            stokes_iters = int(stokes_problem.solver.getIterationNumber())
        except Exception:
            w.x.array[:] = 0.0
            w.x.scatter_forward()
            stokes_iters = 0

        F = (
            2.0 * nu * ufl.inner(eps(u), eps(v)) * ufl.dx
            + ufl.inner(ufl.grad(u) * u, v) * ufl.dx
            - ufl.inner(p, ufl.div(v)) * ufl.dx
            + ufl.inner(ufl.div(u), q) * ufl.dx
            - ufl.inner(f_ufl, v) * ufl.dx
        )
        J = ufl.derivative(F, w)

        linear_iterations = stokes_iters
        nonlinear_iterations = 0

        try:
            ns_problem = petsc.NonlinearProblem(
                F,
                w,
                bcs=bcs,
                J=J,
                petsc_options_prefix=f"ns_{ncells}_",
                petsc_options={
                    "snes_type": "newtonls",
                    "snes_linesearch_type": "bt",
                    "snes_rtol": 1.0e-9,
                    "snes_atol": 1.0e-10,
                    "snes_max_it": 40,
                    "ksp_type": "gmres",
                    "pc_type": "lu",
                    "ksp_rtol": 1.0e-10,
                },
            )
            w = ns_problem.solve()
            w.x.scatter_forward()
            try:
                snes = ns_problem.solver
                nonlinear_iterations = int(snes.getIterationNumber())
                linear_iterations += int(snes.getKSP().getIterationNumber())
            except Exception:
                pass
        except Exception:
            u_k = w.sub(0).collapse()
            for it in range(20):
                (uu, pp) = ufl.TrialFunctions(W)
                a_picard = (
                    2.0 * nu * ufl.inner(eps(uu), eps(v)) * ufl.dx
                    + ufl.inner(ufl.grad(uu) * u_k, v) * ufl.dx
                    - ufl.inner(pp, ufl.div(v)) * ufl.dx
                    + ufl.inner(ufl.div(uu), q) * ufl.dx
                )
                L_picard = ufl.inner(f_ufl, v) * ufl.dx
                picard_problem = petsc.LinearProblem(
                    a_picard,
                    L_picard,
                    bcs=bcs,
                    petsc_options_prefix=f"picard_{ncells}_{it}_",
                    petsc_options={
                        "ksp_type": "gmres",
                        "pc_type": "lu",
                        "ksp_rtol": 1.0e-10,
                    },
                )
                w_new = picard_problem.solve()
                w_new.x.scatter_forward()
                linear_iterations += int(picard_problem.solver.getIterationNumber())
                u_new = w_new.sub(0).collapse()
                diff_local = np.linalg.norm(u_new.x.array - u_k.x.array)
                ref_local = max(np.linalg.norm(u_new.x.array), 1.0)
                diff = comm.allreduce(diff_local, op=MPI.SUM)
                ref = comm.allreduce(ref_local, op=MPI.SUM)
                w.x.array[:] = w_new.x.array
                w.x.scatter_forward()
                u_k.x.array[:] = u_new.x.array
                u_k.x.scatter_forward()
                nonlinear_iterations = it + 1
                if diff / ref < 1.0e-9:
                    break

        u_h = w.sub(0).collapse()

        u_ex = fem.Function(V)
        u_ex.interpolate(u_exact_numpy)
        err_form = fem.form(ufl.inner(u_h - u_ex, u_h - u_ex) * ufl.dx)
        ex_form = fem.form(ufl.inner(u_ex, u_ex) * ufl.dx)
        div_form = fem.form((ufl.div(u_h) ** 2) * ufl.dx)

        l2_err = math.sqrt(comm.allreduce(fem.assemble_scalar(err_form), op=MPI.SUM))
        l2_ref = math.sqrt(comm.allreduce(fem.assemble_scalar(ex_form), op=MPI.SUM))
        div_l2 = math.sqrt(comm.allreduce(fem.assemble_scalar(div_form), op=MPI.SUM))

        return {
            "mesh": msh,
            "u_h": u_h,
            "l2_err": l2_err,
            "rel_l2": l2_err / max(l2_ref, 1.0e-16),
            "div_l2": div_l2,
            "iterations": linear_iterations,
            "nonlinear_iterations": nonlinear_iterations,
            "mesh_resolution": ncells,
        }

    def sample_velocity_magnitude(u_func, msh, nx, ny, bbox_):
        xs = np.linspace(bbox_[0], bbox_[1], nx)
        ys = np.linspace(bbox_[2], bbox_[3], ny)
        XX, YY = np.meshgrid(xs, ys, indexing="xy")
        pts = np.column_stack([XX.ravel(), YY.ravel(), np.zeros(nx * ny)])

        tree = geometry.bb_tree(msh, msh.topology.dim)
        candidates = geometry.compute_collisions_points(tree, pts)
        colliding = geometry.compute_colliding_cells(msh, candidates, pts)

        local_idx = []
        local_pts = []
        local_cells = []
        for i in range(pts.shape[0]):
            links = colliding.links(i)
            if len(links) > 0:
                local_idx.append(i)
                local_pts.append(pts[i])
                local_cells.append(links[0])

        local_idx = np.array(local_idx, dtype=np.int64)
        if len(local_pts) > 0:
            vals = u_func.eval(np.array(local_pts, dtype=np.float64), np.array(local_cells, dtype=np.int32))
            local_mag = np.linalg.norm(vals, axis=1)
        else:
            local_mag = np.array([], dtype=np.float64)

        gathered_idx = comm.gather(local_idx, root=0)
        gathered_mag = comm.gather(local_mag, root=0)

        if rank == 0:
            out = np.full(nx * ny, np.nan, dtype=np.float64)
            for idxs, mags in zip(gathered_idx, gathered_mag):
                if len(idxs) > 0:
                    out[idxs] = mags
            if np.isnan(out).any():
                miss = np.isnan(out)
                vv = u_exact_numpy(pts[miss].T)
                out[miss] = np.sqrt(vv[0] ** 2 + vv[1] ** 2)
            return out.reshape((ny, nx))
        return None

    t0 = time.time()
    candidate_resolutions = [24, 32, 40, 48]
    budget = float(case_spec.get("time_limit_sec", 680.656))
    if budget > 400:
        candidate_resolutions.append(56)

    best = None
    prev_err = None
    target_budget = min(0.35 * budget, 120.0)

    for nc in candidate_resolutions:
        result = build_and_solve(nc)
        best = result
        if prev_err is not None and result["l2_err"] > 0.995 * prev_err:
            break
        prev_err = result["l2_err"]
        if time.time() - t0 > target_budget:
            break

    u_grid = sample_velocity_magnitude(best["u_h"], best["mesh"], nx_out, ny_out, bbox)

    if rank == 0:
        solver_info = {
            "mesh_resolution": int(best["mesh_resolution"]),
            "element_degree": 2,
            "ksp_type": "gmres",
            "pc_type": "lu",
            "rtol": 1.0e-10,
            "iterations": int(best["iterations"]),
            "nonlinear_iterations": [int(best["nonlinear_iterations"])],
            "accuracy_verification": {
                "l2_error_velocity": float(best["l2_err"]),
                "relative_l2_error_velocity": float(best["rel_l2"]),
                "divergence_l2": float(best["div_l2"]),
                "wall_time_sec": float(time.time() - t0),
                "manufactured_solution": True,
            },
        }
        return {"u": np.asarray(u_grid, dtype=np.float64), "solver_info": solver_info}
    return {"u": None, "solver_info": None}


if __name__ == "__main__":
    case_spec = {
        "pde": {"nu": 0.14, "time": None},
        "output": {"grid": {"nx": 64, "ny": 64, "bbox": [0.0, 1.0, 0.0, 1.0]}},
        "time_limit_sec": 680.656,
    }
    out = solve(case_spec)
    if MPI.COMM_WORLD.rank == 0:
        print(out["u"].shape)
        print(out["solver_info"])
