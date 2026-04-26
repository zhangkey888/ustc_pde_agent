import time
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl

ScalarType = PETSc.ScalarType


def _sample_function_on_grid(domain, uh, grid_spec):
    nx = int(grid_spec["nx"])
    ny = int(grid_spec["ny"])
    xmin, xmax, ymin, ymax = grid_spec["bbox"]

    xs = np.linspace(xmin, xmax, nx, dtype=np.float64)
    ys = np.linspace(ymin, ymax, ny, dtype=np.float64)
    XX, YY = np.meshgrid(xs, ys, indexing="xy")
    pts = np.zeros((nx * ny, 3), dtype=np.float64)
    pts[:, 0] = XX.ravel()
    pts[:, 1] = YY.ravel()

    tree = geometry.bb_tree(domain, domain.topology.dim)
    candidates = geometry.compute_collisions_points(tree, pts)
    colliding = geometry.compute_colliding_cells(domain, candidates, pts)

    local_vals = np.full(nx * ny, np.nan, dtype=np.float64)
    probe_pts = []
    probe_cells = []
    ids = []
    for i in range(nx * ny):
        links = colliding.links(i)
        if len(links) > 0:
            probe_pts.append(pts[i])
            probe_cells.append(links[0])
            ids.append(i)

    if ids:
        values = uh.eval(np.array(probe_pts, dtype=np.float64),
                         np.array(probe_cells, dtype=np.int32))
        values = np.asarray(values).reshape(len(ids), -1)[:, 0]
        local_vals[np.array(ids, dtype=np.int32)] = values

    gathered = domain.comm.gather(local_vals, root=0)
    if domain.comm.rank == 0:
        global_vals = np.full(nx * ny, np.nan, dtype=np.float64)
        for arr in gathered:
            mask = ~np.isnan(arr)
            global_vals[mask] = arr[mask]

        miss = np.isnan(global_vals)
        if np.any(miss):
            px = pts[miss, 0]
            py = pts[miss, 1]
            bmask = (
                np.isclose(px, xmin) | np.isclose(px, xmax) |
                np.isclose(py, ymin) | np.isclose(py, ymax)
            )
            tmp = global_vals[miss]
            tmp[bmask] = 0.0
            global_vals[miss] = tmp

        global_vals = np.nan_to_num(global_vals, nan=0.0)
        return global_vals.reshape(ny, nx)
    return None


def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD
    rank = comm.rank
    t0 = time.perf_counter()

    time_limit = 3.959
    candidates = [(48, 2), (64, 2), (80, 2)]
    best = None

    for n, degree in candidates:
        domain = mesh.create_rectangle(
            comm,
            [np.array([0.0, 0.0], dtype=np.float64), np.array([1.0, 1.0], dtype=np.float64)],
            [n, n],
            cell_type=mesh.CellType.quadrilateral,
        )
        V = fem.functionspace(domain, ("Lagrange", degree))

        u = ufl.TrialFunction(V)
        v = ufl.TestFunction(V)
        x = ufl.SpatialCoordinate(domain)

        f_expr = (
            ufl.sin(6.0 * ufl.pi * x[0]) * ufl.sin(5.0 * ufl.pi * x[1])
            + 0.4 * ufl.sin(11.0 * ufl.pi * x[0]) * ufl.sin(9.0 * ufl.pi * x[1])
        )

        a = fem.form(ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx)
        L = fem.form(f_expr * v * ufl.dx)

        fdim = domain.topology.dim - 1
        facets = mesh.locate_entities_boundary(
            domain, fdim, lambda X: np.ones(X.shape[1], dtype=bool)
        )
        dofs = fem.locate_dofs_topological(V, fdim, facets)
        bc = fem.dirichletbc(ScalarType(0.0), dofs, V)

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

        ksp = PETSc.KSP().create(comm)
        ksp.setOperators(A)
        ksp.setType("cg")
        pc = ksp.getPC()
        pc.setType("hypre")
        try:
            pc.setHYPREType("boomeramg")
        except Exception:
            pass
        rtol = 1e-10
        ksp.setTolerances(rtol=rtol, atol=1e-14, max_it=5000)

        try:
            ksp.solve(b, uh.x.petsc_vec)
            if ksp.getConvergedReason() <= 0:
                raise RuntimeError("CG failed")
        except Exception:
            ksp.destroy()
            ksp = PETSc.KSP().create(comm)
            ksp.setOperators(A)
            ksp.setType("preonly")
            ksp.getPC().setType("lu")
            rtol = 1e-12
            ksp.setTolerances(rtol=rtol, atol=1e-14, max_it=1)
            ksp.solve(b, uh.x.petsc_vec)

        uh.x.scatter_forward()

        u_exact = (
            ufl.sin(6.0 * ufl.pi * x[0]) * ufl.sin(5.0 * ufl.pi * x[1]) /
            (((6.0 * ufl.pi) ** 2) + ((5.0 * ufl.pi) ** 2))
            + 0.4 * ufl.sin(11.0 * ufl.pi * x[0]) * ufl.sin(9.0 * ufl.pi * x[1]) /
            (((11.0 * ufl.pi) ** 2) + ((9.0 * ufl.pi) ** 2))
        )
        err_sq_local = fem.assemble_scalar(fem.form((uh - u_exact) ** 2 * ufl.dx))
        ex_sq_local = fem.assemble_scalar(fem.form((u_exact) ** 2 * ufl.dx))
        err_sq = comm.allreduce(err_sq_local, op=MPI.SUM)
        ex_sq = comm.allreduce(ex_sq_local, op=MPI.SUM)
        rel_l2 = float(np.sqrt(err_sq / ex_sq)) if ex_sq > 0 else float(np.sqrt(err_sq))

        elapsed = time.perf_counter() - t0
        best = {
            "domain": domain,
            "uh": uh,
            "mesh_resolution": n,
            "element_degree": degree,
            "ksp_type": ksp.getType(),
            "pc_type": ksp.getPC().getType(),
            "rtol": rtol,
            "iterations": int(max(ksp.getIterationNumber(), 1 if ksp.getType() == "preonly" else 0)),
            "rel_l2_error": rel_l2,
        }

        if elapsed > 0.85 * time_limit:
            break

    u_grid = _sample_function_on_grid(best["domain"], best["uh"], case_spec["output"]["grid"])
    solver_info = {
        "mesh_resolution": int(best["mesh_resolution"]),
        "element_degree": int(best["element_degree"]),
        "ksp_type": str(best["ksp_type"]),
        "pc_type": str(best["pc_type"]),
        "rtol": float(best["rtol"]),
        "iterations": int(best["iterations"]),
        "rel_L2_error": float(best["rel_l2_error"]),
    }

    if rank == 0:
        return {"u": np.asarray(u_grid, dtype=np.float64), "solver_info": solver_info}
    return {"u": None, "solver_info": solver_info}
