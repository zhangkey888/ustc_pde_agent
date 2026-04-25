import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc


def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD

    # ---- Parse parameters ----
    pde_cfg = case_spec.get("pde", {}) or {}
    params = case_spec.get("parameters", {}) or {}
    epsilon = float(params.get("epsilon", pde_cfg.get("epsilon", 1.0)))

    time_info = pde_cfg.get("time", None) or case_spec.get("time", None)
    t0 = 0.0
    t_end = 0.3
    dt_suggested = 0.0005
    if time_info is not None:
        t0 = float(time_info.get("t0", t0))
        t_end = float(time_info.get("t_end", t_end))
        dt_suggested = float(time_info.get("dt", dt_suggested))

    # Output grid
    grid = case_spec["output"]["grid"]
    nx_out = int(grid["nx"])
    ny_out = int(grid["ny"])
    bbox = grid["bbox"]

    # ---- Discretization ----
    mesh_resolution = 96
    element_degree = 2

    dt_val = min(dt_suggested, 0.0002)
    n_steps = max(1, int(round((t_end - t0) / dt_val)))
    dt_val = (t_end - t0) / n_steps

    domain = mesh.create_unit_square(comm, mesh_resolution, mesh_resolution,
                                      cell_type=mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", element_degree))

    x = ufl.SpatialCoordinate(domain)
    g_ufl = ufl.exp(4.0 * x[0]) * ufl.sin(ufl.pi * x[1])
    C_val = -1.0 - epsilon * (16.0 - np.pi ** 2)

    u_base = fem.Function(V)
    u_base.interpolate(fem.Expression(g_ufl, V.element.interpolation_points))
    base_arr = u_base.x.array.copy()

    u_n = fem.Function(V)
    u_n.x.array[:] = base_arr.copy()

    f_func = fem.Function(V)
    f_func.x.array[:] = C_val * base_arr.copy()

    # Boundary
    fdim = domain.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
    )
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    u_bc = fem.Function(V)
    u_bc.x.array[:] = base_arr.copy()
    bc = fem.dirichletbc(u_bc, boundary_dofs)

    dt_const = fem.Constant(domain, PETSc.ScalarType(dt_val))
    eps_const = fem.Constant(domain, PETSc.ScalarType(epsilon))

    u_trial = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    a = (u_trial * v / dt_const) * ufl.dx \
        + eps_const * ufl.inner(ufl.grad(u_trial), ufl.grad(v)) * ufl.dx
    L_form_ufl = (u_n * v / dt_const) * ufl.dx + f_func * v * ufl.dx

    a_form = fem.form(a)
    L_form = fem.form(L_form_ufl)

    # Assemble LHS matrix ONCE
    A = petsc.assemble_matrix(a_form, bcs=[bc])
    A.assemble()

    b = petsc.create_vector(L_form.function_spaces)

    # KSP: direct LU, factored once
    ksp = PETSc.KSP().create(comm)
    ksp.setOperators(A)
    ksp.setType(PETSc.KSP.Type.PREONLY)
    pc = ksp.getPC()
    pc.setType(PETSc.PC.Type.LU)
    ksp.setUp()  # trigger factorization now

    u_sol = fem.Function(V)
    u_sol.x.array[:] = u_n.x.array[:]

    def sample_on_grid(func):
        xs = np.linspace(bbox[0], bbox[1], nx_out)
        ys = np.linspace(bbox[2], bbox[3], ny_out)
        XX, YY = np.meshgrid(xs, ys)
        pts = np.c_[XX.ravel(), YY.ravel(), np.zeros(nx_out * ny_out)]
        tree = geometry.bb_tree(domain, domain.topology.dim)
        cand = geometry.compute_collisions_points(tree, pts)
        coll = geometry.compute_colliding_cells(domain, cand, pts)
        cells = []
        pts_proc = []
        idx_map = []
        for i in range(pts.shape[0]):
            links = coll.links(i)
            if len(links) > 0:
                pts_proc.append(pts[i])
                cells.append(links[0])
                idx_map.append(i)
        vals = np.full(pts.shape[0], np.nan)
        if len(pts_proc) > 0:
            res = func.eval(np.array(pts_proc), np.array(cells, dtype=np.int32))
            vals[idx_map] = res.flatten()
        return vals.reshape(ny_out, nx_out)

    u_initial_grid = sample_on_grid(u_n)

    total_iters = 0
    for step in range(n_steps):
        t_cur = t0 + (step + 1) * dt_val
        scale = np.exp(-t_cur)
        u_bc.x.array[:] = scale * base_arr
        f_func.x.array[:] = scale * C_val * base_arr

        # Assemble RHS
        with b.localForm() as loc:
            loc.set(0.0)
        petsc.assemble_vector(b, L_form)
        petsc.apply_lifting(b, [a_form], bcs=[[bc]])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        petsc.set_bc(b, [bc])

        ksp.solve(b, u_sol.x.petsc_vec)
        u_sol.x.scatter_forward()
        u_n.x.array[:] = u_sol.x.array[:]
        total_iters += ksp.getIterationNumber()

    u_grid = sample_on_grid(u_sol)

    xs = np.linspace(bbox[0], bbox[1], nx_out)
    ys = np.linspace(bbox[2], bbox[3], ny_out)
    XX, YY = np.meshgrid(xs, ys)
    u_ex = np.exp(-t_end) * np.exp(4.0 * XX) * np.sin(np.pi * YY)
    err = float(np.sqrt(np.mean((u_grid - u_ex) ** 2)))
    if comm.rank == 0:
        print(f"[solver] mesh={mesh_resolution} p={element_degree} "
              f"dt={dt_val} n_steps={n_steps} iters={total_iters} L2~{err:.3e}")

    return {
        "u": u_grid,
        "u_initial": u_initial_grid,
        "solver_info": {
            "mesh_resolution": int(mesh_resolution),
            "element_degree": int(element_degree),
            "ksp_type": "preonly",
            "pc_type": "lu",
            "rtol": 1e-12,
            "iterations": int(total_iters),
            "dt": float(dt_val),
            "n_steps": int(n_steps),
            "time_scheme": "backward_euler",
        },
    }


if __name__ == "__main__":
    import time
    case_spec = {
        "pde": {"epsilon": 1.0,
                 "time": {"t0": 0.0, "t_end": 0.3, "dt": 0.005, "scheme": "backward_euler"}},
        "output": {"grid": {"nx": 64, "ny": 64, "bbox": [0.0, 1.0, 0.0, 1.0]}},
    }
    t0 = time.time()
    out = solve(case_spec)
    print(f"wall time: {time.time() - t0:.2f}s")
    print("u shape:", out["u"].shape)
    print("info:", out["solver_info"])
