import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc


def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD

    # Parameters
    t0 = 0.0
    t_end = 0.1
    dt_val = 0.005
    n_steps = int(round((t_end - t0) / dt_val))

    # Mesh
    N = 96
    degree = 2
    domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", degree))

    x = ufl.SpatialCoordinate(domain)
    t_const = fem.Constant(domain, PETSc.ScalarType(t0))
    dt_const = fem.Constant(domain, PETSc.ScalarType(dt_val))

    # Exact solution: u = exp(-t) * sin(pi*x) * sin(2*pi*y)
    pi = ufl.pi
    u_exact_expr = ufl.exp(-t_const) * ufl.sin(pi * x[0]) * ufl.sin(2 * pi * x[1])

    # kappa
    kappa = 1 + 30 * ufl.exp(-150 * ((x[0] - 0.35) ** 2 + (x[1] - 0.65) ** 2))

    # f = du/dt - div(kappa * grad(u))
    du_dt = -u_exact_expr  # d/dt exp(-t)*...
    flux = kappa * ufl.grad(u_exact_expr)
    f_expr = du_dt - ufl.div(flux)

    # Initial condition
    u_n = fem.Function(V)
    t_const.value = t0
    u0_expr = fem.Expression(u_exact_expr, V.element.interpolation_points)
    u_n.interpolate(u0_expr)

    # Initial grid output
    out = case_spec["output"]["grid"]
    nx_out, ny_out = out["nx"], out["ny"]
    bbox = out["bbox"]

    def sample_on_grid(u_func):
        xs = np.linspace(bbox[0], bbox[1], nx_out)
        ys = np.linspace(bbox[2], bbox[3], ny_out)
        XX, YY = np.meshgrid(xs, ys)
        pts = np.c_[XX.ravel(), YY.ravel(), np.zeros(nx_out * ny_out)]
        tree = geometry.bb_tree(domain, domain.topology.dim)
        cand = geometry.compute_collisions_points(tree, pts)
        coll = geometry.compute_colliding_cells(domain, cand, pts)
        cells = []
        pts_ok = []
        idx_map = []
        for i in range(pts.shape[0]):
            links = coll.links(i)
            if len(links) > 0:
                pts_ok.append(pts[i])
                cells.append(links[0])
                idx_map.append(i)
        vals = np.full(pts.shape[0], np.nan)
        if pts_ok:
            v = u_func.eval(np.array(pts_ok), np.array(cells, dtype=np.int32))
            vals[idx_map] = v.flatten()
        return vals.reshape(ny_out, nx_out)

    u_initial_grid = sample_on_grid(u_n)

    # Trial/Test
    u_tr = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    # Backward Euler: (u - u_n)/dt - div(kappa grad u) = f
    # Weak: (u,v)/dt + (kappa grad u, grad v) = (u_n, v)/dt + (f, v)
    a = (u_tr * v / dt_const) * ufl.dx + ufl.inner(kappa * ufl.grad(u_tr), ufl.grad(v)) * ufl.dx
    L = (u_n * v / dt_const) * ufl.dx + f_expr * v * ufl.dx

    # Dirichlet BC from exact solution
    fdim = domain.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
    )
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    u_bc = fem.Function(V)
    bc_expr = fem.Expression(u_exact_expr, V.element.interpolation_points)
    u_bc.interpolate(bc_expr)
    bc = fem.dirichletbc(u_bc, boundary_dofs)

    a_form = fem.form(a)
    L_form = fem.form(L)

    A = petsc.assemble_matrix(a_form, bcs=[bc])
    A.assemble()

    b = petsc.create_vector(L_form.function_spaces)

    solver = PETSc.KSP().create(comm)
    solver.setOperators(A)
    solver.setType("cg")
    pc = solver.getPC()
    pc.setType("hypre")
    pc.setHYPREType("boomeramg")
    solver.setTolerances(rtol=1e-10, atol=1e-12, max_it=500)

    u_sol = fem.Function(V)

    total_iters = 0
    t_cur = t0
    for step in range(n_steps):
        t_cur += dt_val
        t_const.value = t_cur

        # Update BC
        u_bc.interpolate(bc_expr)

        # Assemble RHS
        with b.localForm() as loc:
            loc.set(0)
        petsc.assemble_vector(b, L_form)
        petsc.apply_lifting(b, [a_form], bcs=[[bc]])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        petsc.set_bc(b, [bc])

        solver.solve(b, u_sol.x.petsc_vec)
        u_sol.x.scatter_forward()
        total_iters += solver.getIterationNumber()

        u_n.x.array[:] = u_sol.x.array

    u_grid = sample_on_grid(u_n)

    # Error check
    t_const.value = t_end
    err_form = fem.form((u_n - u_exact_expr) ** 2 * ufl.dx)
    err_l2 = np.sqrt(comm.allreduce(fem.assemble_scalar(err_form), op=MPI.SUM))

    return {
        "u": u_grid,
        "u_initial": u_initial_grid,
        "solver_info": {
            "mesh_resolution": N,
            "element_degree": degree,
            "ksp_type": "cg",
            "pc_type": "hypre",
            "rtol": 1e-10,
            "iterations": int(total_iters),
            "dt": dt_val,
            "n_steps": n_steps,
            "time_scheme": "backward_euler",
            "l2_error": float(err_l2),
        },
    }


if __name__ == "__main__":
    import time
    spec = {
        "output": {"grid": {"nx": 128, "ny": 128, "bbox": [0, 1, 0, 1]}},
        "pde": {"time": {"t0": 0, "t_end": 0.1}},
    }
    t0 = time.time()
    res = solve(spec)
    dt = time.time() - t0
    print(f"Time: {dt:.2f}s")
    print("Solver info:", res["solver_info"])

    # Compare to exact
    nx_o, ny_o = 128, 128
    xs = np.linspace(0, 1, nx_o)
    ys = np.linspace(0, 1, ny_o)
    XX, YY = np.meshgrid(xs, ys)
    u_ex = np.exp(-0.1) * np.sin(np.pi * XX) * np.sin(2 * np.pi * YY)
    err = np.max(np.abs(res["u"] - u_ex))
    print(f"Max grid error at t_end: {err:.3e}")
    rms = np.sqrt(np.mean((res["u"] - u_ex) ** 2))
    print(f"RMS grid error: {rms:.3e}")
