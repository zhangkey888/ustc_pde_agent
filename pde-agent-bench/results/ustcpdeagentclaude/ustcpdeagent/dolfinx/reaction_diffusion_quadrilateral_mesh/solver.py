import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc

ScalarType = PETSc.ScalarType


def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD

    grid = case_spec["output"]["grid"]
    nx_out = grid["nx"]
    ny_out = grid["ny"]
    bbox = grid["bbox"]

    t0 = 0.0
    t_end = 0.4
    dt = 0.005
    if "pde" in case_spec and "time" in case_spec["pde"]:
        tinfo = case_spec["pde"]["time"]
        t0 = tinfo.get("t0", t0)
        t_end = tinfo.get("t_end", t_end)
        dt = tinfo.get("dt", dt)
        dt = min(dt, 0.005)

    n_steps = int(round((t_end - t0) / dt))
    dt = (t_end - t0) / n_steps

    N = 64
    degree = 2
    domain = mesh.create_rectangle(
        comm, [np.array([0.0, 0.0]), np.array([1.0, 1.0])],
        [N, N], cell_type=mesh.CellType.quadrilateral
    )

    V = fem.functionspace(domain, ("Lagrange", degree))

    epsilon = 1.0
    t_const = fem.Constant(domain, ScalarType(t0))
    x = ufl.SpatialCoordinate(domain)

    u_ex = ufl.exp(-t_const) * ufl.exp(x[0]) * ufl.sin(ufl.pi * x[1])
    # R(u) = u; f = du/dt - eps*lap(u) + R(u)
    f_expr = -u_ex - epsilon * ufl.div(ufl.grad(u_ex)) + u_ex

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    u_n = fem.Function(V)
    init_expr = fem.Expression(u_ex, V.element.interpolation_points)
    u_n.interpolate(init_expr)

    u_initial_func = fem.Function(V)
    u_initial_func.x.array[:] = u_n.x.array[:]

    dt_const = fem.Constant(domain, ScalarType(dt))

    a = (u * v * ufl.dx
         + dt_const * epsilon * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
         + dt_const * u * v * ufl.dx)
    L = u_n * v * ufl.dx + dt_const * f_expr * v * ufl.dx

    fdim = domain.topology.dim - 1
    bfac = mesh.locate_entities_boundary(
        domain, fdim, lambda xx: np.ones(xx.shape[1], dtype=bool)
    )
    bdofs = fem.locate_dofs_topological(V, fdim, bfac)

    u_bc = fem.Function(V)
    bc_expr = fem.Expression(u_ex, V.element.interpolation_points)
    u_bc.interpolate(bc_expr)
    bc = fem.dirichletbc(u_bc, bdofs)

    a_form = fem.form(a)
    L_form = fem.form(L)

    A = petsc.assemble_matrix(a_form, bcs=[bc])
    A.assemble()
    b = petsc.create_vector(L_form.function_spaces)

    solver = PETSc.KSP().create(comm)
    solver.setOperators(A)
    solver.setType("cg")
    solver.getPC().setType("hypre")
    solver.setTolerances(rtol=1e-10, atol=1e-12, max_it=1000)

    u_sol = fem.Function(V)

    total_iters = 0
    t = t0
    for step in range(n_steps):
        t_new = t + dt
        t_const.value = t_new
        u_bc.interpolate(bc_expr)

        with b.localForm() as loc:
            loc.set(0)
        petsc.assemble_vector(b, L_form)
        petsc.apply_lifting(b, [a_form], bcs=[[bc]])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        petsc.set_bc(b, [bc])

        solver.solve(b, u_sol.x.petsc_vec)
        u_sol.x.scatter_forward()
        total_iters += solver.getIterationNumber()

        u_n.x.array[:] = u_sol.x.array[:]
        t = t_new

    # Sample onto output grid
    xs = np.linspace(bbox[0], bbox[1], nx_out)
    ys = np.linspace(bbox[2], bbox[3], ny_out)
    XX, YY = np.meshgrid(xs, ys)
    pts = np.c_[XX.ravel(), YY.ravel(), np.zeros(nx_out * ny_out)]

    tree = geometry.bb_tree(domain, domain.topology.dim)
    cc = geometry.compute_collisions_points(tree, pts)
    col = geometry.compute_colliding_cells(domain, cc, pts)

    pop = []; cop = []; em = []
    for i in range(pts.shape[0]):
        lk = col.links(i)
        if len(lk) > 0:
            pop.append(pts[i]); cop.append(lk[0]); em.append(i)

    u_vals = np.full(pts.shape[0], np.nan)
    u_init_vals = np.full(pts.shape[0], np.nan)
    if len(pop) > 0:
        vals = u_sol.eval(np.array(pop), np.array(cop, dtype=np.int32))
        u_vals[em] = vals.flatten()
        vi = u_initial_func.eval(np.array(pop), np.array(cop, dtype=np.int32))
        u_init_vals[em] = vi.flatten()

    u_grid = u_vals.reshape(ny_out, nx_out)
    u_init_grid = u_init_vals.reshape(ny_out, nx_out)

    u_ex_np = np.exp(-t_end) * np.exp(XX) * np.sin(np.pi * YY)
    err = np.sqrt(np.mean((u_grid - u_ex_np) ** 2))
    print(f"[solver] L2 err={err:.3e}, iters={total_iters}, dt={dt}, n_steps={n_steps}")

    return {
        "u": u_grid,
        "u_initial": u_init_grid,
        "solver_info": {
            "mesh_resolution": N,
            "element_degree": degree,
            "ksp_type": "cg",
            "pc_type": "hypre",
            "rtol": 1e-10,
            "iterations": total_iters,
            "dt": dt,
            "n_steps": n_steps,
            "time_scheme": "backward_euler",
        },
    }


if __name__ == "__main__":
    import time
    case_spec = {
        "pde": {"time": {"t0": 0.0, "t_end": 0.4, "dt": 0.01}},
        "output": {"grid": {"nx": 64, "ny": 64, "bbox": [0.0, 1.0, 0.0, 1.0]}},
    }
    t0 = time.time()
    out = solve(case_spec)
    print(f"Wall time: {time.time()-t0:.2f}s, shape={out['u'].shape}")
