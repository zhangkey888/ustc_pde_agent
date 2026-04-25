import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc


def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD

    # Grid output spec
    grid = case_spec["output"]["grid"]
    nx_out = grid["nx"]
    ny_out = grid["ny"]
    bbox = grid["bbox"]

    # Time params
    t0 = 0.0
    t_end = 0.1
    dt_val = 0.0025
    n_steps = int(round((t_end - t0) / dt_val))

    # Mesh
    N = 96
    domain = mesh.create_rectangle(
        comm,
        [np.array([bbox[0], bbox[2]]), np.array([bbox[1], bbox[3]])],
        [N, N],
        cell_type=mesh.CellType.quadrilateral,
    )

    degree = 2
    V = fem.functionspace(domain, ("Lagrange", degree))

    x = ufl.SpatialCoordinate(domain)
    t_const = fem.Constant(domain, PETSc.ScalarType(0.0))
    kappa = 1.0

    # Exact solution expression
    u_exact = ufl.exp(-t_const) * ufl.sin(4 * ufl.pi * x[0]) * ufl.sin(4 * ufl.pi * x[1])
    # f = du/dt - kappa * laplace(u)
    # du/dt = -exp(-t)*sin*sin
    # laplace = -2*(4pi)^2 * exp(-t)*sin*sin
    # f = -exp(-t)*s*s - kappa*(-32*pi^2)*exp(-t)*s*s = exp(-t)*s*s * (32*pi^2 - 1)
    f_expr = ufl.exp(-t_const) * ufl.sin(4 * ufl.pi * x[0]) * ufl.sin(4 * ufl.pi * x[1]) * (32 * ufl.pi**2 - 1)

    # BC function
    u_bc = fem.Function(V)
    bc_expr = fem.Expression(u_exact, V.element.interpolation_points)

    fdim = domain.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
    )
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc = fem.dirichletbc(u_bc, boundary_dofs)

    # Initial condition
    u_n = fem.Function(V)
    t_const.value = 0.0
    ic_expr = fem.Expression(u_exact, V.element.interpolation_points)
    u_n.interpolate(ic_expr)

    # Store initial condition for output
    u_initial_func = fem.Function(V)
    u_initial_func.x.array[:] = u_n.x.array

    u_sol = fem.Function(V)
    u_sol.x.array[:] = u_n.x.array

    # Backward Euler: (u - u_n)/dt - kappa*laplace(u) = f
    # => u - dt*kappa*laplace(u) = u_n + dt*f
    # Weak: (u,v) + dt*kappa*(grad u, grad v) = (u_n, v) + dt*(f, v)
    u_tr = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    dt_c = fem.Constant(domain, PETSc.ScalarType(dt_val))

    a = u_tr * v * ufl.dx + dt_c * kappa * ufl.inner(ufl.grad(u_tr), ufl.grad(v)) * ufl.dx
    L = u_n * v * ufl.dx + dt_c * f_expr * v * ufl.dx

    a_form = fem.form(a)
    L_form = fem.form(L)

    A = petsc.assemble_matrix(a_form, bcs=[bc])
    A.assemble()
    b = petsc.create_vector(L_form.function_spaces)

    solver = PETSc.KSP().create(comm)
    solver.setOperators(A)
    solver.setType(PETSc.KSP.Type.CG)
    solver.getPC().setType(PETSc.PC.Type.HYPRE)
    solver.setTolerances(rtol=1e-10, atol=1e-12, max_it=1000)

    total_iters = 0
    t_cur = t0
    for step in range(n_steps):
        t_cur += dt_val
        t_const.value = t_cur
        # Update BC
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

        u_n.x.array[:] = u_sol.x.array

    # Sample on uniform grid
    xs = np.linspace(bbox[0], bbox[1], nx_out)
    ys = np.linspace(bbox[2], bbox[3], ny_out)
    XX, YY = np.meshgrid(xs, ys)
    pts = np.c_[XX.ravel(), YY.ravel(), np.zeros(nx_out * ny_out)]

    tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(tree, pts)
    colliding = geometry.compute_colliding_cells(domain, cell_candidates, pts)

    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    for i in range(pts.shape[0]):
        links = colliding.links(i)
        if len(links) > 0:
            points_on_proc.append(pts[i])
            cells_on_proc.append(links[0])
            eval_map.append(i)

    u_vals = np.zeros(pts.shape[0])
    if len(points_on_proc) > 0:
        vals = u_sol.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        for k, idx in enumerate(eval_map):
            u_vals[idx] = vals[k, 0] if vals.ndim > 1 else vals[k]

    u_grid = u_vals.reshape(ny_out, nx_out)

    # Initial
    u_init_vals = np.zeros(pts.shape[0])
    if len(points_on_proc) > 0:
        vals_i = u_initial_func.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        for k, idx in enumerate(eval_map):
            u_init_vals[idx] = vals_i[k, 0] if vals_i.ndim > 1 else vals_i[k]
    u_initial_grid = u_init_vals.reshape(ny_out, nx_out)

    # Compute error vs exact
    u_exact_grid = np.exp(-t_end) * np.sin(4*np.pi*XX) * np.sin(4*np.pi*YY)
    err = np.sqrt(np.mean((u_grid - u_exact_grid)**2))
    print(f"RMS error: {err:.3e}, total KSP iters: {total_iters}")

    return {
        "u": u_grid,
        "u_initial": u_initial_grid,
        "solver_info": {
            "mesh_resolution": N,
            "element_degree": degree,
            "ksp_type": "cg",
            "pc_type": "hypre",
            "rtol": 1e-10,
            "iterations": total_iters,
            "dt": dt_val,
            "n_steps": n_steps,
            "time_scheme": "backward_euler",
        },
    }


if __name__ == "__main__":
    import time
    case_spec = {
        "output": {"grid": {"nx": 64, "ny": 64, "bbox": [0.0, 1.0, 0.0, 1.0]}},
        "pde": {"time": {"t0": 0.0, "t_end": 0.1, "dt": 0.005}},
    }
    t0 = time.time()
    out = solve(case_spec)
    print(f"Wall time: {time.time()-t0:.2f}s")
    print(f"Output shape: {out['u'].shape}")
