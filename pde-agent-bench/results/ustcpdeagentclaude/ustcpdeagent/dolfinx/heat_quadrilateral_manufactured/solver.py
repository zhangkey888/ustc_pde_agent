import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc


def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD
    ScalarType = PETSc.ScalarType

    # Parameters
    t0 = 0.0
    t_end = 0.1
    dt_val = 0.005
    n_steps = int(round((t_end - t0) / dt_val))
    kappa = 1.0

    N = 80
    degree = 2

    domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.quadrilateral)
    V = fem.functionspace(domain, ("Lagrange", degree))

    x = ufl.SpatialCoordinate(domain)
    t_const = fem.Constant(domain, ScalarType(t0))

    # Exact solution and source
    u_exact_expr = ufl.exp(-t_const) * ufl.sin(ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])
    # f = du/dt - kappa*lap(u) = -exp(-t)sin*sin + kappa*2*pi^2*exp(-t)sin*sin
    f_expr = (-1.0 + kappa * 2.0 * ufl.pi**2) * ufl.exp(-t_const) * ufl.sin(ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])

    # Initial condition
    u_n = fem.Function(V)
    t_const.value = t0
    u_init_expr = fem.Expression(u_exact_expr, V.element.interpolation_points)
    u_n.interpolate(u_init_expr)

    # BC function (will update over time)
    u_bc = fem.Function(V)
    u_bc.interpolate(u_init_expr)

    fdim = domain.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool))
    bdofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc = fem.dirichletbc(u_bc, bdofs)

    # Variational form - backward Euler
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    dt_c = fem.Constant(domain, ScalarType(dt_val))
    kappa_c = fem.Constant(domain, ScalarType(kappa))

    a = u * v * ufl.dx + dt_c * kappa_c * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = (u_n + dt_c * f_expr) * v * ufl.dx

    a_form = fem.form(a)
    L_form = fem.form(L)

    A = petsc.assemble_matrix(a_form, bcs=[bc])
    A.assemble()
    b = petsc.create_vector(L_form.function_spaces)

    solver = PETSc.KSP().create(comm)
    solver.setOperators(A)
    solver.setType(PETSc.KSP.Type.CG)
    solver.getPC().setType(PETSc.PC.Type.HYPRE)
    rtol = 1e-10
    solver.setTolerances(rtol=rtol, atol=1e-14, max_it=1000)

    uh = fem.Function(V)
    total_iters = 0

    t_current = t0
    for step in range(n_steps):
        t_current += dt_val
        t_const.value = t_current
        # Update BC
        u_bc.interpolate(fem.Expression(u_exact_expr, V.element.interpolation_points))

        # Assemble RHS
        with b.localForm() as loc:
            loc.set(0)
        petsc.assemble_vector(b, L_form)
        petsc.apply_lifting(b, [a_form], bcs=[[bc]])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        petsc.set_bc(b, [bc])

        solver.solve(b, uh.x.petsc_vec)
        uh.x.scatter_forward()
        total_iters += solver.getIterationNumber()

        u_n.x.array[:] = uh.x.array

    # Sample on output grid
    out_grid = case_spec["output"]["grid"]
    nx_o = out_grid["nx"]
    ny_o = out_grid["ny"]
    bbox = out_grid["bbox"]

    xs = np.linspace(bbox[0], bbox[1], nx_o)
    ys = np.linspace(bbox[2], bbox[3], ny_o)
    XX, YY = np.meshgrid(xs, ys)
    pts = np.c_[XX.ravel(), YY.ravel(), np.zeros(nx_o * ny_o)]

    tree = geometry.bb_tree(domain, domain.topology.dim)
    cand = geometry.compute_collisions_points(tree, pts)
    colliding = geometry.compute_colliding_cells(domain, cand, pts)

    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    for i in range(pts.shape[0]):
        links = colliding.links(i)
        if len(links) > 0:
            points_on_proc.append(pts[i])
            cells_on_proc.append(links[0])
            eval_map.append(i)

    u_grid = np.full(nx_o * ny_o, np.nan)
    if len(points_on_proc) > 0:
        vals = uh.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u_grid[eval_map] = vals.flatten()
    u_grid = u_grid.reshape(ny_o, nx_o)

    # Initial condition on grid
    u_n0 = fem.Function(V)
    t_const.value = t0
    u_n0.interpolate(fem.Expression(u_exact_expr, V.element.interpolation_points))
    u_init_grid = np.full(nx_o * ny_o, np.nan)
    if len(points_on_proc) > 0:
        vals0 = u_n0.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u_init_grid[eval_map] = vals0.flatten()
    u_init_grid = u_init_grid.reshape(ny_o, nx_o)

    return {
        "u": u_grid,
        "u_initial": u_init_grid,
        "solver_info": {
            "mesh_resolution": N,
            "element_degree": degree,
            "ksp_type": "cg",
            "pc_type": "hypre",
            "rtol": rtol,
            "iterations": total_iters,
            "dt": dt_val,
            "n_steps": n_steps,
            "time_scheme": "backward_euler",
        },
    }


if __name__ == "__main__":
    import time
    case_spec = {
        "pde": {"time": {"t0": 0.0, "t_end": 0.1, "dt": 0.01}},
        "output": {"grid": {"nx": 64, "ny": 64, "bbox": [0.0, 1.0, 0.0, 1.0]}},
    }
    t0 = time.time()
    result = solve(case_spec)
    t1 = time.time()
    print(f"Time: {t1-t0:.3f}s")

    # Compute error against exact
    u_grid = result["u"]
    nx_o = 64; ny_o = 64
    xs = np.linspace(0, 1, nx_o)
    ys = np.linspace(0, 1, ny_o)
    XX, YY = np.meshgrid(xs, ys)
    u_ex = np.exp(-0.1) * np.sin(np.pi * XX) * np.sin(np.pi * YY)
    err = np.sqrt(np.mean((u_grid - u_ex) ** 2))
    err_max = np.max(np.abs(u_grid - u_ex))
    print(f"RMSE: {err:.3e}, Max: {err_max:.3e}")
    print(f"Iters: {result['solver_info']['iterations']}")
