import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc

ScalarType = PETSc.ScalarType


def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD

    # Params
    t0 = 0.0
    t_end = 0.08
    dt_val = 0.0005
    kappa_val = 1.0

    # Output grid
    grid = case_spec["output"]["grid"]
    nx_out = grid["nx"]
    ny_out = grid["ny"]
    bbox = grid["bbox"]

    # Mesh - coarser to avoid timeout
    N = 40
    degree = 3
    domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", degree))

    x = ufl.SpatialCoordinate(domain)
    t_const = fem.Constant(domain, ScalarType(0.0))
    kappa = fem.Constant(domain, ScalarType(kappa_val))
    dt_c = fem.Constant(domain, ScalarType(dt_val))

    # Exact solution & source
    u_exact_expr = ufl.exp(-t_const) * ufl.sin(3 * ufl.pi * x[0]) * ufl.sin(3 * ufl.pi * x[1])
    sin_term = ufl.sin(3 * ufl.pi * x[0]) * ufl.sin(3 * ufl.pi * x[1])
    f_expr = (-1.0 + 2.0 * (3.0 * ufl.pi) ** 2 * kappa_val) * ufl.exp(-t_const) * sin_term

    # Initial condition
    u_n = fem.Function(V)
    u_init_expr = ufl.sin(3 * ufl.pi * x[0]) * ufl.sin(3 * ufl.pi * x[1])
    u_n.interpolate(fem.Expression(u_init_expr, V.element.interpolation_points))

    # Boundary condition
    fdim = domain.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
    )
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    u_bc = fem.Function(V)
    u_bc_expr = ufl.exp(-t_const) * sin_term
    u_bc_expression = fem.Expression(u_bc_expr, V.element.interpolation_points)
    u_bc.interpolate(u_bc_expression)
    bc = fem.dirichletbc(u_bc, boundary_dofs)

    # Variational form
    u_tr = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    a = u_tr * v * ufl.dx + dt_c * kappa * ufl.inner(ufl.grad(u_tr), ufl.grad(v)) * ufl.dx
    L = (u_n + dt_c * f_expr) * v * ufl.dx

    a_form = fem.form(a)
    L_form = fem.form(L)

    A = petsc.assemble_matrix(a_form, bcs=[bc])
    A.assemble()
    b = petsc.create_vector(L_form.function_spaces)

    # LU solver for reliability
    solver = PETSc.KSP().create(domain.comm)
    solver.setOperators(A)
    solver.setType(PETSc.KSP.Type.PREONLY)
    solver.getPC().setType(PETSc.PC.Type.LU)

    u_sol = fem.Function(V)
    n_steps = int(round((t_end - t0) / dt_val))
    total_iters = n_steps  # LU counts as 1

    t_cur = t0
    for step in range(n_steps):
        t_cur += dt_val
        t_const.value = t_cur
        u_bc.interpolate(u_bc_expression)

        with b.localForm() as loc:
            loc.set(0)
        petsc.assemble_vector(b, L_form)
        petsc.apply_lifting(b, [a_form], bcs=[[bc]])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        petsc.set_bc(b, [bc])

        solver.solve(b, u_sol.x.petsc_vec)
        u_sol.x.scatter_forward()
        u_n.x.array[:] = u_sol.x.array

    # Sample on grid
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

    u_vals = np.full(pts.shape[0], np.nan)
    if len(points_on_proc) > 0:
        vals = u_sol.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u_vals[eval_map] = vals.flatten()
    u_grid = u_vals.reshape(ny_out, nx_out)

    # Accuracy check
    u_exact_grid = np.exp(-t_cur) * np.sin(3 * np.pi * XX) * np.sin(3 * np.pi * YY)
    err_l2 = np.sqrt(np.mean((u_grid - u_exact_grid) ** 2))
    err_max = np.max(np.abs(u_grid - u_exact_grid))
    print(f"L2 err: {err_l2:.3e}, max err: {err_max:.3e}")

    u_init_grid = np.sin(3 * np.pi * XX) * np.sin(3 * np.pi * YY)

    return {
        "u": u_grid,
        "u_initial": u_init_grid,
        "solver_info": {
            "mesh_resolution": N,
            "element_degree": degree,
            "ksp_type": "preonly",
            "pc_type": "lu",
            "rtol": 0.0,
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
        "pde": {"time": True},
    }
    t0 = time.time()
    res = solve(case_spec)
    elapsed = time.time() - t0
    print(f"Elapsed: {elapsed:.2f}s")
    print(f"solver_info: {res['solver_info']}")
