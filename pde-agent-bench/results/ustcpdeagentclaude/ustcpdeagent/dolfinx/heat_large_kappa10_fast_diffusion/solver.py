import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc


def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD
    kappa_val = 10.0
    t0 = 0.0
    t_end = 0.05
    # Use finer dt for better BE accuracy
    n_steps = 50
    dt_val = (t_end - t0) / n_steps

    N = 96
    degree = 2
    domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", degree))

    x = ufl.SpatialCoordinate(domain)
    t_const = fem.Constant(domain, PETSc.ScalarType(0.0))
    kappa = fem.Constant(domain, PETSc.ScalarType(kappa_val))
    dt_c = fem.Constant(domain, PETSc.ScalarType(dt_val))

    u_exact_expr = ufl.exp(-t_const) * ufl.sin(ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])
    f_expr = (-1.0 + kappa_val * 2.0 * ufl.pi**2) * ufl.exp(-t_const) * ufl.sin(ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    u_n = fem.Function(V)

    t_const.value = 0.0
    u_init_expr = fem.Expression(u_exact_expr, V.element.interpolation_points)
    u_n.interpolate(u_init_expr)

    a = (u * v + dt_c * kappa * ufl.inner(ufl.grad(u), ufl.grad(v))) * ufl.dx
    L = (u_n + dt_c * f_expr) * v * ufl.dx

    u_bc = fem.Function(V)
    fdim = domain.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(domain, fdim, lambda xx: np.ones(xx.shape[1], dtype=bool))
    bdofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc = fem.dirichletbc(u_bc, bdofs)

    bc_expr = fem.Expression(u_exact_expr, V.element.interpolation_points)

    a_form = fem.form(a)
    L_form = fem.form(L)
    A = petsc.assemble_matrix(a_form, bcs=[bc])
    A.assemble()

    b = petsc.create_vector(L_form.function_spaces)

    solver = PETSc.KSP().create(comm)
    solver.setOperators(A)
    solver.setType("cg")
    solver.getPC().setType("hypre")
    solver.setTolerances(rtol=1e-10, atol=1e-12)

    u_sol = fem.Function(V)
    total_iters = 0

    u_initial_func = fem.Function(V)
    u_initial_func.x.array[:] = u_n.x.array

    t = t0
    for step in range(n_steps):
        t_new = t + dt_val
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

        u_n.x.array[:] = u_sol.x.array
        t = t_new

    grid = case_spec["output"]["grid"]
    nx = grid["nx"]
    ny = grid["ny"]
    bbox = grid["bbox"]
    xs = np.linspace(bbox[0], bbox[1], nx)
    ys = np.linspace(bbox[2], bbox[3], ny)
    XX, YY = np.meshgrid(xs, ys)
    pts = np.c_[XX.ravel(), YY.ravel(), np.zeros(nx * ny)]

    tree = geometry.bb_tree(domain, domain.topology.dim)
    cand = geometry.compute_collisions_points(tree, pts)
    coll = geometry.compute_colliding_cells(domain, cand, pts)

    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    for i in range(pts.shape[0]):
        links = coll.links(i)
        if len(links) > 0:
            points_on_proc.append(pts[i])
            cells_on_proc.append(links[0])
            eval_map.append(i)

    u_flat = np.full(nx * ny, np.nan)
    if len(points_on_proc) > 0:
        vals = u_sol.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u_flat[eval_map] = vals.flatten()
    u_grid = u_flat.reshape(ny, nx)

    u_init_flat = np.full(nx * ny, np.nan)
    if len(points_on_proc) > 0:
        vals0 = u_initial_func.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u_init_flat[eval_map] = vals0.flatten()
    u_init_grid = u_init_flat.reshape(ny, nx)

    return {
        "u": u_grid,
        "u_initial": u_init_grid,
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
        },
    }


if __name__ == "__main__":
    import time
    case_spec = {
        "output": {"grid": {"nx": 64, "ny": 64, "bbox": [0.0, 1.0, 0.0, 1.0]}}
    }
    t0 = time.time()
    res = solve(case_spec)
    elapsed = time.time() - t0
    u = res["u"]
    nx, ny = 64, 64
    xs = np.linspace(0, 1, nx)
    ys = np.linspace(0, 1, ny)
    XX, YY = np.meshgrid(xs, ys)
    u_exact = np.exp(-0.05) * np.sin(np.pi * XX) * np.sin(np.pi * YY)
    err = np.sqrt(np.mean((u - u_exact) ** 2))
    linf = np.max(np.abs(u - u_exact))
    print(f"time={elapsed:.3f}s, RMSE={err:.3e}, Linf={linf:.3e}")
    print(res["solver_info"])
