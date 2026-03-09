import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc

ScalarType = PETSc.ScalarType


def solve(case_spec: dict = None) -> dict:
    t_end = 0.1
    dt_val = 0.01
    scheme = "backward_euler"

    if case_spec is not None:
        pde = case_spec.get("pde", {})
        time_info = pde.get("time", {})
        if time_info:
            t_end = time_info.get("t_end", t_end)
            dt_val = time_info.get("dt", dt_val)
            scheme = time_info.get("scheme", scheme)

    N = 64
    degree = 2
    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", degree))
    tdim = domain.topology.dim
    fdim = tdim - 1
    x = ufl.SpatialCoordinate(domain)
    t = fem.Constant(domain, ScalarType(0.0))
    dt_c = fem.Constant(domain, ScalarType(dt_val))
    pi = ufl.pi
    kappa = 1.0 + 0.3 * ufl.cos(2 * pi * x[0]) * ufl.cos(2 * pi * x[1])
    u_exact_ufl = ufl.exp(-t) * ufl.sin(2 * pi * x[0]) * ufl.sin(2 * pi * x[1])
    du_dt = -ufl.exp(-t) * ufl.sin(2 * pi * x[0]) * ufl.sin(2 * pi * x[1])
    grad_u_exact = ufl.grad(u_exact_ufl)
    f_expr = du_dt - ufl.div(kappa * grad_u_exact)
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    u_n = fem.Function(V, name="u_n")
    u_h = fem.Function(V, name="u_h")
    u_n.interpolate(fem.Expression(ufl.sin(2*pi*x[0])*ufl.sin(2*pi*x[1]), V.element.interpolation_points))
    u_initial_func = fem.Function(V)
    u_initial_func.x.array[:] = u_n.x.array[:]
    a = (u * v / dt_c) * ufl.dx + ufl.inner(kappa * ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = (u_n * v / dt_c) * ufl.dx + f_expr * v * ufl.dx
    boundary_facets = mesh.locate_entities_boundary(domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool))
    dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    u_bc = fem.Function(V)
    bc = fem.dirichletbc(u_bc, dofs)
    a_form = fem.form(a)
    L_form = fem.form(L)
    A = petsc.assemble_matrix(a_form, bcs=[bc])
    A.assemble()
    b = petsc.create_vector(V)
    solver = PETSc.KSP().create(domain.comm)
    solver.setOperators(A)
    solver.setType(PETSc.KSP.Type.CG)
    pc = solver.getPC()
    pc.setType(PETSc.PC.Type.HYPRE)
    solver.setTolerances(rtol=1e-10, atol=1e-12, max_it=2000)
    solver.setUp()
    n_steps = int(round(t_end / dt_val))
    current_t = 0.0
    total_iterations = 0
    for step in range(n_steps):
        current_t += dt_val
        t.value = current_t
        u_bc.interpolate(fem.Expression(u_exact_ufl, V.element.interpolation_points))
        with b.localForm() as loc:
            loc.set(0)
        petsc.assemble_vector(b, L_form)
        petsc.apply_lifting(b, [a_form], bcs=[[bc]])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        petsc.set_bc(b, [bc])
        solver.solve(b, u_h.x.petsc_vec)
        u_h.x.scatter_forward()
        total_iterations += solver.getIterationNumber()
        u_n.x.array[:] = u_h.x.array[:]
    nx_out, ny_out = 50, 50
    xs = np.linspace(0, 1, nx_out)
    ys = np.linspace(0, 1, ny_out)
    XX, YY = np.meshgrid(xs, ys, indexing='ij')
    points_2d = np.column_stack([XX.ravel(), YY.ravel()])
    points_3d = np.zeros((points_2d.shape[0], 3))
    points_3d[:, :2] = points_2d
    bb_tree = geometry.bb_tree(domain, tdim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, points_3d)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points_3d)
    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    for i in range(len(points_3d)):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(points_3d[i])
            cells_on_proc.append(links[0])
            eval_map.append(i)
    u_values = np.full(len(points_3d), np.nan)
    if len(points_on_proc) > 0:
        vals = u_h.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u_values[eval_map] = vals.flatten()
    u_grid = u_values.reshape((nx_out, ny_out))
    u_init_values = np.full(len(points_3d), np.nan)
    if len(points_on_proc) > 0:
        vals_init = u_initial_func.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u_init_values[eval_map] = vals_init.flatten()
    u_init_grid = u_init_values.reshape((nx_out, ny_out))
    solver_info = {
        "mesh_resolution": N, "element_degree": degree,
        "ksp_type": "cg", "pc_type": "hypre", "rtol": 1e-10,
        "iterations": total_iterations, "dt": dt_val,
        "n_steps": n_steps, "time_scheme": scheme,
    }
    return {"u": u_grid, "u_initial": u_init_grid, "solver_info": solver_info}


if __name__ == "__main__":
    import time
    start = time.perf_counter()
    result = solve()
    elapsed = time.perf_counter() - start
    print(f"Wall time: {elapsed:.3f}s")
    print(f"Solution shape: {result['u'].shape}")
    print(f"Solution range: [{np.nanmin(result['u']):.6f}, {np.nanmax(result['u']):.6f}]")
    nx_out, ny_out = 50, 50
    xs = np.linspace(0, 1, nx_out)
    ys = np.linspace(0, 1, ny_out)
    XX, YY = np.meshgrid(xs, ys, indexing='ij')
    u_exact = np.exp(-0.1) * np.sin(2*np.pi*XX) * np.sin(2*np.pi*YY)
    err = np.sqrt(np.nanmean((result['u'] - u_exact)**2))
    print(f"L2 error (grid): {err:.6e}")
    print(f"Solver info: {result['solver_info']}")
