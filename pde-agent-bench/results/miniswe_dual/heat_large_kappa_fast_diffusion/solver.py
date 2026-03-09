import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc

ScalarType = PETSc.ScalarType


def solve(case_spec: dict = None) -> dict:
    kappa_val = 5.0
    t_end = 0.08
    dt_val = 0.004
    scheme = "backward_euler"

    if case_spec is not None:
        coeffs = case_spec.get("pde", {}).get("coefficients", {})
        kappa_val = coeffs.get("kappa", kappa_val)
        time_params = case_spec.get("pde", {}).get("time", {})
        if time_params:
            t_end = time_params.get("t_end", t_end)
            dt_val = time_params.get("dt", dt_val)
            scheme = time_params.get("scheme", scheme)

    resolutions = [48, 80, 120]
    element_degree = 2
    prev_norm = None
    final_result = None

    for N in resolutions:
        result = _solve_at_resolution(N, element_degree, kappa_val, t_end, dt_val, scheme)
        curr_norm = result["norm"]
        if prev_norm is not None:
            rel_change = abs(curr_norm - prev_norm) / (abs(curr_norm) + 1e-15)
            if rel_change < 0.005:
                final_result = result
                break
        prev_norm = curr_norm
        final_result = result

    return final_result["output"]


def _solve_at_resolution(N, degree, kappa_val, t_end, dt_val, scheme):
    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", degree))
    x = ufl.SpatialCoordinate(domain)
    pi = ufl.pi
    t = fem.Constant(domain, ScalarType(0.0))
    u_exact_ufl = ufl.exp(-t) * ufl.sin(2 * pi * x[0]) * ufl.sin(pi * x[1])
    kappa_c = fem.Constant(domain, ScalarType(kappa_val))
    f_ufl = ufl.exp(-t) * ufl.sin(2 * pi * x[0]) * ufl.sin(pi * x[1]) * (-1.0 + kappa_val * 5.0 * pi**2)
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    u_n = fem.Function(V, name="u_n")
    u_h = fem.Function(V, name="u_h")
    dt_c = fem.Constant(domain, ScalarType(dt_val))
    u_n.interpolate(fem.Expression(
        ufl.sin(2 * pi * x[0]) * ufl.sin(pi * x[1]),
        V.element.interpolation_points
    ))
    a = (u / dt_c) * v * ufl.dx + kappa_c * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = (u_n / dt_c) * v * ufl.dx + f_ufl * v * ufl.dx
    tdim = domain.topology.dim
    fdim = tdim - 1
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x_arr: np.ones(x_arr.shape[1], dtype=bool)
    )
    u_bc = fem.Function(V)
    bc_expr = fem.Expression(u_exact_ufl, V.element.interpolation_points)
    u_bc.interpolate(bc_expr)
    dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc = fem.dirichletbc(u_bc, dofs)
    bcs = [bc]
    a_form = fem.form(a)
    L_form = fem.form(L)
    A = petsc.assemble_matrix(a_form, bcs=bcs)
    A.assemble()
    b = petsc.create_vector(V)
    solver = PETSc.KSP().create(domain.comm)
    solver.setOperators(A)
    solver.setType(PETSc.KSP.Type.CG)
    pc = solver.getPC()
    pc.setType(PETSc.PC.Type.HYPRE)
    solver.setTolerances(rtol=1e-10, atol=1e-12, max_it=2000)
    solver.setUp()
    n_steps = int(np.round(t_end / dt_val))
    actual_dt = t_end / n_steps
    dt_c.value = actual_dt
    total_iterations = 0
    current_time = 0.0
    for step in range(n_steps):
        current_time += actual_dt
        t.value = current_time
        u_bc.interpolate(bc_expr)
        with b.localForm() as loc:
            loc.set(0)
        petsc.assemble_vector(b, L_form)
        petsc.apply_lifting(b, [a_form], bcs=[bcs])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        petsc.set_bc(b, bcs)
        solver.solve(b, u_h.x.petsc_vec)
        u_h.x.scatter_forward()
        total_iterations += solver.getIterationNumber()
        u_n.x.array[:] = u_h.x.array[:]
    l2_norm = np.sqrt(domain.comm.allreduce(
        fem.assemble_scalar(fem.form(u_h * u_h * ufl.dx)), op=MPI.SUM
    ))
    nx_out, ny_out = 50, 50
    xs = np.linspace(0, 1, nx_out)
    ys = np.linspace(0, 1, ny_out)
    XX, YY = np.meshgrid(xs, ys, indexing='ij')
    points_2d = np.column_stack([XX.ravel(), YY.ravel()])
    points_3d = np.zeros((points_2d.shape[0], 3))
    points_3d[:, 0] = points_2d[:, 0]
    points_3d[:, 1] = points_2d[:, 1]
    bb_tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, points_3d)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points_3d)
    u_values = np.full(points_3d.shape[0], np.nan)
    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    for i in range(points_3d.shape[0]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(points_3d[i])
            cells_on_proc.append(links[0])
            eval_map.append(i)
    if len(points_on_proc) > 0:
        vals = u_h.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u_values[eval_map] = vals.flatten()
    u_grid = u_values.reshape((nx_out, ny_out))
    u_init_func = fem.Function(V)
    u_init_func.interpolate(fem.Expression(
        ufl.sin(2 * pi * x[0]) * ufl.sin(pi * x[1]),
        V.element.interpolation_points
    ))
    u_init_values = np.full(points_3d.shape[0], np.nan)
    if len(points_on_proc) > 0:
        vals_init = u_init_func.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u_init_values[eval_map] = vals_init.flatten()
    u_init_grid = u_init_values.reshape((nx_out, ny_out))
    solver.destroy()
    A.destroy()
    b.destroy()
    output = {
        "u": u_grid,
        "u_initial": u_init_grid,
        "solver_info": {
            "mesh_resolution": N,
            "element_degree": degree,
            "ksp_type": "cg",
            "pc_type": "hypre",
            "rtol": 1e-10,
            "iterations": total_iterations,
            "dt": actual_dt,
            "n_steps": n_steps,
            "time_scheme": "backward_euler",
        }
    }
    return {"norm": l2_norm, "output": output}


if __name__ == "__main__":
    import time
    start = time.time()
    result = solve()
    elapsed = time.time() - start
    print(f"Wall time: {elapsed:.2f}s")
    print(f"Solution shape: {result['u'].shape}")
    print(f"Solution range: [{np.nanmin(result['u']):.6f}, {np.nanmax(result['u']):.6f}]")
    print(f"Solver info: {result['solver_info']}")
    t_end = 0.08
    xs = np.linspace(0, 1, 50)
    ys = np.linspace(0, 1, 50)
    XX, YY = np.meshgrid(xs, ys, indexing='ij')
    u_exact = np.exp(-t_end) * np.sin(2 * np.pi * XX) * np.sin(np.pi * YY)
    mask = ~np.isnan(result['u'])
    error = np.sqrt(np.mean((result['u'][mask] - u_exact[mask])**2))
    print(f"L2 error (grid): {error:.6e}")
    print(f"Max error: {np.max(np.abs(result['u'][mask] - u_exact[mask])):.6e}")
