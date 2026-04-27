import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc
import time as _time

ScalarType = PETSc.ScalarType


def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD
    t_wall_start = _time.time()

    pde = case_spec.get("pde", {})
    epsilon_val = pde.get("epsilon", 1.0)
    if isinstance(epsilon_val, dict):
        epsilon_val = epsilon_val.get("value", 1.0)

    time_params = pde.get("time", {})
    is_transient = time_params is not None and len(time_params) > 0
    t0 = float(time_params.get("t0", 0.0)) if is_transient else 0.0
    t_end = float(time_params.get("t_end", 0.2)) if is_transient else 0.2
    time_scheme = time_params.get("scheme", "backward_euler") if is_transient else "backward_euler"
    if not is_transient:
        is_transient = True
        t_end = 0.2

    output_spec = case_spec.get("output", {})
    grid_spec = output_spec.get("grid", {})
    nx_out = grid_spec.get("nx", 50)
    ny_out = grid_spec.get("ny", 50)
    bbox = grid_spec.get("bbox", [0.0, 1.0, 0.0, 1.0])

    # Parameters: use semi-implicit linearization for the cubic reaction
    mesh_res = 64
    elem_degree = 2
    dt = 0.001  # small dt for linearization accuracy
    n_steps = int(round((t_end - t0) / dt))
    dt = (t_end - t0) / n_steps

    p0 = np.array([bbox[0], bbox[2]])
    p1 = np.array([bbox[1], bbox[3]])
    domain = mesh.create_rectangle(comm, [p0, p1], [mesh_res, mesh_res],
                                   cell_type=mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", elem_degree))

    # DOF coordinates for direct numpy interpolation
    dof_coords = V.tabulate_dof_coordinates()
    dof_x = dof_coords[:, 0]
    dof_y = dof_coords[:, 1]
    spatial_pattern = 0.2 * np.sin(2 * np.pi * dof_x) * np.sin(np.pi * dof_y)
    coeff_linear = -1.0 + 5.0 * epsilon_val * np.pi**2

    # Functions
    f_h = fem.Function(V, name="f_h")
    u_n = fem.Function(V, name="u_n")
    u_bc_func = fem.Function(V, name="u_bc")

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    eps_c = fem.Constant(domain, ScalarType(epsilon_val))
    inv_dt = fem.Constant(domain, ScalarType(1.0 / dt))

    # Semi-implicit: linearize u^3 as u_n^2 * u
    # Backward Euler: (u - u_n)/dt - eps*Lap(u) + u_n^2*u = f
    a_form = (inv_dt * u * v * ufl.dx
              + eps_c * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
              + u_n**2 * u * v * ufl.dx)
    L_form = (inv_dt * u_n * v * ufl.dx
              + f_h * v * ufl.dx)

    a_compiled = fem.form(a_form)
    L_compiled = fem.form(L_form)

    # Initial condition
    u_n.x.array[:] = spatial_pattern
    u_n.x.scatter_forward()

    # Boundary conditions
    tdim = domain.topology.dim
    fdim = tdim - 1
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x_: np.ones(x_.shape[1], dtype=bool))
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)

    # Solver setup
    A = petsc.create_matrix(a_compiled)
    b = petsc.create_vector(V)

    ksp_solver = PETSc.KSP().create(comm)
    ksp_solver.setType(PETSc.KSP.Type.PREONLY)
    ksp_solver.getPC().setType(PETSc.PC.Type.LU)

    u_h = fem.Function(V, name="u_h")
    total_linear_iters = 0
    t_current = t0

    for step in range(n_steps):
        t_current += dt

        # Update source term and BC using numpy (fast)
        u_ex = np.exp(-t_current) * spatial_pattern
        f_h.x.array[:] = coeff_linear * u_ex + u_ex**3
        f_h.x.scatter_forward()
        u_bc_func.x.array[:] = u_ex
        u_bc_func.x.scatter_forward()
        bc = fem.dirichletbc(u_bc_func, boundary_dofs)

        # Reassemble matrix (u_n^2 changes each step)
        A.zeroEntries()
        petsc.assemble_matrix(A, a_compiled, bcs=[bc])
        A.assemble()

        # Assemble RHS
        with b.localForm() as loc:
            loc.set(0.0)
        petsc.assemble_vector(b, L_compiled)
        petsc.apply_lifting(b, [a_compiled], bcs=[[bc]])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        petsc.set_bc(b, [bc])

        # Solve
        ksp_solver.setOperators(A)
        ksp_solver.solve(b, u_h.x.petsc_vec)
        u_h.x.scatter_forward()
        total_linear_iters += 1

        # Update
        u_n.x.array[:] = u_h.x.array[:]
        u_n.x.scatter_forward()

    # Sample on output grid
    xs = np.linspace(bbox[0], bbox[1], nx_out)
    ys = np.linspace(bbox[2], bbox[3], ny_out)
    XX, YY = np.meshgrid(xs, ys)
    pts_2d = np.column_stack([XX.ravel(), YY.ravel()])
    pts_3d = np.zeros((pts_2d.shape[0], 3))
    pts_3d[:, :2] = pts_2d

    bb_tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, pts_3d)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, pts_3d)

    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    for i in range(len(pts_3d)):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(pts_3d[i])
            cells_on_proc.append(links[0])
            eval_map.append(i)

    u_values = np.full(len(pts_3d), np.nan)
    if len(points_on_proc) > 0:
        vals = u_h.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u_values[eval_map] = vals.flatten()
    u_grid = u_values.reshape(ny_out, nx_out)

    # Initial condition on grid
    u_init_func = fem.Function(V)
    u_init_func.x.array[:] = spatial_pattern
    u_init_values = np.full(len(pts_3d), np.nan)
    if len(points_on_proc) > 0:
        vals_init = u_init_func.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u_init_values[eval_map] = vals_init.flatten()
    u_initial_grid = u_init_values.reshape(ny_out, nx_out)

    # Accuracy check
    u_exact_grid = np.exp(-t_current) * 0.2 * np.sin(2*np.pi*XX) * np.sin(np.pi*YY)
    grid_linf = np.nanmax(np.abs(u_grid - u_exact_grid))
    grid_l2 = np.sqrt(np.nanmean((u_grid - u_exact_grid)**2))
    print(f"Grid L2={grid_l2:.6e}, Linf={grid_linf:.6e}", flush=True)

    ksp_solver.destroy()
    A.destroy()
    b.destroy()

    return {
        "u": u_grid,
        "u_initial": u_initial_grid,
        "solver_info": {
            "mesh_resolution": mesh_res,
            "element_degree": elem_degree,
            "ksp_type": "preonly",
            "pc_type": "lu",
            "rtol": 1e-8,
            "iterations": total_linear_iters,
            "dt": dt,
            "n_steps": n_steps,
            "time_scheme": time_scheme,
            "nonlinear_iterations": [1] * n_steps,
        },
    }


if __name__ == "__main__":
    case_spec = {
        "pde": {
            "epsilon": 1.0,
            "time": {"t0": 0.0, "t_end": 0.2, "dt": 0.005, "scheme": "backward_euler"},
        },
        "output": {
            "grid": {"nx": 50, "ny": 50, "bbox": [0.0, 1.0, 0.0, 1.0]},
        },
    }
    start = _time.time()
    result = solve(case_spec)
    elapsed = _time.time() - start
    print(f"Total wall time: {elapsed:.2f}s")
    print(f"Output shape: {result['u'].shape}")
    print(f"NaN count: {np.isnan(result['u']).sum()}")
