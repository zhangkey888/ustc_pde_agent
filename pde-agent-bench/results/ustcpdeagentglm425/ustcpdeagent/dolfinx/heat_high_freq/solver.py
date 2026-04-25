import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc

ScalarType = PETSc.ScalarType

def solve(case_spec: dict) -> dict:
    # ----- Parse case_spec -----
    pde = case_spec["pde"]
    coeff = pde.get("coefficients", {})
    time_info = pde.get("time", {})
    output_info = case_spec["output"]
    grid_info = output_info["grid"]

    kappa = coeff.get("kappa", 1.0)
    t0 = time_info.get("t0", 0.0)
    t_end = time_info.get("t_end", 0.1)
    dt_suggested = time_info.get("dt", 0.005)

    nx_out = grid_info["nx"]
    ny_out = grid_info["ny"]
    bbox = grid_info["bbox"]  # [xmin, xmax, ymin, ymax]
    xmin, xmax, ymin, ymax = bbox[0], bbox[1], bbox[2], bbox[3]

    # ----- Parameters (adaptive for accuracy) -----
    # High frequency solution (4*pi) needs good resolution
    # Each sin(4*pi*x) period is 0.25, so at least ~10 pts per period
    mesh_res = 80       # 80x80 mesh
    element_degree = 2  # P2 elements
    dt = 0.001          # smaller dt for better temporal accuracy
    n_steps = int(round((t_end - t0) / dt))
    if n_steps * dt < (t_end - t0) - 1e-12:
        n_steps += 1

    # ----- Manufactured solution and source term -----
    # u_exact = exp(-t) * sin(4*pi*x) * sin(4*pi*y)
    # f = du/dt - kappa * div(grad(u))
    #   = -exp(-t)*sin(4*pi*x)*sin(4*pi*y) - kappa*(-32*pi^2)*exp(-t)*sin(4*pi*x)*sin(4*pi*y)
    #   = (-1 + 32*pi^2*kappa) * exp(-t)*sin(4*pi*x)*sin(4*pi*y)

    # ----- Create mesh and function space -----
    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_square(comm, mesh_res, mesh_res, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", element_degree))

    # ----- Define functions -----
    u_n = fem.Function(V)       # solution at previous time step
    u_sol = fem.Function(V)     # current solution
    f_func = fem.Function(V)    # source term (time-dependent)
    g_func = fem.Function(V)   # boundary condition (time-dependent)

    # ----- Initial condition -----
    t = t0
    u_n.interpolate(lambda x: np.exp(-t) * np.sin(4*np.pi*x[0]) * np.sin(4*np.pi*x[1]))

    # Save initial condition for output
    u_initial_vals = u_n.x.array.copy()

    # ----- Boundary conditions -----
    # Dirichlet BC on entire boundary: u = g = exp(-t)*sin(4*pi*x)*sin(4*pi*y)
    tdim = domain.topology.dim
    fdim = tdim - 1
    boundary_facets = mesh.locate_entities_boundary(domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool))
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc = fem.dirichletbc(g_func, boundary_dofs)

    # ----- Variational form (backward Euler) -----
    # u - dt*kappa*div(grad(u)) = u_n + dt*f
    u_trial = ufl.TrialFunction(V)
    v_test = ufl.TestFunction(V)

    a_form = ufl.inner(u_trial, v_test) * ufl.dx + dt * kappa * ufl.inner(ufl.grad(u_trial), ufl.grad(v_test)) * ufl.dx
    L_form = ufl.inner(u_n, v_test) * ufl.dx + dt * ufl.inner(f_func, v_test) * ufl.dx

    a_compiled = fem.form(a_form)
    L_compiled = fem.form(L_form)

    # ----- Assemble matrix (time-independent) -----
    A = petsc.assemble_matrix(a_compiled, bcs=[bc])
    A.assemble()

    # ----- Setup solver -----
    ksp_type = "cg"
    pc_type = "ilu"
    rtol = 1e-10

    solver = PETSc.KSP().create(domain.comm)
    solver.setOperators(A)
    solver.setType(PETSc.KSP.Type.CG)
    solver.getPC().setType(PETSc.PC.Type.ILU)
    solver.setTolerances(rtol=rtol, atol=1e-14, max_it=500)
    solver.setFromOptions()

    b = petsc.create_vector(L_compiled.function_spaces)

    # ----- Time stepping -----
    total_iterations = 0

    for step in range(n_steps):
        t_new = t + dt

        # Update source term f at t_new
        f_func.interpolate(
            lambda x: (-1.0 + 32.0 * np.pi**2 * kappa) * np.exp(-t_new) * np.sin(4*np.pi*x[0]) * np.sin(4*np.pi*x[1])
        )

        # Update boundary condition g at t_new
        g_func.interpolate(
            lambda x: np.exp(-t_new) * np.sin(4*np.pi*x[0]) * np.sin(4*np.pi*x[1])
        )

        # Assemble RHS
        with b.localForm() as loc:
            loc.set(0)
        petsc.assemble_vector(b, L_compiled)

        # Apply lifting for BCs
        petsc.apply_lifting(b, [a_compiled], bcs=[[bc]])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        petsc.set_bc(b, [bc])

        # Solve
        solver.solve(b, u_sol.x.petsc_vec)
        u_sol.x.scatter_forward()

        # Count iterations
        total_iterations += solver.getIterationNumber()

        # Update previous solution
        u_n.x.array[:] = u_sol.x.array[:]
        t = t_new

    # ----- Sample solution on output grid -----
    xs = np.linspace(xmin, xmax, nx_out)
    ys = np.linspace(ymin, ymax, ny_out)
    XX, YY = np.meshgrid(xs, ys)
    points = np.zeros((nx_out * ny_out, 3))
    points[:, 0] = XX.ravel()
    points[:, 1] = YY.ravel()
    points[:, 2] = 0.0

    bb_tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, points)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points)

    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    for i in range(points.shape[0]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(points[i])
            cells_on_proc.append(links[0])
            eval_map.append(i)

    u_values = np.full((points.shape[0],), np.nan, dtype=np.float64)
    if len(points_on_proc) > 0:
        vals = u_sol.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u_values[eval_map] = vals.flatten()

    # Gather on all ranks (single process typically, but be safe)
    u_values_global = np.zeros_like(u_values)
    domain.comm.Allreduce(u_values, u_values_global, op=MPI.SUM)
    # Replace NaN with 0 (points outside domain - shouldn't happen for unit square)
    u_values_global = np.nan_to_num(u_values_global, nan=0.0)

    u_grid = u_values_global.reshape(ny_out, nx_out)

    # Also sample initial condition on the same grid for u_initial
    u_initial_grid = np.zeros((ny_out, nx_out), dtype=np.float64)
    # Re-interpolate initial condition for sampling
    u_init_func = fem.Function(V)
    u_init_func.interpolate(lambda x: np.exp(-t0) * np.sin(4*np.pi*x[0]) * np.sin(4*np.pi*x[1]))
    u_init_vals_at_points = np.full((points.shape[0],), np.nan, dtype=np.float64)
    if len(points_on_proc) > 0:
        vals_init = u_init_func.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u_init_vals_at_points[eval_map] = vals_init.flatten()
    u_init_global = np.zeros_like(u_init_vals_at_points)
    domain.comm.Allreduce(u_init_vals_at_points, u_init_global, op=MPI.SUM)
    u_init_global = np.nan_to_num(u_init_global, nan=0.0)
    u_initial_grid = u_init_global.reshape(ny_out, nx_out)

    # ----- Compute L2 error for verification -----
    u_exact_expr = lambda x: np.exp(-t_end) * np.sin(4*np.pi*x[0]) * np.sin(4*np.pi*x[1])
    u_exact_func = fem.Function(V)
    u_exact_func.interpolate(u_exact_expr)
    error_sq = domain.comm.allreduce(
        fem.assemble_scalar(fem.form(ufl.inner(u_sol - u_exact_func, u_sol - u_exact_func) * ufl.dx)),
        op=MPI.SUM
    )
    l2_error = np.sqrt(max(error_sq, 0.0))
    if domain.comm.rank == 0:
        print(f"L2 error at t={t_end}: {l2_error:.6e}")
        print(f"Total KSP iterations: {total_iterations}")
        print(f"Mesh: {mesh_res}, Degree: {element_degree}, dt: {dt}, Steps: {n_steps}")

    # ----- Build output -----
    result = {
        "u": u_grid,
        "u_initial": u_initial_grid,
        "solver_info": {
            "mesh_resolution": mesh_res,
            "element_degree": element_degree,
            "ksp_type": ksp_type,
            "pc_type": pc_type,
            "rtol": rtol,
            "iterations": total_iterations,
            "dt": dt,
            "n_steps": n_steps,
            "time_scheme": "backward_euler",
        }
    }

    return result


if __name__ == "__main__":
    # Test with default case_spec
    case_spec = {
        "pde": {
            "type": "heat",
            "coefficients": {"kappa": 1.0},
            "time": {"t0": 0.0, "t_end": 0.1, "dt": 0.005},
        },
        "output": {
            "grid": {
                "nx": 64,
                "ny": 64,
                "bbox": [0.0, 1.0, 0.0, 1.0],
            }
        },
    }
    result = solve(case_spec)
    print(f"Output shape: {result['u'].shape}")
    print(f"L2 error (from solver): computed above")
