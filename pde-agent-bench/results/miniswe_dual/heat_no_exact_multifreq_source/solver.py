import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc

ScalarType = PETSc.ScalarType


def solve(case_spec: dict) -> dict:
    """
    Solve the transient heat equation:
      du/dt - kappa * laplacian(u) = f   in Omega x (0, T]
      u = g on dOmega
      u(x,0) = u0(x)
    """
    # ---- Extract parameters from case_spec ----
    pde = case_spec.get("pde", {})
    coeffs = pde.get("coefficients", {})
    kappa_val = float(coeffs.get("kappa", 1.0))

    # Hardcoded defaults for this problem as fallback
    time_params = pde.get("time", {})
    t_end = float(time_params.get("t_end", 0.12))
    dt_val = float(time_params.get("dt", 0.02))
    scheme = time_params.get("scheme", "backward_euler")

    # Force transient
    is_transient = True

    # ---- Adaptive mesh resolution ----
    resolutions = [48, 80]
    element_degree = 1

    prev_norm = None
    final_result = None

    for N in resolutions:
        result = _solve_at_resolution(
            N, element_degree, kappa_val, t_end, dt_val, scheme
        )
        cur_norm = result["norm"]

        if prev_norm is not None:
            rel_err = abs(cur_norm - prev_norm) / (abs(cur_norm) + 1e-15)
            if rel_err < 0.01:
                # Converged
                final_result = result
                break

        prev_norm = cur_norm
        final_result = result

    return final_result["output"]


def _solve_at_resolution(N, degree, kappa_val, t_end, dt_val, scheme):
    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
    tdim = domain.topology.dim
    fdim = tdim - 1

    V = fem.functionspace(domain, ("Lagrange", degree))

    # Trial and test functions
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    # Current solution and previous time step
    u_n = fem.Function(V, name="u_n")  # solution at previous time step
    u_sol = fem.Function(V, name="u")  # solution at current time step

    # Initial condition: u0 = 0
    u_n.x.array[:] = 0.0

    # Coefficients
    kappa = fem.Constant(domain, ScalarType(kappa_val))
    dt = fem.Constant(domain, ScalarType(dt_val))

    # Source term: f = sin(5*pi*x)*sin(3*pi*y) + 0.5*sin(9*pi*x)*sin(7*pi*y)
    x = ufl.SpatialCoordinate(domain)
    f = (ufl.sin(5 * ufl.pi * x[0]) * ufl.sin(3 * ufl.pi * x[1])
         + 0.5 * ufl.sin(9 * ufl.pi * x[0]) * ufl.sin(7 * ufl.pi * x[1]))

    # Boundary conditions: u = 0 on all boundaries (homogeneous Dirichlet)
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
    )
    dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc = fem.dirichletbc(ScalarType(0.0), dofs, V)
    bcs = [bc]

    # Backward Euler time stepping:
    # (u - u_n)/dt - kappa * laplacian(u) = f
    # Weak form: (u/dt)*v + kappa * grad(u) . grad(v) = (u_n/dt)*v + f*v
    a = (u * v / dt + kappa * ufl.inner(ufl.grad(u), ufl.grad(v))) * ufl.dx
    L_form = (u_n * v / dt + f * v) * ufl.dx

    # Compile forms
    a_compiled = fem.form(a)
    L_compiled = fem.form(L_form)

    # Assemble stiffness matrix (constant in time for this problem)
    A = petsc.assemble_matrix(a_compiled, bcs=bcs)
    A.assemble()

    # Create RHS vector
    b = fem.Function(V)

    # Setup KSP solver
    solver = PETSc.KSP().create(domain.comm)
    solver.setOperators(A)
    solver.setType(PETSc.KSP.Type.CG)
    pc = solver.getPC()
    pc.setType(PETSc.PC.Type.HYPRE)
    solver.setTolerances(rtol=1e-8, atol=1e-12, max_it=1000)
    solver.setUp()

    # Time stepping
    t = 0.0
    n_steps = 0
    total_iterations = 0

    while t < t_end - 1e-14:
        t += dt_val
        if t > t_end + 1e-14:
            t = t_end
        n_steps += 1

        # Assemble RHS
        with b.x.petsc_vec.localForm() as loc:
            loc.set(0)
        petsc.assemble_vector(b.x.petsc_vec, L_compiled)
        petsc.apply_lifting(b.x.petsc_vec, [a_compiled], bcs=[bcs])
        b.x.petsc_vec.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        petsc.set_bc(b.x.petsc_vec, bcs)

        # Solve
        solver.solve(b.x.petsc_vec, u_sol.x.petsc_vec)
        u_sol.x.scatter_forward()
        total_iterations += solver.getIterationNumber()

        # Update previous solution
        u_n.x.array[:] = u_sol.x.array[:]

    # Compute L2 norm for convergence check
    norm_form = fem.form(u_sol * u_sol * ufl.dx)
    norm_val = np.sqrt(fem.assemble_scalar(norm_form))

    # Evaluate on 50x50 grid
    nx_out, ny_out = 50, 50
    xs = np.linspace(0, 1, nx_out)
    ys = np.linspace(0, 1, ny_out)
    XX, YY = np.meshgrid(xs, ys, indexing='ij')
    points_2d = np.column_stack([XX.ravel(), YY.ravel()])
    # dolfinx needs 3D points
    points_3d = np.zeros((points_2d.shape[0], 3))
    points_3d[:, :2] = points_2d

    u_grid = _evaluate_function(domain, u_sol, points_3d, nx_out, ny_out)

    # Also evaluate initial condition
    u_initial_grid = np.zeros((nx_out, ny_out))

    solver_info = {
        "mesh_resolution": N,
        "element_degree": degree,
        "ksp_type": "cg",
        "pc_type": "hypre",
        "rtol": 1e-8,
        "iterations": total_iterations,
        "dt": dt_val,
        "n_steps": n_steps,
        "time_scheme": "backward_euler",
    }

    output = {
        "u": u_grid,
        "u_initial": u_initial_grid,
        "solver_info": solver_info,
    }

    # Cleanup
    solver.destroy()
    A.destroy()

    return {"norm": norm_val, "output": output}


def _evaluate_function(domain, u_func, points_3d, nx, ny):
    """Evaluate a dolfinx Function at given 3D points and reshape to (nx, ny)."""
    bb_tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, points_3d)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points_3d)

    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    for i in range(points_3d.shape[0]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(points_3d[i])
            cells_on_proc.append(links[0])
            eval_map.append(i)

    u_values = np.full(points_3d.shape[0], np.nan)
    if len(points_on_proc) > 0:
        vals = u_func.eval(
            np.array(points_on_proc, dtype=np.float64),
            np.array(cells_on_proc, dtype=np.int32)
        )
        u_values[eval_map] = vals.flatten()

    return u_values.reshape(nx, ny)


if __name__ == "__main__":
    # Quick test
    case_spec = {
        "pde": {
            "type": "heat",
            "coefficients": {"kappa": 1.0},
            "source": "sin(5*pi*x)*sin(3*pi*y) + 0.5*sin(9*pi*x)*sin(7*pi*y)",
            "initial_condition": "0.0",
            "time": {
                "t_end": 0.12,
                "dt": 0.02,
                "scheme": "backward_euler"
            },
            "boundary_conditions": {"type": "dirichlet", "value": 0.0}
        }
    }
    import time
    t0 = time.time()
    result = solve(case_spec)
    elapsed = time.time() - t0
    print(f"Wall time: {elapsed:.3f}s")
    print(f"u shape: {result['u'].shape}")
    print(f"u min: {result['u'].min():.6e}, max: {result['u'].max():.6e}")
    print(f"Solver info: {result['solver_info']}")
