import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc
import time as time_module

ScalarType = PETSc.ScalarType


def solve(case_spec: dict) -> dict:
    """Solve the heat equation using backward Euler time-stepping."""
    
    # ---- Extract parameters from case_spec ----
    pde = case_spec.get("pde", {})
    coeffs = pde.get("coefficients", {})
    kappa_val = coeffs.get("kappa", 0.5)
    
    time_params = pde.get("time", {})
    t_end = time_params.get("t_end", 0.2)
    dt_suggested = time_params.get("dt", 0.02)
    scheme = time_params.get("scheme", "backward_euler")
    
    output_spec = case_spec.get("output", {})
    nx_out = output_spec.get("nx", 50)
    ny_out = output_spec.get("ny", 50)
    
    # ---- Adaptive mesh refinement ----
    resolutions = [48, 80, 128]
    element_degree = 2
    dt = dt_suggested
    
    prev_norm = None
    final_result = None
    chosen_N = None
    
    for N in resolutions:
        result = _solve_heat(
            N, element_degree, kappa_val, t_end, dt, scheme,
            nx_out, ny_out
        )
        
        current_norm = np.linalg.norm(result["u"])
        
        if prev_norm is not None:
            rel_error = abs(current_norm - prev_norm) / (abs(current_norm) + 1e-15)
            if rel_error < 0.005:  # 0.5% convergence
                final_result = result
                chosen_N = N
                break
        
        prev_norm = current_norm
        final_result = result
        chosen_N = N
    
    # Build solver_info
    solver_info = {
        "mesh_resolution": chosen_N,
        "element_degree": element_degree,
        "ksp_type": "cg",
        "pc_type": "hypre",
        "rtol": 1e-10,
        "iterations": final_result["total_iterations"],
        "dt": dt,
        "n_steps": final_result["n_steps"],
        "time_scheme": scheme,
    }
    
    return {
        "u": final_result["u"],
        "u_initial": final_result["u_initial"],
        "solver_info": solver_info,
    }


def _solve_heat(N, degree, kappa_val, t_end, dt, scheme, nx_out, ny_out):
    """Core heat equation solver for a given mesh resolution."""
    
    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
    
    V = fem.functionspace(domain, ("Lagrange", degree))
    
    # Spatial coordinates and time
    x = ufl.SpatialCoordinate(domain)
    
    # Time parameter as a Constant
    t_const = fem.Constant(domain, ScalarType(0.0))
    kappa = fem.Constant(domain, ScalarType(kappa_val))
    dt_const = fem.Constant(domain, ScalarType(dt))
    
    # Manufactured solution: u = exp(-2*t)*cos(pi*x)*cos(pi*y)
    pi = ufl.pi
    u_exact_ufl = ufl.exp(-2 * t_const) * ufl.cos(pi * x[0]) * ufl.cos(pi * x[1])
    
    # Source term: f = u_t - kappa * laplacian(u)
    # u_t = -2*exp(-2t)*cos(pi*x)*cos(pi*y)
    # laplacian(u) = -2*pi^2*exp(-2t)*cos(pi*x)*cos(pi*y)
    # -kappa*laplacian(u) = 2*kappa*pi^2*exp(-2t)*cos(pi*x)*cos(pi*y)
    # f = u_t + (-kappa*laplacian(u)) = (-2 + 2*kappa*pi^2)*exp(-2t)*cos(pi*x)*cos(pi*y)
    f_ufl = (-2.0 + 2.0 * kappa_val * pi**2) * ufl.exp(-2 * t_const) * ufl.cos(pi * x[0]) * ufl.cos(pi * x[1])
    
    # Functions
    u_n = fem.Function(V, name="u_n")  # solution at previous time step
    u_h = fem.Function(V, name="u_h")  # solution at current time step
    
    # Set initial condition: u(x, 0) = cos(pi*x)*cos(pi*y)
    u_exact_expr_t0 = fem.Expression(
        ufl.cos(pi * x[0]) * ufl.cos(pi * x[1]),
        V.element.interpolation_points
    )
    u_n.interpolate(u_exact_expr_t0)
    
    # Store initial condition for output
    u_initial_grid = _evaluate_on_grid(domain, V, u_n, nx_out, ny_out)
    
    # Backward Euler: (u - u_n)/dt - kappa * laplacian(u) = f(t_{n+1})
    # Weak form: (u/dt)*v + kappa * grad(u) . grad(v) = (u_n/dt)*v + f*v
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    
    # Bilinear form (LHS)
    a = (u * v / dt_const + kappa * ufl.inner(ufl.grad(u), ufl.grad(v))) * ufl.dx
    
    # Linear form (RHS)
    L = (u_n * v / dt_const + f_ufl * v) * ufl.dx
    
    # Compile forms
    a_form = fem.form(a)
    L_form = fem.form(L)
    
    # Boundary conditions: u = u_exact on boundary
    tdim = domain.topology.dim
    fdim = tdim - 1
    
    # Find all boundary facets
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
    )
    
    bc_func = fem.Function(V)
    bc_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc = fem.dirichletbc(bc_func, bc_dofs)
    
    # Assemble matrix (constant in time for backward Euler with constant kappa and dt)
    A = petsc.assemble_matrix(a_form, bcs=[bc])
    A.assemble()
    
    # Setup KSP solver
    solver = PETSc.KSP().create(domain.comm)
    solver.setOperators(A)
    solver.setType(PETSc.KSP.Type.CG)
    pc = solver.getPC()
    pc.setType(PETSc.PC.Type.HYPRE)
    solver.setTolerances(rtol=1e-10, atol=1e-14, max_it=2000)
    solver.setUp()
    
    # Time stepping
    t = 0.0
    n_steps = int(round(t_end / dt))
    total_iterations = 0
    
    # Create RHS vector
    b = petsc.create_vector(V)
    
    for step in range(n_steps):
        t += dt
        t_const.value = t
        
        # Update boundary condition
        u_exact_expr = fem.Expression(u_exact_ufl, V.element.interpolation_points)
        bc_func.interpolate(u_exact_expr)
        
        # Assemble RHS
        with b.localForm() as loc:
            loc.set(0)
        petsc.assemble_vector(b, L_form)
        
        # Apply lifting for non-zero Dirichlet BCs
        petsc.apply_lifting(b, [a_form], bcs=[[bc]])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        petsc.set_bc(b, [bc])
        
        # Solve
        solver.solve(b, u_h.x.petsc_vec)
        u_h.x.scatter_forward()
        
        total_iterations += solver.getIterationNumber()
        
        # Update for next step
        u_n.x.array[:] = u_h.x.array[:]
    
    # Evaluate on output grid
    u_grid = _evaluate_on_grid(domain, V, u_h, nx_out, ny_out)
    
    # Cleanup
    solver.destroy()
    A.destroy()
    b.destroy()
    
    return {
        "u": u_grid,
        "u_initial": u_initial_grid,
        "n_steps": n_steps,
        "total_iterations": total_iterations,
    }


def _evaluate_on_grid(domain, V, u_func, nx, ny):
    """Evaluate a FEM function on a uniform nx x ny grid over [0,1]^2."""
    
    # Create grid points
    xs = np.linspace(0, 1, nx)
    ys = np.linspace(0, 1, ny)
    XX, YY = np.meshgrid(xs, ys, indexing='ij')
    
    points_2d = np.column_stack([XX.ravel(), YY.ravel()])
    # dolfinx needs 3D points
    points_3d = np.zeros((points_2d.shape[0], 3))
    points_3d[:, :2] = points_2d
    
    # Build bounding box tree
    bb_tree = geometry.bb_tree(domain, domain.topology.dim)
    
    # Find cells
    cell_candidates = geometry.compute_collisions_points(bb_tree, points_3d)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points_3d)
    
    # Evaluate
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
        pts_arr = np.array(points_on_proc)
        cells_arr = np.array(cells_on_proc, dtype=np.int32)
        vals = u_func.eval(pts_arr, cells_arr)
        u_values[eval_map] = vals.flatten()
    
    return u_values.reshape((nx, ny))


if __name__ == "__main__":
    # Test with default case_spec
    case_spec = {
        "pde": {
            "coefficients": {"kappa": 0.5},
            "time": {"t_end": 0.2, "dt": 0.02, "scheme": "backward_euler"},
            "manufactured_solution": "exp(-2*t)*cos(pi*x)*cos(pi*y)",
        },
        "domain": {"type": "unit_square"},
        "output": {"nx": 50, "ny": 50},
    }
    
    start = time_module.time()
    result = solve(case_spec)
    elapsed = time_module.time() - start
    
    u_grid = result["u"]
    print(f"Solution shape: {u_grid.shape}")
    print(f"Solution range: [{u_grid.min():.6f}, {u_grid.max():.6f}]")
    print(f"Wall time: {elapsed:.3f}s")
    print(f"Solver info: {result['solver_info']}")
    
    # Compute error against exact solution at t_end
    t_end = 0.2
    xs = np.linspace(0, 1, 50)
    ys = np.linspace(0, 1, 50)
    XX, YY = np.meshgrid(xs, ys, indexing='ij')
    u_exact = np.exp(-2 * t_end) * np.cos(np.pi * XX) * np.cos(np.pi * YY)
    
    error = np.sqrt(np.mean((u_grid - u_exact)**2))
    max_error = np.max(np.abs(u_grid - u_exact))
    print(f"L2 error (grid): {error:.6e}")
    print(f"Max error (grid): {max_error:.6e}")
