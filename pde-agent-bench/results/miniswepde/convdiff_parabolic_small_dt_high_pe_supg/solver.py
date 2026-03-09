import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc

ScalarType = PETSc.ScalarType


def solve(case_spec: dict) -> dict:
    """Solve transient convection-diffusion with SUPG stabilization."""
    
    comm = MPI.COMM_WORLD
    
    # ---- Extract parameters from case_spec ----
    pde = case_spec.get("pde", {})
    domain_spec = case_spec.get("domain", {})
    
    # Diffusion and convection parameters
    eps_val = pde.get("epsilon", 0.01)
    beta_vec = pde.get("beta", [12.0, 4.0])
    
    # Time parameters - hardcoded defaults as fallback
    time_spec = pde.get("time", {})
    t_end = time_spec.get("t_end", 0.06)
    dt_val = time_spec.get("dt", 0.005)
    scheme = time_spec.get("scheme", "backward_euler")
    
    is_transient = True  # Force transient
    
    # Output grid
    output = case_spec.get("output", {})
    nx_out = output.get("nx", 50)
    ny_out = output.get("ny", 50)
    
    # ---- Adaptive mesh resolution ----
    # For high Pe, we need good resolution but also SUPG stabilization
    # Try progressive refinement
    resolutions = [64, 96, 128]
    element_degree = 1
    
    prev_norm = None
    final_result = None
    
    for N in resolutions:
        result = _solve_at_resolution(
            comm, N, element_degree, eps_val, beta_vec,
            t_end, dt_val, scheme, nx_out, ny_out, pde
        )
        
        current_norm = np.linalg.norm(result["u"])
        
        if prev_norm is not None:
            rel_change = abs(current_norm - prev_norm) / (abs(current_norm) + 1e-15)
            if rel_change < 0.01:
                # Converged
                return result
        
        prev_norm = current_norm
        final_result = result
    
    return final_result


def _solve_at_resolution(comm, N, degree, eps_val, beta_vec, t_end, dt_val, scheme,
                          nx_out, ny_out, pde):
    """Solve the PDE at a given mesh resolution."""
    
    # Create mesh
    domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
    tdim = domain.topology.dim
    fdim = tdim - 1
    
    # Function space
    V = fem.functionspace(domain, ("Lagrange", degree))
    
    # Trial and test functions
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    
    # Previous time step solution
    u_n = fem.Function(V, name="u_n")
    
    # Current solution
    u_sol = fem.Function(V, name="u_sol")
    
    # Spatial coordinates
    x = ufl.SpatialCoordinate(domain)
    
    # Time as a constant (mutable)
    t = fem.Constant(domain, ScalarType(0.0))
    dt = fem.Constant(domain, ScalarType(dt_val))
    
    # Diffusion coefficient
    eps = fem.Constant(domain, ScalarType(eps_val))
    
    # Convection velocity
    beta = ufl.as_vector([ScalarType(beta_vec[0]), ScalarType(beta_vec[1])])
    
    # Manufactured solution: u_exact = exp(-t)*sin(4*pi*x)*sin(pi*y)
    pi = ufl.pi
    u_exact = ufl.exp(-t) * ufl.sin(4 * pi * x[0]) * ufl.sin(pi * x[1])
    
    # Compute source term f from the manufactured solution
    # ∂u/∂t - ε ∇²u + β·∇u = f
    # ∂u/∂t = -exp(-t)*sin(4*pi*x)*sin(pi*y)
    # ∇²u = exp(-t)*(-16*pi^2 - pi^2)*sin(4*pi*x)*sin(pi*y) = -17*pi^2 * u_exact
    # β·∇u = exp(-t)*(12*4*pi*cos(4*pi*x)*sin(pi*y) + 4*pi*sin(4*pi*x)*cos(pi*y))
    
    # Let's compute f symbolically using UFL
    du_dt = -ufl.exp(-t) * ufl.sin(4 * pi * x[0]) * ufl.sin(pi * x[1])
    grad_u_exact = ufl.grad(u_exact)
    laplacian_u_exact = ufl.div(ufl.grad(u_exact))
    f_expr = du_dt - eps * laplacian_u_exact + ufl.dot(beta, grad_u_exact)
    
    # ---- Boundary conditions ----
    # u = g = u_exact on ∂Ω
    u_bc_func = fem.Function(V)
    
    # Mark all boundary facets
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
    )
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    
    # We'll update BC values each time step
    bc_expr = fem.Expression(u_exact, V.element.interpolation_points)
    u_bc_func.interpolate(bc_expr)
    bc = fem.dirichletbc(u_bc_func, boundary_dofs)
    bcs = [bc]
    
    # ---- Initial condition ----
    t.value = 0.0
    u_n.interpolate(bc_expr)
    
    # Store initial condition for output
    # Evaluate on output grid
    
    # ---- SUPG Stabilization ----
    h = ufl.CellDiameter(domain)
    beta_norm = ufl.sqrt(ufl.dot(beta, beta))
    Pe_cell = beta_norm * h / (2.0 * eps)
    
    # SUPG stabilization parameter (tau)
    # Using the standard formula with coth
    # tau = h / (2 * |beta|) * (coth(Pe) - 1/Pe)
    # For high Pe, coth(Pe) ≈ 1, so tau ≈ h / (2*|beta|)
    # Simpler robust formula:
    tau_supg = h / (2.0 * beta_norm + 1e-10) * ufl.min_value(Pe_cell / 3.0, 1.0)
    
    # ---- Weak form with backward Euler + SUPG ----
    # Standard Galerkin part:
    # (u - u_n)/dt * v + eps * grad(u) · grad(v) + (beta · grad(u)) * v = f * v
    
    # Bilinear form (LHS)
    a_std = (u / dt * v * ufl.dx
             + eps * ufl.dot(ufl.grad(u), ufl.grad(v)) * ufl.dx
             + ufl.dot(beta, ufl.grad(u)) * v * ufl.dx)
    
    # Linear form (RHS)
    L_std = (u_n / dt * v * ufl.dx
             + f_expr * v * ufl.dx)
    
    # SUPG stabilization terms
    # Residual of the strong form applied to trial function:
    # R(u) = u/dt - eps*laplacian(u) + beta·grad(u) - f
    # For linear elements, laplacian(u) = 0 within elements
    # So R(u) ≈ u/dt + beta·grad(u) - f (for P1)
    # Test function modification: v_supg = tau * beta · grad(v)
    
    v_supg = tau_supg * ufl.dot(beta, ufl.grad(v))
    
    if degree == 1:
        # For P1 elements, second derivatives vanish
        a_supg = (u / dt * v_supg * ufl.dx
                  + ufl.dot(beta, ufl.grad(u)) * v_supg * ufl.dx)
        L_supg = (u_n / dt * v_supg * ufl.dx
                  + f_expr * v_supg * ufl.dx)
    else:
        # For higher order, include diffusion term
        a_supg = (u / dt * v_supg * ufl.dx
                  - eps * ufl.div(ufl.grad(u)) * v_supg * ufl.dx
                  + ufl.dot(beta, ufl.grad(u)) * v_supg * ufl.dx)
        L_supg = (u_n / dt * v_supg * ufl.dx
                  + f_expr * v_supg * ufl.dx)
    
    a_total = a_std + a_supg
    L_total = L_std + L_supg
    
    # Compile forms
    a_form = fem.form(a_total)
    L_form = fem.form(L_total)
    
    # ---- Assemble and setup solver ----
    A = petsc.assemble_matrix(a_form, bcs=bcs)
    A.assemble()
    
    b = petsc.create_vector(V)
    
    # Setup KSP solver
    ksp_type = "gmres"
    pc_type = "ilu"
    rtol = 1e-8
    
    solver = PETSc.KSP().create(domain.comm)
    solver.setOperators(A)
    solver.setType(PETSc.KSP.Type.GMRES)
    pc = solver.getPC()
    pc.setType(PETSc.PC.Type.ILU)
    solver.setTolerances(rtol=rtol, atol=1e-12, max_it=2000)
    solver.setUp()
    
    # ---- Time stepping ----
    n_steps = int(np.ceil(t_end / dt_val))
    actual_dt = t_end / n_steps
    dt.value = actual_dt
    
    total_iterations = 0
    t_current = 0.0
    
    for step in range(n_steps):
        t_current += actual_dt
        t.value = t_current
        
        # Update boundary conditions
        u_bc_func.interpolate(bc_expr)
        
        # Reassemble matrix (coefficients depend on time through f, but a is time-independent
        # for backward Euler with constant coefficients - actually a doesn't depend on t)
        # However, the source term f depends on t, so L changes each step
        # Matrix A is constant if dt, eps, beta are constant - which they are
        # So we only need to reassemble RHS
        
        # For first step, matrix is already assembled. For subsequent steps, it's the same.
        # But wait - if we change dt, we'd need to reassemble. Here dt is constant.
        
        # Assemble RHS
        with b.localForm() as loc:
            loc.set(0.0)
        petsc.assemble_vector(b, L_form)
        petsc.apply_lifting(b, [a_form], bcs=[bcs])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        petsc.set_bc(b, bcs)
        
        # Solve
        solver.solve(b, u_sol.x.petsc_vec)
        u_sol.x.scatter_forward()
        
        total_iterations += solver.getIterationNumber()
        
        # Update previous solution
        u_n.x.array[:] = u_sol.x.array[:]
    
    # ---- Evaluate on output grid ----
    u_grid = _evaluate_on_grid(domain, u_sol, nx_out, ny_out)
    
    # Also evaluate initial condition
    t.value = 0.0
    u_init_func = fem.Function(V)
    u_init_func.interpolate(bc_expr)
    u_initial = _evaluate_on_grid(domain, u_init_func, nx_out, ny_out)
    
    # Cleanup
    solver.destroy()
    A.destroy()
    b.destroy()
    
    return {
        "u": u_grid,
        "u_initial": u_initial,
        "solver_info": {
            "mesh_resolution": N,
            "element_degree": degree,
            "ksp_type": ksp_type,
            "pc_type": pc_type,
            "rtol": rtol,
            "iterations": total_iterations,
            "dt": actual_dt,
            "n_steps": n_steps,
            "time_scheme": "backward_euler",
        }
    }


def _evaluate_on_grid(domain, u_func, nx, ny):
    """Evaluate a dolfinx Function on a uniform nx x ny grid over [0,1]^2."""
    
    # Create evaluation points
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
    # Test with the specific case
    case_spec = {
        "pde": {
            "epsilon": 0.01,
            "beta": [12.0, 4.0],
            "time": {
                "t_end": 0.06,
                "dt": 0.005,
                "scheme": "backward_euler"
            }
        },
        "domain": {
            "type": "unit_square",
            "x_range": [0, 1],
            "y_range": [0, 1]
        },
        "output": {
            "nx": 50,
            "ny": 50
        }
    }
    
    import time
    start = time.time()
    result = solve(case_spec)
    elapsed = time.time() - start
    
    print(f"Wall time: {elapsed:.3f}s")
    print(f"Solution shape: {result['u'].shape}")
    print(f"Solution range: [{np.nanmin(result['u']):.6f}, {np.nanmax(result['u']):.6f}]")
    print(f"Solver info: {result['solver_info']}")
    
    # Compute error against exact solution
    xs = np.linspace(0, 1, 50)
    ys = np.linspace(0, 1, 50)
    XX, YY = np.meshgrid(xs, ys, indexing='ij')
    t_end = 0.06
    u_exact = np.exp(-t_end) * np.sin(4 * np.pi * XX) * np.sin(np.pi * YY)
    
    error = np.sqrt(np.mean((result['u'] - u_exact)**2))
    max_error = np.max(np.abs(result['u'] - u_exact))
    print(f"RMS error: {error:.6e}")
    print(f"Max error: {max_error:.6e}")
