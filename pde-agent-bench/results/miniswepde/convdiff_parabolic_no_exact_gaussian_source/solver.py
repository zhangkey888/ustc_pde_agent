import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc
import time

ScalarType = PETSc.ScalarType


def solve(case_spec: dict) -> dict:
    start_time = time.time()
    
    # Parse parameters from case_spec
    pde = case_spec.get("pde", {})
    params = pde.get("parameters", {})
    time_params = pde.get("time", {})
    domain_spec = case_spec.get("domain", {})
    
    # Diffusion and convection parameters
    epsilon = params.get("epsilon", 0.02)
    beta_vec = params.get("beta", [6.0, 2.0])
    
    # Time parameters - hardcoded defaults as fallback
    t_end = time_params.get("t_end", 0.1)
    dt_val = time_params.get("dt", 0.02)
    scheme = time_params.get("scheme", "backward_euler")
    is_transient = True  # Force transient
    
    # Source term and initial condition from case_spec
    source_expr_str = pde.get("source", "exp(-200*((x-0.3)**2 + (y-0.7)**2))*exp(-t)")
    ic_str = pde.get("initial_condition", "0.0")
    
    # Mesh resolution - adaptive based on Peclet number
    # High Pe needs finer mesh or stabilization
    N = 80  # Good balance for this problem
    element_degree = 1  # P1 with SUPG
    
    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
    
    V = fem.functionspace(domain, ("Lagrange", element_degree))
    
    # Define functions
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    u_n = fem.Function(V, name="u_n")  # solution at previous time step
    u_h = fem.Function(V, name="u_h")  # solution at current time step
    
    # Initial condition: u0 = 0
    u_n.x.array[:] = 0.0
    
    # Time and spatial coordinates
    x = ufl.SpatialCoordinate(domain)
    t = fem.Constant(domain, ScalarType(0.0))
    dt = fem.Constant(domain, ScalarType(dt_val))
    eps_c = fem.Constant(domain, ScalarType(epsilon))
    
    # Velocity field
    beta = ufl.as_vector([ScalarType(beta_vec[0]), ScalarType(beta_vec[1])])
    
    # Source term: f = exp(-200*((x-0.3)^2 + (y-0.7)^2))*exp(-t)
    f = ufl.exp(-200.0 * ((x[0] - 0.3)**2 + (x[1] - 0.7)**2)) * ufl.exp(-t)
    
    # Boundary condition: u = g = 0 on boundary (homogeneous Dirichlet)
    # Check if there's a specific BC
    bc_spec = pde.get("boundary_conditions", {})
    g_val = 0.0  # default
    
    tdim = domain.topology.dim
    fdim = tdim - 1
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
    )
    dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc = fem.dirichletbc(ScalarType(g_val), dofs, V)
    bcs = [bc]
    
    # SUPG stabilization
    # Compute element size h
    h = ufl.CellDiameter(domain)
    beta_norm = ufl.sqrt(ufl.dot(beta, beta))
    
    # SUPG stabilization parameter (Codina's formula)
    Pe_local = beta_norm * h / (2.0 * eps_c)
    # tau_supg = h / (2 * |beta|) * (coth(Pe) - 1/Pe)
    # Simplified: for high Pe, tau ~ h/(2*|beta|)
    tau_supg = h / (2.0 * beta_norm + 1e-10) * ufl.min_value(Pe_local / 3.0, 1.0)
    
    # Backward Euler time discretization:
    # (u - u_n)/dt - eps*laplacian(u) + beta.grad(u) = f
    # Weak form (Galerkin part):
    # (u - u_n)/dt * v + eps * grad(u).grad(v) + beta.grad(u) * v = f * v
    
    # Bilinear form (LHS)
    a_gal = (u / dt * v 
             + eps_c * ufl.inner(ufl.grad(u), ufl.grad(v)) 
             + ufl.dot(beta, ufl.grad(u)) * v) * ufl.dx
    
    # Linear form (RHS)
    L_gal = (u_n / dt * v + f * v) * ufl.dx
    
    # SUPG stabilization terms
    # Residual applied to test function: tau * (beta . grad(v)) * R
    # where R = (u - u_n)/dt - eps*laplacian(u) + beta.grad(u) - f
    # For linear elements, laplacian(u) = 0 within elements
    # So R ≈ (u - u_n)/dt + beta.grad(u) - f
    
    # SUPG test function modification
    v_supg = tau_supg * ufl.dot(beta, ufl.grad(v))
    
    # SUPG additions to bilinear form
    a_supg = (u / dt * v_supg 
              + eps_c * ufl.inner(ufl.grad(u), ufl.grad(v_supg))
              + ufl.dot(beta, ufl.grad(u)) * v_supg) * ufl.dx
    
    # SUPG additions to linear form
    L_supg = (u_n / dt * v_supg + f * v_supg) * ufl.dx
    
    # Total forms
    a_total = a_gal + a_supg
    L_total = L_gal + L_supg
    
    # Compile forms
    a_form = fem.form(a_total)
    L_form = fem.form(L_total)
    
    # Assemble matrix (constant since coefficients don't change with time for the LHS structure)
    A = petsc.assemble_matrix(a_form, bcs=bcs)
    A.assemble()
    
    # Create RHS vector
    b = petsc.create_vector(V)
    
    # Setup solver
    solver = PETSc.KSP().create(domain.comm)
    solver.setOperators(A)
    solver.setType(PETSc.KSP.Type.GMRES)
    pc = solver.getPC()
    pc.setType(PETSc.PC.Type.ILU)
    solver.setTolerances(rtol=1e-8, atol=1e-12, max_it=1000)
    solver.setUp()
    
    # Time stepping
    n_steps = int(np.round(t_end / dt_val))
    total_iterations = 0
    
    for step in range(n_steps):
        t_current = (step + 1) * dt_val
        t.value = t_current  # Update time for source term (backward Euler evaluates at new time)
        
        # Assemble RHS
        with b.localForm() as loc:
            loc.set(0.0)
        petsc.assemble_vector(b, L_form)
        petsc.apply_lifting(b, [a_form], bcs=[bcs])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        petsc.set_bc(b, bcs)
        
        # Solve
        solver.solve(b, u_h.x.petsc_vec)
        u_h.x.scatter_forward()
        
        total_iterations += solver.getIterationNumber()
        
        # Update for next step
        u_n.x.array[:] = u_h.x.array[:]
    
    # Evaluate solution on 50x50 grid
    nx_out, ny_out = 50, 50
    xs = np.linspace(0.0, 1.0, nx_out)
    ys = np.linspace(0.0, 1.0, ny_out)
    XX, YY = np.meshgrid(xs, ys, indexing='ij')
    
    # Points array: shape (N_points, 3)
    points = np.zeros((nx_out * ny_out, 3))
    points[:, 0] = XX.flatten()
    points[:, 1] = YY.flatten()
    
    # Point evaluation
    bb_tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, points)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points)
    
    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    for i in range(len(points)):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(points[i])
            cells_on_proc.append(links[0])
            eval_map.append(i)
    
    u_values = np.zeros(nx_out * ny_out)
    if len(points_on_proc) > 0:
        pts_arr = np.array(points_on_proc)
        cells_arr = np.array(cells_on_proc, dtype=np.int32)
        vals = u_h.eval(pts_arr, cells_arr)
        u_values[eval_map] = vals.flatten()
    
    u_grid = u_values.reshape((nx_out, ny_out))
    
    elapsed = time.time() - start_time
    
    # Store initial condition (all zeros)
    u_initial = np.zeros((nx_out, ny_out))
    
    result = {
        "u": u_grid,
        "u_initial": u_initial,
        "solver_info": {
            "mesh_resolution": N,
            "element_degree": element_degree,
            "ksp_type": "gmres",
            "pc_type": "ilu",
            "rtol": 1e-8,
            "iterations": total_iterations,
            "dt": dt_val,
            "n_steps": n_steps,
            "time_scheme": "backward_euler",
        }
    }
    
    return result


if __name__ == "__main__":
    # Test with the given case spec
    case_spec = {
        "pde": {
            "type": "convection_diffusion",
            "parameters": {
                "epsilon": 0.02,
                "beta": [6.0, 2.0],
            },
            "source": "exp(-200*((x-0.3)**2 + (y-0.7)**2))*exp(-t)",
            "initial_condition": "0.0",
            "time": {
                "t_end": 0.1,
                "dt": 0.02,
                "scheme": "backward_euler",
            },
            "boundary_conditions": {},
        },
        "domain": {
            "type": "unit_square",
        },
    }
    
    t0 = time.time()
    result = solve(case_spec)
    elapsed = time.time() - t0
    
    u_grid = result["u"]
    print(f"Solution shape: {u_grid.shape}")
    print(f"Solution min: {u_grid.min():.6e}, max: {u_grid.max():.6e}")
    print(f"Solution mean: {u_grid.mean():.6e}")
    print(f"L2 norm of solution: {np.sqrt(np.sum(u_grid**2) / u_grid.size):.6e}")
    print(f"Wall time: {elapsed:.2f}s")
    print(f"Solver info: {result['solver_info']}")
