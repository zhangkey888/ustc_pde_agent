import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc
import time as time_module


def solve(case_spec: dict) -> dict:
    """Solve reaction-diffusion equation with Crank-Nicolson time stepping.
    
    PDE: du/dt - eps * laplacian(u) + sigma * u = f  in Omega x (0,T]
         u = g on dOmega
         u(x,0) = u0(x) in Omega
    
    Manufactured solution: u = exp(-t)*(sin(pi*x)*sin(pi*y) + 0.2*sin(6*pi*x)*sin(5*pi*y))
    """
    
    comm = MPI.COMM_WORLD
    
    # ---- Extract parameters from case_spec ----
    pde = case_spec.get("pde", {})
    
    # Diffusion coefficient
    epsilon_val = pde.get("epsilon", 1.0)
    
    # Reaction coefficient (for linear reaction R(u) = sigma * u)
    sigma_val = pde.get("sigma", 1.0)
    
    # Time parameters - hardcoded defaults as fallback
    time_params = pde.get("time", {})
    t_end = time_params.get("t_end", 0.4)
    dt_val = time_params.get("dt", 0.005)
    scheme = time_params.get("scheme", "crank_nicolson")
    is_transient = True  # Force transient for this problem
    
    # Domain
    domain_spec = case_spec.get("domain", {})
    x_min = domain_spec.get("x_min", 0.0)
    x_max = domain_spec.get("x_max", 1.0)
    y_min = domain_spec.get("y_min", 0.0)
    y_max = domain_spec.get("y_max", 1.0)
    
    # Output grid
    output = case_spec.get("output", {})
    nx_out = output.get("nx", 80)
    ny_out = output.get("ny", 80)
    
    # ---- Solver parameters ----
    element_degree = 2
    N = 80  # Mesh resolution - good balance for multifrequency content with P2
    
    # Create mesh
    domain = mesh.create_rectangle(
        comm,
        [np.array([x_min, y_min]), np.array([x_max, y_max])],
        [N, N],
        cell_type=mesh.CellType.triangle
    )
    
    V = fem.functionspace(domain, ("Lagrange", element_degree))
    
    # Spatial coordinates
    x = ufl.SpatialCoordinate(domain)
    
    # Constants
    eps_c = fem.Constant(domain, PETSc.ScalarType(epsilon_val))
    sig_c = fem.Constant(domain, PETSc.ScalarType(sigma_val))
    dt_c = fem.Constant(domain, PETSc.ScalarType(dt_val))
    
    pi = np.pi
    
    # Time constants for old and new time levels
    t_old = fem.Constant(domain, PETSc.ScalarType(0.0))
    t_new = fem.Constant(domain, PETSc.ScalarType(dt_val))
    
    # Helper: spatial part of manufactured solution
    def phi_spatial(x_coord):
        return (ufl.sin(pi * x_coord[0]) * ufl.sin(pi * x_coord[1])
                + 0.2 * ufl.sin(6 * pi * x_coord[0]) * ufl.sin(5 * pi * x_coord[1]))
    
    # Helper: exact solution at given time parameter
    def make_u_exact(t_param):
        return ufl.exp(-t_param) * phi_spatial(x)
    
    # Helper: source term at given time parameter
    # f = du/dt - eps*laplacian(u) + sigma*u
    # du/dt = -exp(-t)*phi(x,y)
    def make_f(t_param):
        u_ex = make_u_exact(t_param)
        dudt_loc = -ufl.exp(-t_param) * phi_spatial(x)
        neg_eps_lap = -eps_c * ufl.div(ufl.grad(u_ex))
        return dudt_loc + neg_eps_lap + sig_c * u_ex
    
    f_old = make_f(t_old)
    f_new = make_f(t_new)
    
    # ---- Variational formulation (Crank-Nicolson) ----
    u_n = fem.Function(V, name="u_n")  # solution at previous time step
    u_h = fem.Function(V, name="u_h")  # solution at current time step
    v = ufl.TestFunction(V)
    u_trial = ufl.TrialFunction(V)
    
    # Crank-Nicolson parameter
    theta = 0.5
    if scheme == "backward_euler":
        theta = 1.0
    elif scheme == "bdf2":
        theta = 1.0  # Will handle BDF2 separately if needed
    
    # Bilinear form (LHS)
    a_form = (
        u_trial * v / dt_c * ufl.dx
        + theta * eps_c * ufl.inner(ufl.grad(u_trial), ufl.grad(v)) * ufl.dx
        + theta * sig_c * u_trial * v * ufl.dx
    )
    
    # Linear form (RHS)
    L_form = (
        u_n * v / dt_c * ufl.dx
        - (1.0 - theta) * eps_c * ufl.inner(ufl.grad(u_n), ufl.grad(v)) * ufl.dx
        - (1.0 - theta) * sig_c * u_n * v * ufl.dx
        + theta * f_new * v * ufl.dx
        + (1.0 - theta) * f_old * v * ufl.dx
    )
    
    # ---- Boundary conditions ----
    tdim = domain.topology.dim
    fdim = tdim - 1
    
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x_arr: np.ones(x_arr.shape[1], dtype=bool)
    )
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    
    u_bc_func = fem.Function(V)
    u_exact_new = make_u_exact(t_new)
    bc_expr = fem.Expression(u_exact_new, V.element.interpolation_points)
    
    def update_bc():
        u_bc_func.interpolate(bc_expr)
    
    bc = fem.dirichletbc(u_bc_func, boundary_dofs)
    bcs = [bc]
    
    # ---- Initial condition ----
    t_init = fem.Constant(domain, PETSc.ScalarType(0.0))
    u_exact_init = make_u_exact(t_init)
    u_n.interpolate(fem.Expression(u_exact_init, V.element.interpolation_points))
    
    # ---- Compile forms ----
    a_compiled = fem.form(a_form)
    L_compiled = fem.form(L_form)
    
    # ---- KSP solver setup ----
    ksp = PETSc.KSP().create(domain.comm)
    ksp.setType(PETSc.KSP.Type.GMRES)
    pc = ksp.getPC()
    pc.setType("hypre")
    ksp.setTolerances(rtol=1e-10, atol=1e-12, max_it=1000)
    
    # ---- Time stepping ----
    n_steps = int(round(t_end / dt_val))
    total_iterations = 0
    
    for step in range(n_steps):
        current_t = step * dt_val
        next_t = (step + 1) * dt_val
        
        # Update time constants
        t_old.value = current_t
        t_new.value = next_t
        
        # Update boundary condition
        update_bc()
        
        # Assemble matrix
        A = petsc.assemble_matrix(a_compiled, bcs=bcs)
        A.assemble()
        
        # Assemble RHS
        b = petsc.assemble_vector(L_compiled)
        petsc.apply_lifting(b, [a_compiled], bcs=[bcs])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        petsc.set_bc(b, bcs)
        
        # Solve
        ksp.setOperators(A)
        ksp.solve(b, u_h.x.petsc_vec)
        u_h.x.scatter_forward()
        
        total_iterations += ksp.getIterationNumber()
        
        # Update u_n for next step
        u_n.x.array[:] = u_h.x.array[:]
        
        # Clean up PETSc objects
        A.destroy()
        b.destroy()
    
    # ---- Evaluate on output grid ----
    xs = np.linspace(x_min, x_max, nx_out)
    ys = np.linspace(y_min, y_max, ny_out)
    XX, YY = np.meshgrid(xs, ys, indexing='ij')
    
    points_2d = np.column_stack([XX.ravel(), YY.ravel()])
    points_3d = np.zeros((points_2d.shape[0], 3))
    points_3d[:, :2] = points_2d
    
    u_grid = _evaluate_function(domain, u_h, points_3d, nx_out, ny_out)
    
    # Evaluate initial condition on grid
    u_init_func = fem.Function(V)
    u_init_func.interpolate(fem.Expression(u_exact_init, V.element.interpolation_points))
    u_initial_grid = _evaluate_function(domain, u_init_func, points_3d, nx_out, ny_out)
    
    # Clean up solver
    ksp.destroy()
    
    result = {
        "u": u_grid,
        "u_initial": u_initial_grid,
        "solver_info": {
            "mesh_resolution": N,
            "element_degree": element_degree,
            "ksp_type": "gmres",
            "pc_type": "hypre",
            "rtol": 1e-10,
            "iterations": total_iterations,
            "dt": dt_val,
            "n_steps": n_steps,
            "time_scheme": scheme if scheme else "crank_nicolson",
        }
    }
    
    return result


def _evaluate_function(domain, u_func, points_3d, nx, ny):
    """Evaluate a dolfinx Function at given 3D points and return (nx, ny) grid."""
    bb_tree = geometry.bb_tree(domain, domain.topology.dim)
    
    cell_candidates = geometry.compute_collisions_points(bb_tree, points_3d)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points_3d)
    
    n_points = points_3d.shape[0]
    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    
    for i in range(n_points):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(points_3d[i])
            cells_on_proc.append(links[0])
            eval_map.append(i)
    
    u_values = np.full(n_points, np.nan)
    if len(points_on_proc) > 0:
        pts = np.array(points_on_proc)
        cls = np.array(cells_on_proc, dtype=np.int32)
        vals = u_func.eval(pts, cls)
        u_values[eval_map] = vals.flatten()
    
    return u_values.reshape((nx, ny))


if __name__ == "__main__":
    case_spec = {
        "pde": {
            "epsilon": 1.0,
            "sigma": 1.0,
            "time": {
                "t_end": 0.4,
                "dt": 0.005,
                "scheme": "crank_nicolson"
            }
        },
        "domain": {
            "x_min": 0.0, "x_max": 1.0,
            "y_min": 0.0, "y_max": 1.0
        },
        "output": {
            "nx": 80, "ny": 80
        }
    }
    
    t0 = time_module.time()
    result = solve(case_spec)
    elapsed = time_module.time() - t0
    
    u_grid = result["u"]
    print(f"Solution shape: {u_grid.shape}")
    print(f"Solution range: [{np.nanmin(u_grid):.6f}, {np.nanmax(u_grid):.6f}]")
    print(f"Wall time: {elapsed:.2f}s")
    print(f"Solver info: {result['solver_info']}")
    
    # Compare with exact solution at t=0.4
    xs = np.linspace(0, 1, 80)
    ys = np.linspace(0, 1, 80)
    XX, YY = np.meshgrid(xs, ys, indexing='ij')
    t_end_val = 0.4
    u_exact = np.exp(-t_end_val) * (
        np.sin(np.pi * XX) * np.sin(np.pi * YY)
        + 0.2 * np.sin(6 * np.pi * XX) * np.sin(5 * np.pi * YY)
    )
    
    error = np.sqrt(np.nanmean((u_grid - u_exact)**2))
    print(f"L2 error (grid): {error:.6e}")
    
    max_err = np.nanmax(np.abs(u_grid - u_exact))
    print(f"Max error: {max_err:.6e}")
