import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc
import time as time_module

ScalarType = PETSc.ScalarType

def solve(case_spec: dict) -> dict:
    """Solve the transient heat equation with backward Euler."""
    
    # Extract parameters from case_spec with defaults
    pde = case_spec.get("pde", {})
    coeffs = pde.get("coefficients", {})
    kappa_val = float(coeffs.get("kappa", 1.0))
    
    time_params = pde.get("time", {})
    t_end = float(time_params.get("t_end", 0.08))
    dt_val = float(time_params.get("dt", 0.008))
    scheme = time_params.get("scheme", "backward_euler")
    
    # Output grid
    output = case_spec.get("output", {})
    nx_out = output.get("nx", 50)
    ny_out = output.get("ny", 50)
    
    # Parameters chosen for accuracy and speed
    N = 64
    degree = 2
    
    # Use dt/4 for temporal accuracy (backward Euler is first-order)
    dt_use = dt_val / 4.0  # 0.002
    n_steps = int(round(t_end / dt_use))
    dt_use = t_end / n_steps  # Adjust to hit t_end exactly
    
    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
    
    V = fem.functionspace(domain, ("Lagrange", degree))
    
    xs = ufl.SpatialCoordinate(domain)
    t_const = fem.Constant(domain, ScalarType(0.0))
    dt_const = fem.Constant(domain, ScalarType(dt_use))
    kappa_c = fem.Constant(domain, ScalarType(kappa_val))
    
    # Manufactured solution: u = exp(-t)*exp(5*y)*sin(pi*x)
    # where x = xs[0], y = xs[1]
    u_exact_ufl = ufl.exp(-t_const) * ufl.exp(5.0 * xs[1]) * ufl.sin(ufl.pi * xs[0])
    
    # Source term: f = du/dt - kappa * laplacian(u)
    # u = exp(-t)*exp(5*y)*sin(pi*x)
    # du/dt = -exp(-t)*exp(5*y)*sin(pi*x) = -u
    # d2u/dx2 = -pi^2 * u
    # d2u/dy2 = 25 * u
    # laplacian(u) = (25 - pi^2) * u
    # f = du/dt - kappa * laplacian(u) = -u - kappa*(25 - pi^2)*u
    f_ufl = (-1.0 - kappa_val * (25.0 - ufl.pi**2)) * ufl.exp(-t_const) * ufl.exp(5.0 * xs[1]) * ufl.sin(ufl.pi * xs[0])
    
    u_trial = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    u_n = fem.Function(V, name="u_n")
    
    # Initial condition: u(x,0) = exp(5*y)*sin(pi*x)
    u_ic_ufl = ufl.exp(5.0 * xs[1]) * ufl.sin(ufl.pi * xs[0])
    u_ic_expr = fem.Expression(u_ic_ufl, V.element.interpolation_points)
    u_n.interpolate(u_ic_expr)
    
    # Store initial condition
    u_initial_func = fem.Function(V)
    u_initial_func.interpolate(u_ic_expr)
    
    # Backward Euler weak form:
    # (u^{n+1} - u^n)/dt * v dx + kappa * grad(u^{n+1}) . grad(v) dx = f^{n+1} * v dx
    a = (u_trial / dt_const) * v * ufl.dx + kappa_c * ufl.inner(ufl.grad(u_trial), ufl.grad(v)) * ufl.dx
    L = (u_n / dt_const) * v * ufl.dx + f_ufl * v * ufl.dx
    
    # Boundary conditions - all boundaries with exact solution
    tdim = domain.topology.dim
    fdim = tdim - 1
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x_arr: np.ones(x_arr.shape[1], dtype=bool)
    )
    
    u_bc = fem.Function(V)
    u_exact_expr = fem.Expression(u_exact_ufl, V.element.interpolation_points)
    u_bc.interpolate(u_exact_expr)
    
    dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc = fem.dirichletbc(u_bc, dofs)
    bcs = [bc]
    
    # Compile forms
    a_form = fem.form(a)
    L_form = fem.form(L)
    
    # Assemble matrix (constant since coefficients don't change with time)
    A = petsc.assemble_matrix(a_form, bcs=bcs)
    A.assemble()
    
    # Create RHS vector
    b = petsc.create_vector(V)
    
    # Solution function
    u_sol = fem.Function(V, name="u_sol")
    
    # Setup KSP solver
    solver = PETSc.KSP().create(domain.comm)
    solver.setOperators(A)
    solver.setType(PETSc.KSP.Type.CG)
    pc = solver.getPC()
    pc.setType(PETSc.PC.Type.HYPRE)
    solver.setTolerances(rtol=1e-10, atol=1e-12, max_it=1000)
    solver.setUp()
    
    total_iterations = 0
    
    # Time stepping
    t_current = 0.0
    for step in range(n_steps):
        t_current += dt_use
        t_const.value = t_current
        
        # Update boundary condition
        u_bc.interpolate(u_exact_expr)
        
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
    
    # Evaluate on output grid
    x_out = np.linspace(0, 1, nx_out)
    y_out = np.linspace(0, 1, ny_out)
    X, Y = np.meshgrid(x_out, y_out, indexing='ij')
    
    points_3d = np.zeros((nx_out * ny_out, 3))
    points_3d[:, 0] = X.ravel()
    points_3d[:, 1] = Y.ravel()
    
    bb_tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, points_3d)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points_3d)
    
    u_values = np.full(points_3d.shape[0], np.nan)
    u_initial_values = np.full(points_3d.shape[0], np.nan)
    
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
        vals = u_sol.eval(pts_arr, cells_arr)
        u_values[eval_map] = vals.flatten()
        
        vals_init = u_initial_func.eval(pts_arr, cells_arr)
        u_initial_values[eval_map] = vals_init.flatten()
    
    u_grid = u_values.reshape((nx_out, ny_out))
    u_initial_grid = u_initial_values.reshape((nx_out, ny_out))
    
    # Cleanup
    solver.destroy()
    A.destroy()
    b.destroy()
    
    result = {
        "u": u_grid,
        "u_initial": u_initial_grid,
        "solver_info": {
            "mesh_resolution": N,
            "element_degree": degree,
            "ksp_type": "cg",
            "pc_type": "hypre",
            "rtol": 1e-10,
            "iterations": total_iterations,
            "dt": dt_use,
            "n_steps": n_steps,
            "time_scheme": "backward_euler",
        }
    }
    
    return result


if __name__ == "__main__":
    case_spec = {
        "pde": {
            "coefficients": {"kappa": 1.0},
            "time": {"t_end": 0.08, "dt": 0.008, "scheme": "backward_euler"},
        },
        "domain": {},
        "output": {"nx": 50, "ny": 50},
    }
    
    start = time_module.time()
    result = solve(case_spec)
    elapsed = time_module.time() - start
    
    u_grid = result["u"]
    
    t_end = 0.08
    nx_out, ny_out = 50, 50
    x_out = np.linspace(0, 1, nx_out)
    y_out = np.linspace(0, 1, ny_out)
    X, Y = np.meshgrid(x_out, y_out, indexing='ij')
    # Correct exact solution: u = exp(-t)*exp(5*y)*sin(pi*x)
    u_exact = np.exp(-t_end) * np.exp(5.0 * Y) * np.sin(np.pi * X)
    
    diff = u_grid - u_exact
    l2_err = np.sqrt(np.mean(diff**2))
    linf_err = np.max(np.abs(diff))
    
    print(f"Elapsed time: {elapsed:.3f}s")
    print(f"L2 error (discrete): {l2_err:.6e}")
    print(f"Linf error: {linf_err:.6e}")
    print(f"Grid shape: {u_grid.shape}")
    print(f"Total iterations: {result['solver_info']['iterations']}")
    print(f"dt used: {result['solver_info']['dt']}")
    print(f"n_steps: {result['solver_info']['n_steps']}")
    print(f"Solution range: [{np.nanmin(u_grid):.4f}, {np.nanmax(u_grid):.4f}]")
    print(f"Exact range: [{np.min(u_exact):.4f}, {np.max(u_exact):.4f}]")
