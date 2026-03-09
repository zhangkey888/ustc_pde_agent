import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc
import time

ScalarType = PETSc.ScalarType


def solve(case_spec: dict) -> dict:
    """Solve the transient heat equation using backward Euler."""
    
    # Extract parameters from case_spec
    pde = case_spec.get("pde", {})
    coeffs = pde.get("coefficients", {})
    kappa = coeffs.get("kappa", 1.0)
    
    time_params = pde.get("time", {})
    t_end = time_params.get("t_end", 0.06)
    dt = time_params.get("dt", 0.003)
    scheme = time_params.get("scheme", "backward_euler")
    
    # Output grid
    output = case_spec.get("output", {})
    nx_out = output.get("nx", 50)
    ny_out = output.get("ny", 50)
    
    # Use P2 elements for better accuracy with moderate mesh
    element_degree = 2
    N = 64

    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
    
    V = fem.functionspace(domain, ("Lagrange", element_degree))
    
    # Spatial coordinates
    x = ufl.SpatialCoordinate(domain)
    
    # Time variable as a constant (will be updated)
    t_const = fem.Constant(domain, ScalarType(0.0))
    dt_const = fem.Constant(domain, ScalarType(dt))
    kappa_const = fem.Constant(domain, ScalarType(kappa))
    
    pi = ufl.pi
    
    # Source term derived from manufactured solution u = exp(-t)*sin(4*pi*x)*sin(4*pi*y)
    # f = du/dt - kappa*laplacian(u) = exp(-t)*sin(4*pi*x)*sin(4*pi*y)*(-1 + 32*kappa*pi^2)
    f_ufl = ufl.exp(-t_const) * ufl.sin(4 * pi * x[0]) * ufl.sin(4 * pi * x[1]) * (-1.0 + 32.0 * kappa * pi**2)
    
    # Trial and test functions
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    
    # Solution at previous time step
    u_n = fem.Function(V)
    u_h = fem.Function(V)
    
    # Initial condition: u(x, 0) = sin(4*pi*x)*sin(4*pi*y)
    u_n.interpolate(lambda x_arr: np.sin(4 * np.pi * x_arr[0]) * np.sin(4 * np.pi * x_arr[1]))
    
    # Store initial condition for output
    u_initial_func = fem.Function(V)
    u_initial_func.interpolate(lambda x_arr: np.sin(4 * np.pi * x_arr[0]) * np.sin(4 * np.pi * x_arr[1]))
    
    # Backward Euler weak form:
    # (u - u_n)/dt - kappa*laplacian(u) = f(t_{n+1})
    # => (u, v)/dt + kappa*(grad(u), grad(v)) = (u_n, v)/dt + (f, v)
    a = (u * v / dt_const + kappa_const * ufl.inner(ufl.grad(u), ufl.grad(v))) * ufl.dx
    L = (u_n * v / dt_const + f_ufl * v) * ufl.dx
    
    # Boundary conditions
    tdim = domain.topology.dim
    fdim = tdim - 1
    
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x_arr: np.ones(x_arr.shape[1], dtype=bool)
    )
    
    u_bc = fem.Function(V)
    dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc = fem.dirichletbc(u_bc, dofs)
    bcs = [bc]
    
    # Compile forms
    a_form = fem.form(a)
    L_form = fem.form(L)
    
    # Assemble matrix (constant in time)
    A = petsc.assemble_matrix(a_form, bcs=bcs)
    A.assemble()
    
    # Create RHS vector
    b = fem.Function(V)
    b_vec = b.x.petsc_vec
    
    # Setup solver
    solver = PETSc.KSP().create(domain.comm)
    solver.setOperators(A)
    solver.setType(PETSc.KSP.Type.CG)
    pc = solver.getPC()
    pc.setType(PETSc.PC.Type.HYPRE)
    solver.setTolerances(rtol=1e-10, atol=1e-12, max_it=1000)
    solver.setUp()
    
    # Time stepping
    t = 0.0
    n_steps = int(np.round(t_end / dt))
    total_iterations = 0
    
    for step in range(n_steps):
        t += dt
        t_const.value = t
        
        # Update boundary condition
        t_val = float(t)
        u_bc.interpolate(lambda x_arr, tv=t_val: np.exp(-tv) * np.sin(4 * np.pi * x_arr[0]) * np.sin(4 * np.pi * x_arr[1]))
        
        # Assemble RHS
        with b_vec.localForm() as loc:
            loc.set(0.0)
        petsc.assemble_vector(b_vec, L_form)
        petsc.apply_lifting(b_vec, [a_form], bcs=[bcs])
        b_vec.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        petsc.set_bc(b_vec, bcs)
        
        # Solve
        solver.solve(b_vec, u_h.x.petsc_vec)
        u_h.x.scatter_forward()
        
        total_iterations += solver.getIterationNumber()
        
        # Update previous solution
        u_n.x.array[:] = u_h.x.array[:]
    
    # Evaluate solution on output grid
    x_out = np.linspace(0, 1, nx_out)
    y_out = np.linspace(0, 1, ny_out)
    X, Y = np.meshgrid(x_out, y_out, indexing='ij')
    
    points_2d = np.column_stack([X.ravel(), Y.ravel()])
    points_3d = np.zeros((points_2d.shape[0], 3))
    points_3d[:, :2] = points_2d
    
    # Point evaluation
    bb_tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, points_3d)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points_3d)
    
    u_values = np.full(points_3d.shape[0], np.nan)
    u_init_values = np.full(points_3d.shape[0], np.nan)
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
        vals = u_h.eval(pts_arr, cells_arr)
        u_values[eval_map] = vals.flatten()
        vals_init = u_initial_func.eval(pts_arr, cells_arr)
        u_init_values[eval_map] = vals_init.flatten()
    
    u_grid = u_values.reshape((nx_out, ny_out))
    u_initial_grid = u_init_values.reshape((nx_out, ny_out))
    
    # Cleanup
    solver.destroy()
    A.destroy()
    
    return {
        "u": u_grid,
        "u_initial": u_initial_grid,
        "solver_info": {
            "mesh_resolution": N,
            "element_degree": element_degree,
            "ksp_type": "cg",
            "pc_type": "hypre",
            "rtol": 1e-10,
            "iterations": total_iterations,
            "dt": dt,
            "n_steps": n_steps,
            "time_scheme": "backward_euler",
        }
    }


if __name__ == "__main__":
    case_spec = {
        "pde": {
            "coefficients": {"kappa": 1.0},
            "time": {"t_end": 0.06, "dt": 0.003, "scheme": "backward_euler"},
        },
        "domain": {"type": "unit_square"},
        "output": {"nx": 50, "ny": 50},
    }
    
    t0 = time.time()
    result = solve(case_spec)
    elapsed = time.time() - t0
    
    u_grid = result["u"]
    print(f"Solution shape: {u_grid.shape}")
    print(f"Solution range: [{np.nanmin(u_grid):.6f}, {np.nanmax(u_grid):.6f}]")
    print(f"NaN count: {np.isnan(u_grid).sum()}")
    print(f"Wall time: {elapsed:.3f}s")
    print(f"Iterations: {result['solver_info']['iterations']}")
    print(f"Steps: {result['solver_info']['n_steps']}")
    
    # Compare with exact solution at t_end
    t_end = 0.06
    x_out = np.linspace(0, 1, 50)
    y_out = np.linspace(0, 1, 50)
    X, Y = np.meshgrid(x_out, y_out, indexing='ij')
    u_exact = np.exp(-t_end) * np.sin(4 * np.pi * X) * np.sin(4 * np.pi * Y)
    
    rms_error = np.sqrt(np.mean((u_grid - u_exact)**2))
    max_error = np.max(np.abs(u_grid - u_exact))
    print(f"RMS error: {rms_error:.6e} (limit: 8.90e-03)")
    print(f"Max error: {max_error:.6e}")
    print(f"PASS: {rms_error <= 8.90e-3 and elapsed <= 30.589}")
