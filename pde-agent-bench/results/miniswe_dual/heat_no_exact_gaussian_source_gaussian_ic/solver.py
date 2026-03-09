import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc

ScalarType = PETSc.ScalarType


def solve(case_spec: dict) -> dict:
    """Solve the transient heat equation."""
    
    # Extract parameters from case_spec - handle both formats
    # Format 1: case_spec['pde']['time'] etc.
    # Format 2: case_spec['oracle_config']['pde']['time'] etc.
    pde = case_spec.get("pde", {})
    if not pde:
        pde = case_spec.get("oracle_config", {}).get("pde", {})
    
    time_params = pde.get("time", {})
    
    # Hardcoded defaults as fallback (from problem description)
    t_end = float(time_params.get("t_end", 0.1))
    dt = float(time_params.get("dt", 0.02))
    scheme = time_params.get("scheme", "backward_euler")
    
    # Extract kappa - handle nested format
    coeffs = pde.get("coefficients", {})
    kappa_val = coeffs.get("kappa", 1.0)
    if isinstance(kappa_val, dict):
        kappa = float(kappa_val.get("value", 1.0))
    else:
        kappa = float(kappa_val)
    
    # Force transient
    is_transient = True
    
    # Mesh resolution and element degree
    N = 64
    element_degree = 1
    
    # Create mesh
    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
    
    # Function space
    V = fem.functionspace(domain, ("Lagrange", element_degree))
    
    # Spatial coordinates
    x = ufl.SpatialCoordinate(domain)
    
    # Source term: f = exp(-200*((x-0.3)**2 + (y-0.7)**2))
    f_expr = ufl.exp(-200.0 * ((x[0] - 0.3)**2 + (x[1] - 0.7)**2))
    
    # Initial condition: u0 = exp(-120*((x-0.6)**2 + (y-0.4)**2))
    u_n = fem.Function(V, name="u_n")
    u_n.interpolate(lambda X: np.exp(-120.0 * ((X[0] - 0.6)**2 + (X[1] - 0.4)**2)))
    
    # Store initial condition for output
    u_initial_func = fem.Function(V)
    u_initial_func.interpolate(lambda X: np.exp(-120.0 * ((X[0] - 0.6)**2 + (X[1] - 0.4)**2)))
    
    # Boundary condition: u = 0 on all boundaries
    tdim = domain.topology.dim
    fdim = tdim - 1
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda X: np.ones(X.shape[1], dtype=bool)
    )
    dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc = fem.dirichletbc(ScalarType(0.0), dofs, V)
    bcs = [bc]
    
    # Trial and test functions
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    
    # Backward Euler time stepping
    dt_const = fem.Constant(domain, ScalarType(dt))
    kappa_const = fem.Constant(domain, ScalarType(kappa))
    
    a = (u * v / dt_const + kappa_const * ufl.inner(ufl.grad(u), ufl.grad(v))) * ufl.dx
    L = (f_expr * v + u_n * v / dt_const) * ufl.dx
    
    # Compile forms
    a_form = fem.form(a)
    L_form = fem.form(L)
    
    # Time stepping setup
    n_steps = int(np.ceil(t_end / dt))
    actual_dt = t_end / n_steps
    dt_const.value = actual_dt
    
    # Assemble matrix (constant for all time steps since dt and kappa don't change)
    A = petsc.assemble_matrix(a_form, bcs=bcs)
    A.assemble()
    
    # Setup solver
    solver = PETSc.KSP().create(domain.comm)
    solver.setOperators(A)
    solver.setType(PETSc.KSP.Type.CG)
    pc = solver.getPC()
    pc.setType(PETSc.PC.Type.HYPRE)
    solver.setTolerances(rtol=1e-8, atol=1e-12, max_it=1000)
    solver.setUp()
    
    # Solution function
    u_sol = fem.Function(V, name="u")
    u_sol.x.array[:] = u_n.x.array[:]
    
    total_iterations = 0
    
    for step in range(n_steps):
        # Assemble RHS
        b_vec = petsc.assemble_vector(L_form)
        petsc.apply_lifting(b_vec, [a_form], bcs=[bcs])
        b_vec.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        petsc.set_bc(b_vec, bcs)
        
        # Solve
        solver.solve(b_vec, u_sol.x.petsc_vec)
        u_sol.x.scatter_forward()
        
        total_iterations += solver.getIterationNumber()
        
        # Update previous solution
        u_n.x.array[:] = u_sol.x.array[:]
        
        b_vec.destroy()
    
    # Evaluate solution on 50x50 grid
    nx_out, ny_out = 50, 50
    xs = np.linspace(0, 1, nx_out)
    ys = np.linspace(0, 1, ny_out)
    XX, YY = np.meshgrid(xs, ys, indexing='ij')
    points = np.zeros((3, nx_out * ny_out))
    points[0, :] = XX.flatten()
    points[1, :] = YY.flatten()
    
    # Point evaluation
    bb_tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, points.T)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points.T)
    
    u_values = np.zeros(nx_out * ny_out)
    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    for i in range(nx_out * ny_out):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(points[:, i])
            cells_on_proc.append(links[0])
            eval_map.append(i)
    
    if len(points_on_proc) > 0:
        pts_arr = np.array(points_on_proc)
        cells_arr = np.array(cells_on_proc, dtype=np.int32)
        vals = u_sol.eval(pts_arr, cells_arr)
        u_values[eval_map] = vals.flatten()
    
    u_grid = u_values.reshape((nx_out, ny_out))
    
    # Also evaluate initial condition on same grid
    u_init_values = np.zeros(nx_out * ny_out)
    if len(points_on_proc) > 0:
        vals_init = u_initial_func.eval(pts_arr, cells_arr)
        u_init_values[eval_map] = vals_init.flatten()
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
            "rtol": 1e-8,
            "iterations": total_iterations,
            "dt": actual_dt,
            "n_steps": n_steps,
            "time_scheme": "backward_euler",
        }
    }


if __name__ == "__main__":
    import time
    
    case_spec = {
        "pde": {
            "type": "heat",
            "time": {
                "t_end": 0.1,
                "dt": 0.02,
                "scheme": "backward_euler"
            },
            "coefficients": {
                "kappa": 1.0
            },
            "source_term": "exp(-200*((x-0.3)**2 + (y-0.7)**2))",
            "initial_condition": "exp(-120*((x-0.6)**2 + (y-0.4)**2))"
        },
        "domain": {
            "type": "unit_square",
            "dim": 2
        }
    }
    
    start = time.time()
    result = solve(case_spec)
    elapsed = time.time() - start
    
    print(f"Solve time: {elapsed:.3f}s (limit: 15.69s)")
    print(f"Solution shape: {result['u'].shape}")
    print(f"u min={result['u'].min():.6e}, max={result['u'].max():.6e}")
    print(f"u mean={result['u'].mean():.6e}")
    print(f"Solver info: {result['solver_info']}")
