import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc
import time

def solve(case_spec: dict) -> dict:
    """Solve the transient heat equation."""
    
    start_time = time.time()
    
    # Parse case_spec
    pde = case_spec.get("pde", {})
    
    # Time parameters - hardcoded defaults as fallback
    time_spec = pde.get("time", {})
    t_end = float(time_spec.get("t_end", 0.1))
    dt_suggested = float(time_spec.get("dt", 0.02))
    scheme = time_spec.get("scheme", "backward_euler")
    
    # Coefficients
    coeffs = pde.get("coefficients", {})
    kappa_val = float(coeffs.get("kappa", 1.0))
    
    # Mesh resolution
    N = 64
    element_degree = 1
    dt = dt_suggested
    
    # Create mesh
    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
    
    # Function space
    V = fem.functionspace(domain, ("Lagrange", element_degree))
    
    # Spatial coordinates
    x = ufl.SpatialCoordinate(domain)
    
    # Source term
    f_expr = ufl.exp(-200.0 * ((x[0] - 0.35)**2 + (x[1] - 0.65)**2))
    
    # Diffusion coefficient
    kappa = fem.Constant(domain, PETSc.ScalarType(kappa_val))
    
    # Time step
    n_steps = int(np.ceil(t_end / dt))
    actual_dt = t_end / n_steps
    dt_const = fem.Constant(domain, PETSc.ScalarType(actual_dt))
    
    # Trial and test functions
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    
    # Previous solution
    u_n = fem.Function(V, name="u_n")
    u_n.interpolate(lambda X: np.sin(np.pi * X[0]) * np.sin(np.pi * X[1]))
    
    # Store initial condition
    u_initial_func = fem.Function(V)
    u_initial_func.interpolate(lambda X: np.sin(np.pi * X[0]) * np.sin(np.pi * X[1]))
    
    # Solution function
    u_sol = fem.Function(V, name="u")
    
    # Boundary conditions: u = 0 on boundary
    tdim = domain.topology.dim
    fdim = tdim - 1
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda X: np.ones(X.shape[1], dtype=bool)
    )
    dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc = fem.dirichletbc(PETSc.ScalarType(0.0), dofs, V)
    bcs = [bc]
    
    # Backward Euler weak form
    a = (u / dt_const) * v * ufl.dx + kappa * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = (u_n / dt_const) * v * ufl.dx + f_expr * v * ufl.dx
    
    # Compile forms
    a_form = fem.form(a)
    L_form = fem.form(L)
    
    # Assemble matrix
    A = petsc.assemble_matrix(a_form, bcs=bcs)
    A.assemble()
    
    # Create RHS vector using function space
    b = petsc.create_vector(V)
    
    # Setup KSP solver
    ksp_type_str = "cg"
    pc_type_str = "hypre"
    rtol_val = 1e-8
    
    solver = PETSc.KSP().create(domain.comm)
    solver.setOperators(A)
    solver.setType(ksp_type_str)
    pc = solver.getPC()
    pc.setType(pc_type_str)
    solver.setTolerances(rtol=rtol_val, atol=1e-12, max_it=1000)
    solver.setUp()
    
    # Time stepping
    total_iterations = 0
    t = 0.0
    
    for step in range(n_steps):
        t += actual_dt
        
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
    
    # Evaluate on 50x50 grid
    nx_out, ny_out = 50, 50
    xs = np.linspace(0.0, 1.0, nx_out)
    ys = np.linspace(0.0, 1.0, ny_out)
    XX, YY = np.meshgrid(xs, ys, indexing='ij')
    
    points_3d = np.zeros((nx_out * ny_out, 3))
    points_3d[:, 0] = XX.ravel()
    points_3d[:, 1] = YY.ravel()
    
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
        pts = np.array(points_on_proc)
        cls = np.array(cells_on_proc, dtype=np.int32)
        vals = u_sol.eval(pts, cls)
        u_values[eval_map] = vals.flatten()
        vals_init = u_initial_func.eval(pts, cls)
        u_init_values[eval_map] = vals_init.flatten()
    
    u_grid = u_values.reshape((nx_out, ny_out))
    u_initial_grid = u_init_values.reshape((nx_out, ny_out))
    
    result = {
        "u": u_grid,
        "u_initial": u_initial_grid,
        "solver_info": {
            "mesh_resolution": N,
            "element_degree": element_degree,
            "ksp_type": ksp_type_str,
            "pc_type": pc_type_str,
            "rtol": rtol_val,
            "iterations": total_iterations,
            "dt": actual_dt,
            "n_steps": n_steps,
            "time_scheme": "backward_euler",
        }
    }
    
    return result


if __name__ == "__main__":
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
            "source": "exp(-200*((x-0.35)**2 + (y-0.65)**2))",
            "initial_condition": "sin(pi*x)*sin(pi*y)",
        },
        "domain": {
            "type": "unit_square",
            "dim": 2
        }
    }
    
    t0 = time.time()
    result = solve(case_spec)
    elapsed = time.time() - t0
    
    u_grid = result["u"]
    print(f"Solution shape: {u_grid.shape}")
    print(f"Solution min: {np.nanmin(u_grid):.6f}, max: {np.nanmax(u_grid):.6f}")
    print(f"NaN count: {np.isnan(u_grid).sum()}")
    print(f"Wall time: {elapsed:.3f}s")
    print(f"Solver info: {result['solver_info']}")
