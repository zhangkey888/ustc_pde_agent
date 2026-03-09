import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc

def solve(case_spec: dict) -> dict:
    """Solve the transient heat equation with FEniCSx."""
    
    # Extract parameters with defaults
    pde = case_spec.get("pde", {})
    time_params = pde.get("time", {})
    
    t_end = float(time_params.get("t_end", 0.1))
    dt_val = float(time_params.get("dt", 0.02))
    scheme = time_params.get("scheme", "backward_euler")
    
    kappa = 1.0
    coeffs = pde.get("coefficients", {})
    if "kappa" in coeffs:
        kappa = float(coeffs["kappa"])
    
    # Mesh resolution - start moderate for speed
    N = 64
    degree = 1
    
    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
    
    V = fem.functionspace(domain, ("Lagrange", degree))
    
    # Define functions
    u_n = fem.Function(V, name="u_n")  # solution at previous time step
    u_h = fem.Function(V, name="u_h")  # solution at current time step (for output)
    
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    
    # Initial condition: u0 = 0
    u_n.x.array[:] = 0.0
    
    # Source term
    x = ufl.SpatialCoordinate(domain)
    f = ufl.exp(-220.0 * ((x[0] - 0.25)**2 + (x[1] - 0.25)**2)) + \
        ufl.exp(-220.0 * ((x[0] - 0.75)**2 + (x[1] - 0.7)**2))
    
    # Time step as constant
    dt_c = fem.Constant(domain, PETSc.ScalarType(dt_val))
    kappa_c = fem.Constant(domain, PETSc.ScalarType(kappa))
    
    # Backward Euler weak form:
    # (u - u_n)/dt - kappa * div(grad(u)) = f
    # => (u, v)/dt + kappa*(grad(u), grad(v)) = (u_n, v)/dt + (f, v)
    a = ufl.inner(u, v) / dt_c * ufl.dx + kappa_c * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = ufl.inner(u_n, v) / dt_c * ufl.dx + ufl.inner(f, v) * ufl.dx
    
    # Boundary conditions: homogeneous Dirichlet
    tdim = domain.topology.dim
    fdim = tdim - 1
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
    )
    dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc = fem.dirichletbc(PETSc.ScalarType(0.0), dofs, V)
    bcs = [bc]
    
    # Compile forms
    a_form = fem.form(a)
    L_form = fem.form(L)
    
    # Assemble matrix (constant in time for this problem)
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
    solver.setTolerances(rtol=1e-8, atol=1e-12, max_it=1000)
    solver.setUp()
    
    # Time stepping
    t = 0.0
    n_steps = 0
    total_iterations = 0
    
    while t < t_end - 1e-14:
        t += dt_val
        n_steps += 1
        
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
    
    # Store initial condition for output
    u_initial_func = fem.Function(V)
    u_initial_func.x.array[:] = 0.0
    
    # Evaluate on 50x50 grid
    nx_out, ny_out = 50, 50
    xs = np.linspace(0.0, 1.0, nx_out)
    ys = np.linspace(0.0, 1.0, ny_out)
    XX, YY = np.meshgrid(xs, ys, indexing='ij')
    points_2d = np.column_stack([XX.ravel(), YY.ravel()])
    # dolfinx needs 3D points
    points_3d = np.zeros((points_2d.shape[0], 3))
    points_3d[:, :2] = points_2d
    
    # Build bounding box tree
    bb_tree = geometry.bb_tree(domain, domain.topology.dim)
    
    cell_candidates = geometry.compute_collisions_points(bb_tree, points_3d)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points_3d)
    
    u_values = np.full(points_3d.shape[0], np.nan)
    u0_values = np.full(points_3d.shape[0], 0.0)
    
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
    
    u_grid = u_values.reshape((nx_out, ny_out))
    u0_grid = u0_values.reshape((nx_out, ny_out))
    
    # Clean up PETSc objects
    solver.destroy()
    A.destroy()
    
    return {
        "u": u_grid,
        "u_initial": u0_grid,
        "solver_info": {
            "mesh_resolution": N,
            "element_degree": degree,
            "ksp_type": "cg",
            "pc_type": "hypre",
            "rtol": 1e-8,
            "iterations": total_iterations,
            "dt": dt_val,
            "n_steps": n_steps,
            "time_scheme": scheme,
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
            "source": "exp(-220*((x-0.25)**2 + (y-0.25)**2)) + exp(-220*((x-0.75)**2 + (y-0.7)**2))",
            "initial_condition": "0.0",
            "boundary_condition": "0.0"
        }
    }
    
    t0 = time.time()
    result = solve(case_spec)
    elapsed = time.time() - t0
    
    print(f"Solve time: {elapsed:.3f}s")
    print(f"Solution shape: {result['u'].shape}")
    print(f"Solution range: [{np.nanmin(result['u']):.6f}, {np.nanmax(result['u']):.6f}]")
    print(f"NaN count: {np.isnan(result['u']).sum()}")
    print(f"Solver info: {result['solver_info']}")
