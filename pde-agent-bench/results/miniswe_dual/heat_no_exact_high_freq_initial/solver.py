import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc


def solve(case_spec: dict = None):
    """Solve the transient heat equation."""
    
    # Parse case_spec or use defaults
    if case_spec is None:
        case_spec = {}
    
    pde = case_spec.get("pde", {})
    time_params = pde.get("time", {})
    coeffs = pde.get("coefficients", {})
    
    # Time parameters with hardcoded defaults
    t_end = time_params.get("t_end", 0.12)
    dt = time_params.get("dt", 0.02)
    scheme = time_params.get("scheme", "backward_euler")
    
    # Coefficient
    kappa = coeffs.get("kappa", 1.0)
    if isinstance(kappa, str):
        kappa = float(kappa)
    
    # Mesh resolution
    N = 64
    element_degree = 1
    
    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
    
    V = fem.functionspace(domain, ("Lagrange", element_degree))
    
    # Initial condition: u0 = sin(6*pi*x)*sin(6*pi*y)
    u_n = fem.Function(V, name="u_n")
    u_n.interpolate(lambda x: np.sin(6 * np.pi * x[0]) * np.sin(6 * np.pi * x[1]))
    
    # Build evaluation grid
    nx_out, ny_out = 50, 50
    xs = np.linspace(0, 1, nx_out)
    ys = np.linspace(0, 1, ny_out)
    XX, YY = np.meshgrid(xs, ys, indexing='ij')
    points_2d = np.column_stack([XX.ravel(), YY.ravel()])
    points_3d = np.zeros((points_2d.shape[0], 3))
    points_3d[:, 0] = points_2d[:, 0]
    points_3d[:, 1] = points_2d[:, 1]
    
    # Point evaluation setup
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
    
    points_on_proc_arr = np.array(points_on_proc) if points_on_proc else np.empty((0, 3))
    cells_on_proc_arr = np.array(cells_on_proc, dtype=np.int32) if cells_on_proc else np.empty(0, dtype=np.int32)
    
    # Evaluate initial condition on grid
    u_initial_vals = np.full(points_3d.shape[0], np.nan)
    if len(points_on_proc) > 0:
        vals = u_n.eval(points_on_proc_arr, cells_on_proc_arr)
        u_initial_vals[eval_map] = vals.flatten()
    u_initial_grid = u_initial_vals.reshape((nx_out, ny_out))
    
    # Boundary conditions: u = 0 on boundary
    tdim = domain.topology.dim
    fdim = tdim - 1
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
    )
    dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc = fem.dirichletbc(PETSc.ScalarType(0.0), dofs, V)
    bcs = [bc]
    
    # Variational form for backward Euler
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    
    dt_const = fem.Constant(domain, PETSc.ScalarType(dt))
    kappa_const = fem.Constant(domain, PETSc.ScalarType(kappa))
    
    # f = 0 (no source term specified)
    f = fem.Constant(domain, PETSc.ScalarType(0.0))
    
    a = (u * v / dt_const + kappa_const * ufl.inner(ufl.grad(u), ufl.grad(v))) * ufl.dx
    L = (u_n * v / dt_const + f * v) * ufl.dx
    
    # Compile forms
    a_form = fem.form(a)
    L_form = fem.form(L)
    
    # Assemble matrix (constant in time)
    A = petsc.assemble_matrix(a_form, bcs=bcs)
    A.assemble()
    
    # Create RHS vector
    b_vec = petsc.create_vector(V)
    
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
    
    # Time stepping
    t = 0.0
    n_steps = int(round(t_end / dt))
    total_iterations = 0
    
    for step in range(n_steps):
        t += dt
        
        # Assemble RHS
        with b_vec.localForm() as loc:
            loc.set(0)
        petsc.assemble_vector(b_vec, L_form)
        petsc.apply_lifting(b_vec, [a_form], bcs=[bcs])
        b_vec.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        petsc.set_bc(b_vec, bcs)
        
        # Solve
        solver.solve(b_vec, u_sol.x.petsc_vec)
        u_sol.x.scatter_forward()
        
        total_iterations += solver.getIterationNumber()
        
        # Update previous solution
        u_n.x.array[:] = u_sol.x.array[:]
    
    # Evaluate solution on output grid
    u_vals = np.full(points_3d.shape[0], np.nan)
    if len(points_on_proc) > 0:
        vals = u_sol.eval(points_on_proc_arr, cells_on_proc_arr)
        u_vals[eval_map] = vals.flatten()
    u_grid = u_vals.reshape((nx_out, ny_out))
    
    # Clean up
    solver.destroy()
    A.destroy()
    b_vec.destroy()
    
    result = {
        "u": u_grid,
        "u_initial": u_initial_grid,
        "solver_info": {
            "mesh_resolution": N,
            "element_degree": element_degree,
            "ksp_type": "cg",
            "pc_type": "hypre",
            "rtol": 1e-8,
            "iterations": total_iterations,
            "dt": dt,
            "n_steps": n_steps,
            "time_scheme": "backward_euler",
        }
    }
    
    return result


if __name__ == "__main__":
    import time
    t0 = time.time()
    result = solve()
    elapsed = time.time() - t0
    print(f"Solve time: {elapsed:.3f}s")
    print(f"Solution shape: {result['u'].shape}")
    print(f"Solution min: {np.nanmin(result['u']):.6e}, max: {np.nanmax(result['u']):.6e}")
    print(f"Solver info: {result['solver_info']}")
    print(f"Any NaN in solution: {np.any(np.isnan(result['u']))}")
    
    # Analytical check: for heat equation with u0 = sin(6πx)sin(6πy), κ=1, f=0
    # Exact solution: u(x,y,t) = sin(6πx)sin(6πy) * exp(-72π²t)
    # At t=0.12: exp(-72π²*0.12) = exp(-8.64π²) ≈ exp(-85.27) ≈ very small
    t_end = 0.12
    decay = np.exp(-72 * np.pi**2 * t_end)
    print(f"Analytical decay factor at t={t_end}: {decay:.6e}")
    print(f"Expected solution should be essentially zero everywhere")
