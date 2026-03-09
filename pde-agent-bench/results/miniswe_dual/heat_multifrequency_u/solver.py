import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc

def solve(case_spec: dict) -> dict:
    # Parse parameters
    pde = case_spec.get("pde", {})
    time_params = pde.get("time", {})
    coeffs = pde.get("coefficients", {})
    
    t_end = time_params.get("t_end", 0.1)
    dt_val = time_params.get("dt", 0.01)
    kappa = coeffs.get("kappa", 1.0)
    
    # Mesh and element parameters
    N = 80
    degree = 2
    
    # Use smaller dt for better temporal accuracy
    dt_val = 0.005
    n_steps = int(round(t_end / dt_val))
    dt_val = t_end / n_steps
    
    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
    
    V = fem.functionspace(domain, ("Lagrange", degree))
    
    x = ufl.SpatialCoordinate(domain)
    
    # Time as a constant
    t = fem.Constant(domain, PETSc.ScalarType(0.0))
    dt_c = fem.Constant(domain, PETSc.ScalarType(dt_val))
    kappa_c = fem.Constant(domain, PETSc.ScalarType(kappa))
    
    # Exact solution as UFL expression
    u_exact_ufl = ufl.exp(-t) * (ufl.sin(ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1]) 
                                   + 0.2 * ufl.sin(6*ufl.pi * x[0]) * ufl.sin(6*ufl.pi * x[1]))
    
    # Source term: f = du/dt - kappa * laplacian(u)
    # For u = exp(-t)*(sin(pi*x)*sin(pi*y) + 0.2*sin(6*pi*x)*sin(6*pi*y)):
    # du/dt = -u
    # laplacian(u) = exp(-t)*(-2*pi^2*sin(pi*x)*sin(pi*y) - 0.2*72*pi^2*sin(6*pi*x)*sin(6*pi*y))
    # f = du/dt - kappa*laplacian(u) = -u + kappa*exp(-t)*(2*pi^2*sin(pi*x)*sin(pi*y) + 0.2*72*pi^2*sin(6*pi*x)*sin(6*pi*y))
    f_expr = ufl.exp(-t) * (
        (-1.0 + 2.0 * kappa * ufl.pi**2) * ufl.sin(ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])
        + 0.2 * (-1.0 + 72.0 * kappa * ufl.pi**2) * ufl.sin(6*ufl.pi * x[0]) * ufl.sin(6*ufl.pi * x[1])
    )
    
    # Functions
    u_n = fem.Function(V, name="u_n")
    u_h = fem.Function(V, name="u_h")
    
    # Initial condition
    u_init_expr = fem.Expression(u_exact_ufl, V.element.interpolation_points)
    t.value = 0.0
    u_n.interpolate(u_init_expr)
    
    # Boundary condition (time-dependent)
    u_bc = fem.Function(V)
    bc_expr = fem.Expression(u_exact_ufl, V.element.interpolation_points)
    
    # Find boundary DOFs
    tdim = domain.topology.dim
    fdim = tdim - 1
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
    )
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc = fem.dirichletbc(u_bc, boundary_dofs)
    
    # Backward Euler weak form
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    
    a = (u * v / dt_c + kappa_c * ufl.inner(ufl.grad(u), ufl.grad(v))) * ufl.dx
    L = (u_n * v / dt_c + f_expr * v) * ufl.dx
    
    # Compile forms
    a_form = fem.form(a)
    L_form = fem.form(L)
    
    # Assemble matrix (constant in time for this problem)
    A = petsc.assemble_matrix(a_form, bcs=[bc])
    A.assemble()
    
    b = petsc.create_vector(V)
    
    # Setup solver
    solver = PETSc.KSP().create(domain.comm)
    solver.setOperators(A)
    solver.setType(PETSc.KSP.Type.CG)
    pc = solver.getPC()
    pc.setType(PETSc.PC.Type.HYPRE)
    solver.setTolerances(rtol=1e-10, atol=1e-12, max_it=2000)
    
    total_iterations = 0
    
    # Time stepping
    for step in range(n_steps):
        t.value = (step + 1) * dt_val
        
        # Update boundary condition
        u_bc.interpolate(bc_expr)
        
        # Assemble RHS
        with b.localForm() as loc:
            loc.set(0)
        petsc.assemble_vector(b, L_form)
        petsc.apply_lifting(b, [a_form], bcs=[[bc]])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        petsc.set_bc(b, [bc])
        
        # Solve
        solver.solve(b, u_h.x.petsc_vec)
        u_h.x.scatter_forward()
        
        total_iterations += solver.getIterationNumber()
        
        # Update for next step
        u_n.x.array[:] = u_h.x.array[:]
    
    # Evaluate on 50x50 grid
    nx_out, ny_out = 50, 50
    xs = np.linspace(0, 1, nx_out)
    ys = np.linspace(0, 1, ny_out)
    XX, YY = np.meshgrid(xs, ys, indexing='ij')
    points_2d = np.column_stack([XX.ravel(), YY.ravel()])
    points_3d = np.zeros((points_2d.shape[0], 3))
    points_3d[:, :2] = points_2d
    
    # Point evaluation
    bb_tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, points_3d)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points_3d)
    
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
        vals = u_h.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u_values[eval_map] = vals.flatten()
    
    u_grid = u_values.reshape((nx_out, ny_out))
    
    # Also compute initial condition on same grid
    t.value = 0.0
    u_init_func = fem.Function(V)
    u_init_func.interpolate(u_init_expr)
    
    u_init_values = np.full(points_3d.shape[0], np.nan)
    if len(points_on_proc) > 0:
        vals2 = u_init_func.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u_init_values[eval_map] = vals2.flatten()
    
    u_initial_grid = u_init_values.reshape((nx_out, ny_out))
    
    return {
        "u": u_grid,
        "u_initial": u_initial_grid,
        "solver_info": {
            "mesh_resolution": N,
            "element_degree": degree,
            "ksp_type": "cg",
            "pc_type": "hypre",
            "rtol": 1e-10,
            "iterations": total_iterations,
            "dt": dt_val,
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
                "dt": 0.01,
                "scheme": "backward_euler"
            },
            "coefficients": {
                "kappa": 1.0
            }
        }
    }
    
    start = time.time()
    result = solve(case_spec)
    elapsed = time.time() - start
    
    print(f"Elapsed time: {elapsed:.3f}s")
    print(f"Solution shape: {result['u'].shape}")
    print(f"Solution range: [{np.nanmin(result['u']):.6f}, {np.nanmax(result['u']):.6f}]")
    print(f"Solver info: {result['solver_info']}")
    
    # Verify against exact solution
    nx_out, ny_out = 50, 50
    xs = np.linspace(0, 1, nx_out)
    ys = np.linspace(0, 1, ny_out)
    XX, YY = np.meshgrid(xs, ys, indexing='ij')
    t_end = 0.1
    u_exact = np.exp(-t_end) * (np.sin(np.pi*XX)*np.sin(np.pi*YY) + 0.2*np.sin(6*np.pi*XX)*np.sin(6*np.pi*YY))
    
    error = np.sqrt(np.nanmean((result['u'] - u_exact)**2))
    max_error = np.nanmax(np.abs(result['u'] - u_exact))
    print(f"RMS error: {error:.6e}")
    print(f"Max error: {max_error:.6e}")
    print(f"NaN count: {np.sum(np.isnan(result['u']))}")
