import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
import ufl
from dolfinx.fem import petsc
from petsc4py import PETSc
import time

ScalarType = PETSc.ScalarType

def probe_points(u_func, domain, points_array):
    """
    Evaluate function at points_array (shape (3, N)).
    Returns array of length N with values.
    """
    bb_tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, points_array.T)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points_array.T)
    
    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    for i in range(points_array.shape[1]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(points_array.T[i])
            cells_on_proc.append(links[0])
            eval_map.append(i)
    
    u_values = np.full((points_array.shape[1],), np.nan, dtype=ScalarType)
    if len(points_on_proc) > 0:
        vals = u_func.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u_values[eval_map] = vals.flatten()
    return u_values

def solve_heat_on_mesh(domain, degree, dt, t_end, case_spec):
    """Solve transient heat equation on given mesh with backward Euler."""
    comm = domain.comm
    
    # Function space
    V = fem.functionspace(domain, ("Lagrange", degree))
    
    # Trial and test functions
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    
    # Coefficients
    kappa = fem.Constant(domain, ScalarType(1.0))  # κ = 1.0
    
    # Source term f = exp(-200*((x-0.35)**2 + (y-0.65)**2))
    x = ufl.SpatialCoordinate(domain)
    f_expr = ufl.exp(-200*((x[0]-0.35)**2 + (x[1]-0.65)**2))
    f = fem.Function(V)
    f.interpolate(fem.Expression(f_expr, V.element.interpolation_points))
    
    # Initial condition u0 = sin(pi*x)*sin(pi*y)
    u0_expr = ufl.sin(ufl.pi*x[0]) * ufl.sin(ufl.pi*x[1])
    u_n = fem.Function(V)
    u_n.interpolate(fem.Expression(u0_expr, V.element.interpolation_points))
    
    # Boundary condition: u = 0 on entire boundary (Dirichlet)
    tdim = domain.topology.dim
    fdim = tdim - 1
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
    )
    dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    u_bc = fem.Function(V)
    u_bc.interpolate(lambda x: np.zeros_like(x[0]))
    bc = fem.dirichletbc(u_bc, dofs)
    
    # Time-stepping parameters
    dt_val = ScalarType(dt)
    t = 0.0
    n_steps = int(np.round(t_end / dt))
    if n_steps == 0:
        n_steps = 1
        dt_val = ScalarType(t_end)
    
    # Variational forms for backward Euler
    a = ufl.inner(u, v) * ufl.dx + dt_val * ufl.inner(kappa * ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = ufl.inner(u_n, v) * ufl.dx + dt_val * ufl.inner(f, v) * ufl.dx
    
    # Assemble once (matrix constant in time)
    a_form = fem.form(a)
    L_form = fem.form(L)
    A = petsc.assemble_matrix(a_form, bcs=[bc])
    A.assemble()
    
    # Create solver
    solver = PETSc.KSP().create(comm)
    solver.setOperators(A)
    
    # Try iterative solver first, fallback to direct
    try:
        solver.setType(PETSc.KSP.Type.GMRES)
        solver.getPC().setType(PETSc.PC.Type.HYPRE)
        solver.setTolerances(rtol=1e-8, max_it=1000)
        solver.setFromOptions()
        # Test solve with zero RHS to check convergence
        test_b = petsc.create_vector(L_form.function_spaces)
        test_x = petsc.create_vector(L_form.function_spaces)
        solver.solve(test_b, test_x)
        ksp_type = "gmres"
        pc_type = "hypre"
    except Exception:
        # Fallback to direct solver
        solver.setType(PETSc.KSP.Type.PREONLY)
        solver.getPC().setType(PETSc.PC.Type.LU)
        ksp_type = "preonly"
        pc_type = "lu"
    
    solver.setTolerances(rtol=1e-8)
    rtol = 1e-8
    
    # Time stepping
    u_sol = fem.Function(V)
    total_iterations = 0
    
    for step in range(n_steps):
        # Assemble RHS
        b = petsc.create_vector(L_form.function_spaces)
        with b.localForm() as loc:
            loc.set(0)
        petsc.assemble_vector(b, L_form)
        petsc.apply_lifting(b, [a_form], bcs=[[bc]])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        petsc.set_bc(b, [bc])
        
        # Solve
        solver.solve(b, u_sol.x.petsc_vec)
        u_sol.x.scatter_forward()
        
        # Get iteration count
        total_iterations += solver.getIterationNumber()
        
        # Update for next step
        u_n.x.array[:] = u_sol.x.array
        t += dt_val
    
    # Compute L2 norm of solution for convergence check
    norm = fem.assemble_scalar(fem.form(ufl.inner(u_sol, u_sol) * ufl.dx))
    norm = np.sqrt(comm.allreduce(norm, op=MPI.SUM))
    
    # Determine N (cells per dimension) from domain
    # For unit square created with create_unit_square(N, N, ...)
    # We can approximate by sqrt(total cells/2) for triangles
    total_cells = domain.topology.index_map(tdim).size_global
    N_approx = int(np.round(np.sqrt(total_cells / 2)))  # for triangles
    
    solver_info = {
        "mesh_resolution": N_approx,
        "element_degree": degree,
        "ksp_type": ksp_type,
        "pc_type": pc_type,
        "rtol": rtol,
        "iterations": total_iterations,
        "dt": float(dt_val),
        "n_steps": n_steps,
        "time_scheme": "backward_euler"
    }
    
    return u_sol, norm, solver_info

def solve(case_spec: dict) -> dict:
    """Main solve function with adaptive mesh refinement."""
    comm = MPI.COMM_WORLD
    rank = comm.rank
    
    # HARDCODED DEFAULTS from problem description
    # According to guidelines: if problem mentions t_end or dt, set hardcoded defaults
    t_end = 0.1
    dt_suggested = 0.02
    scheme = "backward_euler"
    
    # Override with case_spec if provided (but problem description has priority)
    if case_spec is not None:
        pde_info = case_spec.get('pde', {})
        time_info = pde_info.get('time', {})
        t_end = time_info.get('t_end', t_end)
        dt_suggested = time_info.get('dt', dt_suggested)
        scheme = time_info.get('scheme', scheme)
    
    # Force transient flag
    is_transient = True
    
    # Adaptive mesh refinement loop
    resolutions = [32, 64, 128]
    degree = 1  # Linear elements
    
    prev_norm = None
    u_sol_final = None
    solver_info_final = None
    mesh_resolution_used = None
    domain_final = None
    
    for N in resolutions:
        if rank == 0:
            print(f"Testing mesh resolution N={N}")
        
        # Create mesh
        domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
        
        # Solve on this mesh
        u_sol, norm, solver_info = solve_heat_on_mesh(domain, degree, dt_suggested, t_end, case_spec)
        
        # Check convergence
        if prev_norm is not None:
            relative_error = abs(norm - prev_norm) / norm if norm > 0 else float('inf')
            if rank == 0:
                print(f"  Relative error: {relative_error:.6f}")
            if relative_error < 0.01:  # 1% convergence
                u_sol_final = u_sol
                solver_info_final = solver_info
                mesh_resolution_used = N
                domain_final = domain
                if rank == 0:
                    print(f"  Converged at N={N}")
                break
        
        prev_norm = norm
        u_sol_final = u_sol
        solver_info_final = solver_info
        mesh_resolution_used = N
        domain_final = domain
    
    # Fallback: use finest mesh result
    if u_sol_final is None:
        # Should not happen, but just in case
        N = resolutions[-1]
        domain_final = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
        u_sol_final, _, solver_info_final = solve_heat_on_mesh(domain_final, degree, dt_suggested, t_end, case_spec)
        mesh_resolution_used = N
    
    # Ensure mesh_resolution in solver_info is the actual N used
    solver_info_final["mesh_resolution"] = mesh_resolution_used
    
    # Sample solution on 50x50 uniform grid
    nx = ny = 50
    x_vals = np.linspace(0.0, 1.0, nx)
    y_vals = np.linspace(0.0, 1.0, ny)
    X, Y = np.meshgrid(x_vals, y_vals, indexing='ij')
    points = np.vstack([X.flatten(), Y.flatten(), np.zeros(nx*ny)]).astype(ScalarType)
    
    u_values = probe_points(u_sol_final, domain_final, points)
    u_grid = u_values.reshape((nx, ny))
    
    # Also sample initial condition for u_initial
    u0 = fem.Function(u_sol_final.function_space)
    x = ufl.SpatialCoordinate(domain_final)
    u0_expr = ufl.sin(ufl.pi*x[0]) * ufl.sin(ufl.pi*x[1])
    u0.interpolate(fem.Expression(u0_expr, u_sol_final.function_space.element.interpolation_points))
    u0_values = probe_points(u0, domain_final, points)
    u_initial = u0_values.reshape((nx, ny))
    
    return {
        "u": u_grid,
        "u_initial": u_initial,
        "solver_info": solver_info_final
    }

if __name__ == "__main__":
    # Test the solver with a dummy case_spec
    case_spec = {
        "pde": {
            "time": {
                "t_end": 0.1,
                "dt": 0.02,
                "scheme": "backward_euler"
            }
        }
    }
    result = solve(case_spec)
    print("Solver info:", result["solver_info"])
    print("u shape:", result["u"].shape)
    print("u_initial shape:", result["u_initial"].shape)
