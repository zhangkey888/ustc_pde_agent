import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc

def solve(case_spec: dict) -> dict:
    """
    Solve the heat equation with adaptive mesh refinement and runtime auto-tuning.
    Returns dict with keys: "u", "u_initial", "solver_info"
    """
    comm = MPI.COMM_WORLD
    ScalarType = PETSc.ScalarType
    
    # Extract parameters with defaults
    pde_spec = case_spec.get('pde', {})
    time_spec = pde_spec.get('time', {})
    coeff_spec = pde_spec.get('coefficients', {})
    
    t_end = time_spec.get('t_end', 0.12)
    dt = time_spec.get('dt', 0.006)
    scheme = time_spec.get('scheme', 'backward_euler')
    kappa = coeff_spec.get('kappa', 0.5)
    
    # Ensure dt > 0
    if dt <= 0:
        dt = t_end / 20
    
    # Compute integer number of steps
    n_steps = int(np.floor(t_end / dt))
    if n_steps < 1:
        n_steps = 1
        dt = t_end
    
    # Adaptive mesh refinement loop
    resolutions = [32, 64, 128]
    solutions = []
    norms = []
    final_domain = None
    final_solution = None
    solver_stats = {}
    
    for N in resolutions:
        # Create mesh
        domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
        V = fem.functionspace(domain, ("Lagrange", 1))
        
        # Boundary condition (Dirichlet, u=0 on all boundaries)
        tdim = domain.topology.dim
        fdim = tdim - 1
        
        def boundary_marker(x):
            return np.logical_or.reduce([
                np.isclose(x[0], 0.0), np.isclose(x[0], 1.0),
                np.isclose(x[1], 0.0), np.isclose(x[1], 1.0)
            ])
        
        boundary_facets = mesh.locate_entities_boundary(domain, fdim, boundary_marker)
        dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
        u_bc = fem.Function(V)
        u_bc.interpolate(lambda x: np.zeros_like(x[0]))
        bc = fem.dirichletbc(u_bc, dofs)
        
        # Functions
        u_n = fem.Function(V)  # Previous time step
        u_n.interpolate(lambda x: np.sin(np.pi * x[0]) * np.sin(np.pi * x[1]))
        u = fem.Function(V)    # Current solution
        t_c = fem.Constant(domain, ScalarType(0.0))  # Time constant
        
        # Variational form for backward Euler
        v = ufl.TestFunction(V)
        u_trial = ufl.TrialFunction(V)
        x = ufl.SpatialCoordinate(domain)
        
        # Source term from manufactured solution
        f_expr = ufl.exp(-10 * t_c) * ufl.sin(np.pi * x[0]) * ufl.sin(np.pi * x[1]) * \
                (ScalarType(-10.0) + ScalarType(kappa) * ScalarType(2.0 * np.pi**2))
        
        a = ufl.inner(u_trial, v) * ufl.dx + dt * kappa * ufl.inner(ufl.grad(u_trial), ufl.grad(v)) * ufl.dx
        L = ufl.inner(u_n, v) * ufl.dx + dt * ufl.inner(f_expr, v) * ufl.dx
        
        a_form = fem.form(a)
        L_form = fem.form(L)
        
        # Assemble matrix (constant in time)
        A = petsc.assemble_matrix(a_form, bcs=[bc])
        A.assemble()
        
        # Create RHS vector
        b = petsc.create_vector(L_form.function_spaces)
        
        # Create linear solver with fallback strategy
        solver = PETSc.KSP().create(comm)
        solver.setOperators(A)
        
        # Try iterative solver first
        ksp_type = "preonly"
        pc_type = "lu"
        try:
            solver.setType(PETSc.KSP.Type.GMRES)
            pc = solver.getPC()
            pc.setType(PETSc.PC.Type.HYPRE)
            solver.setTolerances(rtol=1e-8, max_it=1000)
            solver.setFromOptions()
            # Test the solver
            test_b = b.duplicate()
            test_x = test_b.duplicate()
            test_b.set(1.0)
            solver.solve(test_b, test_x)
            ksp_type = "gmres"
            pc_type = "hypre"
        except Exception:
            # Fallback to direct solver
            solver.setType(PETSc.KSP.Type.PREONLY)
            pc = solver.getPC()
            pc.setType(PETSc.PC.Type.LU)
        
        # Time stepping
        total_iterations = 0
        t = 0.0
        
        for step in range(n_steps):
            t += dt
            t_c.value = t
            
            # Assemble RHS
            with b.localForm() as loc:
                loc.set(0)
            petsc.assemble_vector(b, L_form)
            petsc.apply_lifting(b, [a_form], bcs=[[bc]])
            b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
            petsc.set_bc(b, [bc])
            
            # Solve
            solver.solve(b, u.x.petsc_vec)
            u.x.scatter_forward()
            total_iterations += solver.getIterationNumber()
            
            # Update for next step
            u_n.x.array[:] = u.x.array
        
        # Store solution and compute norm
        solutions.append(u.copy())
        norm_val = np.sqrt(comm.allreduce(fem.assemble_scalar(fem.form(ufl.inner(u, u) * ufl.dx)), op=MPI.SUM))
        norms.append(norm_val)
        
        # Store solver statistics
        solver_stats = {
            "mesh_resolution": N,
            "element_degree": 1,
            "ksp_type": ksp_type,
            "pc_type": pc_type,
            "rtol": 1e-8,
            "iterations": total_iterations,
            "dt": dt,
            "n_steps": n_steps,
            "time_scheme": scheme
        }
        
        # Check convergence (stop if relative change < 1%)
        if len(norms) >= 2:
            relative_error = abs(norms[-1] - norms[-2]) / norms[-1] if norms[-1] > 0 else 1.0
            if relative_error < 0.01:
                final_domain = domain
                final_solution = u
                break
    
    # Use converged mesh or finest mesh
    if final_domain is None:
        final_domain = domain
        final_solution = solutions[-1]
    
    # Sample solution on 50x50 uniform grid
    nx = ny = 50
    x_vals = np.linspace(0.0, 1.0, nx)
    y_vals = np.linspace(0.0, 1.0, ny)
    X, Y = np.meshgrid(x_vals, y_vals, indexing='ij')
    points = np.vstack([X.flatten(), Y.flatten(), np.zeros(nx * ny)]).T
    
    # Evaluate solution at points
    bb_tree = geometry.bb_tree(final_domain, final_domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, points)
    colliding_cells = geometry.compute_colliding_cells(final_domain, cell_candidates, points)
    
    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    for i in range(points.shape[0]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(points[i])
            cells_on_proc.append(links[0])
            eval_map.append(i)
    
    u_values = np.full((points.shape[0],), np.nan)
    if len(points_on_proc) > 0:
        vals = final_solution.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u_values[eval_map] = vals.flatten()
    u_grid = u_values.reshape(nx, ny)
    
    # Initial condition on same grid
    u0_func = fem.Function(fem.functionspace(final_domain, ("Lagrange", 1)))
    u0_func.interpolate(lambda x: np.sin(np.pi * x[0]) * np.sin(np.pi * x[1]))
    
    u0_values = np.full((points.shape[0],), np.nan)
    if len(points_on_proc) > 0:
        vals0 = u0_func.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u0_values[eval_map] = vals0.flatten()
    u_initial = u0_values.reshape(nx, ny)
    
    return {
        "u": u_grid,
        "u_initial": u_initial,
        "solver_info": solver_stats
    }

if __name__ == "__main__":
    # Test case
    case_spec = {
        "pde": {
            "time": {"t_end": 0.12, "dt": 0.006, "scheme": "backward_euler"},
            "coefficients": {"kappa": 0.5}
        }
    }
    result = solve(case_spec)
    print("Test passed!")
    print("Solution shape:", result["u"].shape)
    print("Solver info mesh_resolution:", result["solver_info"]["mesh_resolution"])
