import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
import ufl
from dolfinx.fem import petsc
from petsc4py import PETSc
import time

def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD
    rank = comm.rank
    
    # Parameters
    t_end = 0.1
    dt_suggested = 0.001
    scheme = "backward_euler"
    
    if 'pde' in case_spec and 'time' in case_spec['pde']:
        time_spec = case_spec['pde']['time']
        t_end = time_spec.get('t_end', t_end)
        dt_suggested = time_spec.get('dt', dt_suggested)
        scheme = time_spec.get('scheme', scheme)
    
    kappa = 1.0
    
    # Manufactured solution
    def u_exact(x, t):
        return np.exp(-t) * np.exp(-40 * ((x[0] - 0.5)**2 + (x[1] - 0.5)**2))
    
    def f_source(x, t):
        r2 = (x[0] - 0.5)**2 + (x[1] - 0.5)**2
        u_val = np.exp(-t) * np.exp(-40 * r2)
        return -u_val * (1 + kappa * (6400 * r2 - 160))
    
    # Try different configurations
    configs = [
        {"N": 64, "degree": 1, "dt": 0.001},
        {"N": 128, "degree": 1, "dt": 0.0005},
        {"N": 64, "degree": 2, "dt": 0.001},
    ]
    
    best_error = float('inf')
    best_result = None
    best_info = None
    
    for config in configs:
        N = config["N"]
        degree = config["degree"]
        dt = config["dt"]
        
        if dt > t_end:
            continue
            
        if rank == 0:
            print(f"Trying N={N}, degree={degree}, dt={dt}")
        
        # Create mesh
        domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
        V = fem.functionspace(domain, ("Lagrange", degree))
        
        # Boundary condition function
        u_bc = fem.Function(V)
        
        def boundary_marker(x):
            return np.logical_or.reduce([
                np.isclose(x[0], 0.0), np.isclose(x[0], 1.0),
                np.isclose(x[1], 0.0), np.isclose(x[1], 1.0)
            ])
        
        tdim = domain.topology.dim
        fdim = tdim - 1
        boundary_facets = mesh.locate_entities_boundary(domain, fdim, boundary_marker)
        dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
        
        # Trial/test functions
        u = ufl.TrialFunction(V)
        v = ufl.TestFunction(V)
        
        # Time-stepping
        n_steps = int(np.ceil(t_end / dt))
        dt = t_end / n_steps
        
        # Functions
        u_n = fem.Function(V)
        u_n.interpolate(lambda x: u_exact(x, 0.0))
        u_sol = fem.Function(V)
        
        # Forms
        a = (u * v + dt * kappa * ufl.inner(ufl.grad(u), ufl.grad(v))) * ufl.dx
        L = u_n * v * ufl.dx
        
        # Source term function
        f_fe = fem.Function(V)
        
        # Assemble forms
        a_form = fem.form(a)
        L_form = fem.form(L)
        
        # Time-stepping loop
        total_iterations = 0
        t = 0.0
        
        for step in range(n_steps):
            t_new = t + dt
            
            # Update boundary condition
            u_bc.interpolate(lambda x: u_exact(x, t_new))
            bc = fem.dirichletbc(u_bc, dofs)
            
            # Update source term
            f_fe.interpolate(lambda x: f_source(x, t_new))
            
            # Assemble matrix with BCs
            A = petsc.assemble_matrix(a_form, bcs=[bc])
            A.assemble()
            
            # Apply BCs to matrix (zero rows, set diagonal to 1)
            petsc.apply_lifting(A, [a_form], bcs=[[bc]])
            A.assemble()
            
            # Create RHS vector
            b = petsc.create_vector(L_form.function_spaces)
            
            # Assemble RHS
            with b.localForm() as loc:
                loc.set(0)
            petsc.assemble_vector(b, L_form)
            
            # Add source term
            source_form = fem.form(dt * ufl.inner(f_fe, v) * ufl.dx)
            b_source = petsc.create_vector(source_form.function_spaces)
            petsc.assemble_vector(b_source, source_form)
            b.axpy(1.0, b_source)
            
            # Apply BCs to RHS
            petsc.apply_lifting(b, [a_form], bcs=[[bc]])
            b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
            petsc.set_bc(b, [bc])
            
            # Create solver
            solver = PETSc.KSP().create(domain.comm)
            solver.setOperators(A)
            solver.setType("preonly")
            solver.getPC().setType("lu")
            solver.setTolerances(rtol=1e-12)
            solver.setFromOptions()
            
            # Solve
            solver.solve(b, u_sol.x.petsc_vec)
            u_sol.x.scatter_forward()
            total_iterations += solver.getIterationNumber()
            
            # Update for next step
            u_n.x.array[:] = u_sol.x.array
            t = t_new
        
        # Compute error
        u_exact_fe = fem.Function(V)
        u_exact_fe.interpolate(lambda x: u_exact(x, t_end))
        
        error = u_sol.x.array - u_exact_fe.x.array
        l2_error = np.sqrt(np.mean(error**2))
        
        if rank == 0:
            print(f"  Error: {l2_error:.6e}")
        
        if l2_error < best_error:
            best_error = l2_error
            best_result = (u_sol, domain, V)
            best_info = {
                "mesh_resolution": N,
                "element_degree": degree,
                "ksp_type": "preonly",
                "pc_type": "lu",
                "rtol": 1e-12,
                "iterations": total_iterations,
                "dt": dt,
                "n_steps": n_steps,
                "time_scheme": scheme
            }
            
            if l2_error <= 2.49e-03:
                if rank == 0:
                    print(f"  Met target error!")
                break
    
    if rank == 0:
        print(f"\nBest error: {best_error:.6e}")
    
    # Extract best result
    u_sol, domain, V = best_result
    
    # Sample on 50x50 grid
    nx = ny = 50
    x_vals = np.linspace(0.0, 1.0, nx)
    y_vals = np.linspace(0.0, 1.0, ny)
    X, Y = np.meshgrid(x_vals, y_vals, indexing='ij')
    
    points = np.zeros((3, nx * ny))
    points[0, :] = X.flatten()
    points[1, :] = Y.flatten()
    
    # Evaluate
    bb_tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, points.T)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points.T)
    
    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    for i in range(points.shape[1]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(points.T[i])
            cells_on_proc.append(links[0])
            eval_map.append(i)
    
    u_values = np.full((points.shape[1],), np.nan)
    if len(points_on_proc) > 0:
        vals = u_sol.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u_values[eval_map] = vals.flatten()
    
    u_grid = u_values.reshape(nx, ny)
    
    # Initial condition
    u0_func = fem.Function(V)
    u0_func.interpolate(lambda x: u_exact(x, 0.0))
    
    u0_values = np.full((points.shape[1],), np.nan)
    if len(points_on_proc) > 0:
        vals0 = u0_func.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u0_values[eval_map] = vals0.flatten()
    
    u0_grid = u0_values.reshape(nx, ny)
    
    return {
        "u": u_grid,
        "u_initial": u0_grid,
        "solver_info": best_info
    }

if __name__ == "__main__":
    case_spec = {"pde": {"time": {"t_end": 0.1, "dt": 0.01, "scheme": "backward_euler"}}}
    start = time.time()
    result = solve(case_spec)
    end = time.time()
    if MPI.COMM_WORLD.rank == 0:
        print(f"\nSolve time: {end - start:.3f}s")
        print("Solver info:", result["solver_info"])
