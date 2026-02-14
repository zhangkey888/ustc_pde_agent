import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
import ufl
from petsc4py import PETSc
from dolfinx.fem import petsc

def solve(case_spec: dict) -> dict:
    """
    Solve Poisson equation with adaptive mesh refinement.
    """
    comm = MPI.COMM_WORLD
    ScalarType = PETSc.ScalarType
    
    # Exact solution for error checking
    def exact_solution(x):
        return np.sin(np.pi * x[0]) * np.sin(np.pi * x[1])
    
    # Fixed points for error estimation (10 points in domain interior)
    # Use a deterministic set to avoid randomness
    eval_points = np.array([
        [0.1, 0.1, 0.0],
        [0.3, 0.2, 0.0],
        [0.5, 0.5, 0.0],
        [0.7, 0.3, 0.0],
        [0.9, 0.1, 0.0],
        [0.2, 0.8, 0.0],
        [0.4, 0.6, 0.0],
        [0.6, 0.4, 0.0],
        [0.8, 0.2, 0.0],
        [0.95, 0.95, 0.0]
    ], dtype=ScalarType)
    
    # Adaptive mesh refinement loop
    resolutions = [32, 64, 128]
    u_sol = None
    norm_old = None
    solver_info = {}
    total_iterations = 0
    
    for N in resolutions:
        # Create mesh
        domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
        
        # Function space - P2 elements
        V = fem.functionspace(domain, ("Lagrange", 2))
        
        # Define variational problem
        u = ufl.TrialFunction(V)
        v = ufl.TestFunction(V)
        
        # Coefficients
        kappa = fem.Constant(domain, ScalarType(1.0))
        
        # Source term
        x = ufl.SpatialCoordinate(domain)
        f_expr = 2.0 * np.pi**2 * ufl.sin(np.pi * x[0]) * ufl.sin(np.pi * x[1])
        f = fem.Expression(f_expr, V.element.interpolation_points)
        f_func = fem.Function(V)
        f_func.interpolate(f)
        
        # Bilinear and linear forms
        a = ufl.inner(kappa * ufl.grad(u), ufl.grad(v)) * ufl.dx
        L = ufl.inner(f_func, v) * ufl.dx
        
        # Boundary condition
        def boundary_marker(x):
            return np.ones(x.shape[1], dtype=bool)
        
        tdim = domain.topology.dim
        fdim = tdim - 1
        boundary_facets = mesh.locate_entities_boundary(domain, fdim, boundary_marker)
        dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
        
        u_bc = fem.Function(V)
        u_bc.interpolate(lambda x: np.sin(np.pi * x[0]) * np.sin(np.pi * x[1]))
        bc = fem.dirichletbc(u_bc, dofs)
        
        # Manual assembly to get iteration count
        a_form = fem.form(a)
        L_form = fem.form(L)
        
        # Assemble matrix
        A = petsc.assemble_matrix(a_form, bcs=[bc])
        A.assemble()
        
        # Assemble vector
        b = petsc.create_vector(L_form.function_spaces)
        with b.localForm() as loc:
            loc.set(0)
        petsc.assemble_vector(b, L_form)
        petsc.apply_lifting(b, [a_form], bcs=[[bc]])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        petsc.set_bc(b, [bc])
        
        # Create solution function
        u_sol = fem.Function(V)
        
        # Try iterative solver first
        solver = PETSc.KSP().create(domain.comm)
        solver.setOperators(A)
        
        # Set iterative solver options
        solver.setType(PETSc.KSP.Type.GMRES)
        pc = solver.getPC()
        pc.setType(PETSc.PC.Type.HYPRE)
        solver.setTolerances(rtol=1e-8)
        solver.setFromOptions()
        
        # Solve
        try:
            solver.solve(b, u_sol.x.petsc_vec)
            u_sol.x.scatter_forward()
            iterations = solver.getIterationNumber()
            total_iterations += iterations
            solver_info["ksp_type"] = "gmres"
            solver_info["pc_type"] = "hypre"
            solver_info["rtol"] = 1e-8
        except Exception as e:
            # Fallback to direct solver
            solver.destroy()
            solver = PETSc.KSP().create(domain.comm)
            solver.setOperators(A)
            solver.setType(PETSc.KSP.Type.PREONLY)
            pc = solver.getPC()
            pc.setType(PETSc.PC.Type.LU)
            solver.solve(b, u_sol.x.petsc_vec)
            u_sol.x.scatter_forward()
            iterations = 0
            total_iterations += iterations
            solver_info["ksp_type"] = "preonly"
            solver_info["pc_type"] = "lu"
            solver_info["rtol"] = 1e-8
        
        # Compute L2 norm of solution
        norm_form = fem.form(ufl.inner(u_sol, u_sol) * ufl.dx)
        norm_local = fem.assemble_scalar(norm_form)
        norm_global = domain.comm.allreduce(norm_local, op=MPI.SUM)
        norm_new = np.sqrt(norm_global)
        
        # Compute error at sample points
        bb_tree = geometry.bb_tree(domain, domain.topology.dim)
        cell_candidates = geometry.compute_collisions_points(bb_tree, eval_points)
        colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, eval_points)
        
        points_on_proc = []
        cells_on_proc = []
        eval_map = []
        for i in range(eval_points.shape[0]):
            links = colliding_cells.links(i)
            if len(links) > 0:
                points_on_proc.append(eval_points[i])
                cells_on_proc.append(links[0])
                eval_map.append(i)
        
        point_errors = []
        if len(points_on_proc) > 0:
            vals = u_sol.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
            exact_vals = exact_solution(np.array(points_on_proc).T)
            point_errors = np.abs(vals.flatten() - exact_vals)
        
        # Gather errors across all processes
        all_errors = comm.allgather(point_errors)
        max_error = 0.0
        for err_list in all_errors:
            if len(err_list) > 0:
                max_error = max(max_error, np.max(err_list))
        
        # Check convergence criteria
        converged = False
        if norm_old is not None:
            relative_error = abs(norm_new - norm_old) / norm_new if norm_new > 0 else 1.0
            if relative_error < 0.01 and max_error < 1e-6:
                converged = True
        
        if converged:
            solver_info["mesh_resolution"] = N
            solver_info["element_degree"] = 2
            solver_info["iterations"] = total_iterations
            break
        
        norm_old = norm_new
    
    # If loop finished without break, use finest mesh
    if "mesh_resolution" not in solver_info:
        solver_info["mesh_resolution"] = 128
        solver_info["element_degree"] = 2
        solver_info["iterations"] = total_iterations
    
    # Evaluate solution on 50x50 uniform grid for output
    nx = ny = 50
    x_vals = np.linspace(0.0, 1.0, nx)
    y_vals = np.linspace(0.0, 1.0, ny)
    X, Y = np.meshgrid(x_vals, y_vals, indexing='ij')
    points = np.vstack([X.flatten(), Y.flatten(), np.zeros(nx * ny)]).T
    
    bb_tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, points)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points)
    
    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    for i in range(points.shape[0]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(points[i])
            cells_on_proc.append(links[0])
            eval_map.append(i)
    
    u_values = np.full((points.shape[0],), np.nan, dtype=ScalarType)
    if len(points_on_proc) > 0:
        vals = u_sol.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u_values[eval_map] = vals.flatten()
    
    # Reshape to (nx, ny)
    u_grid = u_values.reshape((nx, ny))
    
    # Return result
    result = {
        "u": u_grid,
        "solver_info": solver_info
    }
    return result

if __name__ == "__main__":
    # Test with dummy case_spec
    case_spec = {"pde": {"type": "elliptic"}}
    result = solve(case_spec)
    print("Solution shape:", result["u"].shape)
    print("Solver info:", result["solver_info"])
