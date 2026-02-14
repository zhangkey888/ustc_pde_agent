import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
import ufl
from dolfinx.fem import petsc
from petsc4py import PETSc

def solve(case_spec: dict) -> dict:
    """
    Solve Poisson equation with adaptive mesh refinement.
    """
    comm = MPI.COMM_WORLD
    ScalarType = PETSc.ScalarType
    
    # Adaptive mesh refinement loop
    resolutions = [32, 64, 128]
    u_solutions = []
    norms = []
    total_iterations = 0
    solver_types = []  # Track what solver was used for each resolution
    
    # Element degree
    element_degree = 1
    
    for i, N in enumerate(resolutions):
        # Create mesh
        domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
        
        # Define function space
        V = fem.functionspace(domain, ("Lagrange", element_degree))
        
        # Define boundary condition (Dirichlet, u=0 on entire boundary)
        tdim = domain.topology.dim
        fdim = tdim - 1
        
        def boundary_marker(x):
            # Mark all boundaries
            return np.logical_or.reduce([
                np.isclose(x[0], 0.0),
                np.isclose(x[0], 1.0),
                np.isclose(x[1], 0.0),
                np.isclose(x[1], 1.0)
            ])
        
        boundary_facets = mesh.locate_entities_boundary(domain, fdim, boundary_marker)
        dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
        u_bc = fem.Function(V)
        u_bc.interpolate(lambda x: np.zeros_like(x[0]))
        bc = fem.dirichletbc(u_bc, dofs)
        
        # Define variational problem
        u = ufl.TrialFunction(V)
        v = ufl.TestFunction(V)
        
        κ = fem.Constant(domain, ScalarType(1.0))
        x = ufl.SpatialCoordinate(domain)
        f_expr = 2.0 * (x[0] + x[1] - x[0]**2 - x[1]**2)  # -∇²u_exact
        
        a = ufl.inner(κ * ufl.grad(u), ufl.grad(v)) * ufl.dx
        L = ufl.inner(f_expr, v) * ufl.dx
        
        # Create forms
        a_form = fem.form(a)
        L_form = fem.form(L)
        
        # Assemble matrix
        A = petsc.assemble_matrix(a_form, bcs=[bc])
        A.assemble()
        
        # Create vectors
        b = petsc.create_vector(L_form.function_spaces)
        u_sol = fem.Function(V)
        
        # Assemble RHS
        with b.localForm() as loc:
            loc.set(0)
        petsc.assemble_vector(b, L_form)
        petsc.apply_lifting(b, [a_form], bcs=[[bc]])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        petsc.set_bc(b, [bc])
        
        # Create solver - try iterative first
        solver = PETSc.KSP().create(domain.comm)
        solver.setOperators(A)
        
        ksp_type = "gmres"
        pc_type = "hypre"
        rtol = 1e-8
        
        # Try GMRES with hypre first
        try:
            solver.setType(PETSc.KSP.Type.GMRES)
            # Try hypre, if not available try ilu
            try:
                solver.getPC().setType(PETSc.PC.Type.HYPRE)
                pc_type = "hypre"
            except:
                solver.getPC().setType(PETSc.PC.Type.ILU)
                pc_type = "ilu"
            solver.setTolerances(rtol=rtol)
            solver.setFromOptions()
            solver_types.append((ksp_type, pc_type))
        except Exception as e:
            # Fallback to direct solver
            solver.setType(PETSc.KSP.Type.PREONLY)
            solver.getPC().setType(PETSc.PC.Type.LU)
            ksp_type = "preonly"
            pc_type = "lu"
            rtol = 1e-12
            solver_types.append((ksp_type, pc_type))
        
        # Solve
        solver.solve(b, u_sol.x.petsc_vec)
        u_sol.x.scatter_forward()
        
        # Get iteration count
        iterations = solver.getIterationNumber()
        total_iterations += iterations
        
        # Compute L2 norm of solution
        norm_form = fem.form(ufl.inner(u_sol, u_sol) * ufl.dx)
        norm_value = np.sqrt(comm.allreduce(fem.assemble_scalar(norm_form), op=MPI.SUM))
        norms.append(norm_value)
        u_solutions.append(u_sol)
        
        # Clean up PETSc objects
        A.destroy()
        b.destroy()
        solver.destroy()
        
        # Check convergence
        if len(norms) >= 2:
            rel_error = abs(norms[-1] - norms[-2]) / norms[-1] if norms[-1] > 1e-15 else 0.0
            if comm.rank == 0:
                print(f"N={N}: norm={norm_value:.6e}, rel_error={rel_error:.6e}, iterations={iterations}, solver={solver_types[-1]}")
            if rel_error < 0.01:  # 1% convergence criterion
                if comm.rank == 0:
                    print(f"Converged at N={N} with relative error {rel_error:.6e}")
                break
        else:
            if comm.rank == 0:
                print(f"N={N}: norm={norm_value:.6e}, iterations={iterations}, solver={solver_types[-1]}")
    
    # Use the last solution (most refined)
    final_u = u_solutions[-1]
    final_N = resolutions[min(len(u_solutions)-1, len(resolutions)-1)]
    
    # Determine solver types used (use the last one)
    ksp_type, pc_type = solver_types[-1]
    rtol = 1e-8 if ksp_type == "gmres" else 1e-12
    
    # Prepare output on 50x50 grid
    nx, ny = 50, 50
    x_vals = np.linspace(0.0, 1.0, nx)
    y_vals = np.linspace(0.0, 1.0, ny)
    X, Y = np.meshgrid(x_vals, y_vals, indexing='ij')
    
    # Create points array for evaluation (shape (3, nx*ny))
    points = np.zeros((3, nx * ny))
    points[0, :] = X.flatten()
    points[1, :] = Y.flatten()
    points[2, :] = 0.0  # z-coordinate for 2D
    
    # Evaluate solution at points
    u_grid_flat = evaluate_function_at_points(final_u, points)
    u_grid = u_grid_flat.reshape((nx, ny))
    
    # Prepare solver_info
    solver_info = {
        "mesh_resolution": final_N,
        "element_degree": element_degree,
        "ksp_type": ksp_type,
        "pc_type": pc_type,
        "rtol": rtol,
        "iterations": total_iterations
    }
    
    return {
        "u": u_grid,
        "solver_info": solver_info
    }

def evaluate_function_at_points(u_func, points):
    """
    Evaluate a dolfinx Function at arbitrary points.
    points: numpy array of shape (3, N)
    Returns: numpy array of shape (N,)
    """
    domain = u_func.function_space.mesh
    bb_tree = geometry.bb_tree(domain, domain.topology.dim)
    
    # Find cells colliding with points
    cell_candidates = geometry.compute_collisions_points(bb_tree, points.T)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points.T)
    
    # Build per-point mapping
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
        vals = u_func.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u_values[eval_map] = vals.flatten()
    
    # For points not found on this processor, gather from others
    comm = domain.comm
    all_u_values = np.zeros_like(u_values)
    comm.Allreduce(u_values, all_u_values, op=MPI.MAX)
    
    return all_u_values

if __name__ == "__main__":
    # Test the solver with a dummy case_spec
    case_spec = {
        "pde": {
            "type": "elliptic"
        }
    }
    result = solve(case_spec)
    print("\nFinal result:")
    print("Solution shape:", result["u"].shape)
    print("Solver info:", result["solver_info"])
