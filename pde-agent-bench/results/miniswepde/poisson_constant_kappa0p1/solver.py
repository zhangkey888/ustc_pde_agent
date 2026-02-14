import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
import ufl
from dolfinx.fem import petsc
from petsc4py import PETSc

def solve(case_spec: dict) -> dict:
    """
    Solve Poisson equation with adaptive mesh refinement.
    Returns dict with:
        - "u": numpy array shape (50, 50) solution on uniform grid
        - "solver_info": dict with mesh_resolution, element_degree, ksp_type, pc_type, rtol, iterations
    """
    comm = MPI.COMM_WORLD
    rank = comm.rank
    
    # Problem parameters
    kappa = 0.1  # constant coefficient
    exact_solution = lambda x: np.sin(np.pi * x[0]) * np.sin(np.pi * x[1])
    source_function = lambda x: kappa * 2.0 * np.pi**2 * np.sin(np.pi * x[0]) * np.sin(np.pi * x[1])
    
    # Adaptive mesh refinement loop
    resolutions = [32, 64, 128]
    element_degree = 1  # linear elements
    u_solution = None
    norm_old = None
    final_resolution = None
    solver_info = {}
    iterations = 0
    ksp_type = "gmres"
    pc_type = "hypre"
    rtol = 1e-8
    
    for N in resolutions:
        # Create mesh
        domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
        
        # Function space
        V = fem.functionspace(domain, ("Lagrange", element_degree))
        
        # Dirichlet boundary condition (exact solution on entire boundary)
        tdim = domain.topology.dim
        fdim = tdim - 1
        boundary_facets = mesh.locate_entities_boundary(
            domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
        )
        dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
        u_bc = fem.Function(V)
        u_bc.interpolate(lambda x: exact_solution(x))
        bc = fem.dirichletbc(u_bc, dofs)
        
        # Source term as a Function interpolated
        f = fem.Function(V)
        f.interpolate(lambda x: source_function(x))
        
        # Variational form
        u = ufl.TrialFunction(V)
        v = ufl.TestFunction(V)
        a = kappa * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
        L = ufl.inner(f, v) * ufl.dx
        
        # Try iterative solver first, fallback to direct
        solver_choice = "iterative"
        ksp_type = "gmres"
        pc_type = "hypre"
        rtol = 1e-8
        
        # Create linear problem
        try:
            problem = petsc.LinearProblem(
                a, L, bcs=[bc],
                petsc_options={"ksp_type": ksp_type, "pc_type": pc_type, "ksp_rtol": rtol},
                petsc_options_prefix="poisson_"
            )
            u_h = problem.solve()
            iterations = problem.solver.getIterationNumber()
            converged = True
        except Exception as e:
            if rank == 0:
                print(f"Iterative solver failed: {e}, switching to direct solver")
            solver_choice = "direct"
            ksp_type = "preonly"
            pc_type = "lu"
            rtol = 0.0
            problem = petsc.LinearProblem(
                a, L, bcs=[bc],
                petsc_options={"ksp_type": ksp_type, "pc_type": pc_type},
                petsc_options_prefix="poisson_"
            )
            u_h = problem.solve()
            iterations = 0  # direct solver doesn't have iterations
            converged = True
        
        # Compute L2 norm of solution
        norm_form = fem.form(ufl.inner(u_h, u_h) * ufl.dx)
        norm_value = np.sqrt(fem.assemble_scalar(norm_form))
        
        # Check convergence
        if norm_old is not None:
            rel_error = abs(norm_value - norm_old) / norm_value
            if rel_error < 0.01:  # 1% convergence
                if rank == 0:
                    print(f"Converged at N={N}, relative error {rel_error:.4f}")
                u_solution = u_h
                final_resolution = N
                solver_info = {
                    "mesh_resolution": N,
                    "element_degree": element_degree,
                    "ksp_type": ksp_type,
                    "pc_type": pc_type,
                    "rtol": rtol,
                    "iterations": iterations
                }
                break
        norm_old = norm_value
        u_solution = u_h
        final_resolution = N
        solver_info = {
            "mesh_resolution": N,
            "element_degree": element_degree,
            "ksp_type": ksp_type,
            "pc_type": pc_type,
            "rtol": rtol,
            "iterations": iterations
        }
    
    # If loop finishes without break, use last solution (N=128)
    if final_resolution is None:
        final_resolution = 128
        solver_info = {
            "mesh_resolution": 128,
            "element_degree": element_degree,
            "ksp_type": ksp_type,
            "pc_type": pc_type,
            "rtol": rtol,
            "iterations": iterations
        }
    
    # Sample solution on 50x50 uniform grid
    nx = ny = 50
    x = np.linspace(0.0, 1.0, nx)
    y = np.linspace(0.0, 1.0, ny)
    X, Y = np.meshgrid(x, y, indexing='ij')
    points = np.vstack([X.ravel(), Y.ravel(), np.zeros(nx * ny)]).T  # 3D points with z=0
    
    # Evaluate function at points
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
    
    u_values = np.full((points.shape[0],), np.nan, dtype=PETSc.ScalarType)
    if len(points_on_proc) > 0:
        vals = u_solution.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u_values[eval_map] = vals.flatten()
    
    # Gather results on rank 0 (assuming serial run, but safe)
    u_grid = u_values.reshape((nx, ny))
    
    # Ensure all ranks return same dict (rank 0 broadcasts)
    if comm.size > 1:
        u_grid = comm.bcast(u_grid, root=0)
        solver_info = comm.bcast(solver_info, root=0)
    
    return {"u": u_grid, "solver_info": solver_info}

if __name__ == "__main__":
    # Test with dummy case_spec
    case_spec = {"pde": {"type": "poisson"}}
    result = solve(case_spec)
    if MPI.COMM_WORLD.rank == 0:
        print("Solution shape:", result["u"].shape)
        print("Solver info:", result["solver_info"])
