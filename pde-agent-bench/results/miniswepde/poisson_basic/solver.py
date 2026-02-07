import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
import ufl
from dolfinx.fem import petsc
from petsc4py import PETSc

def solve(case_spec: dict) -> dict:
    """
    Solve Poisson equation: -∇·(κ ∇u) = f in Ω, u = g on ∂Ω
    with manufactured solution u = sin(πx) sin(πy)
    """
    # Parameters (agent-selectable)
    mesh_resolution = 64
    element_degree = 2
    ksp_type = "cg"
    pc_type = "hypre"
    rtol = 1e-8
    
    # MPI communicator
    comm = MPI.COMM_WORLD
    
    # Create unit square mesh
    domain = mesh.create_unit_square(comm, mesh_resolution, mesh_resolution, 
                                     cell_type=mesh.CellType.triangle)
    
    # Function space
    V = fem.functionspace(domain, ("Lagrange", element_degree))
    
    # Define exact solution
    x = ufl.SpatialCoordinate(domain)
    u_exact = ufl.sin(np.pi * x[0]) * ufl.sin(np.pi * x[1])
    
    # Source term f = -Δu = 2π² sin(πx) sin(πy)
    f_expr = 2.0 * np.pi**2 * ufl.sin(np.pi * x[0]) * ufl.sin(np.pi * x[1])
    
    # Boundary condition (Dirichlet, from exact solution)
    # Mark all boundary facets (x=0, x=1, y=0, y=1) with tolerance
    tol = 1e-10
    def boundary_marker(x):
        return np.logical_or.reduce([
            np.isclose(x[0], 0.0, atol=tol),
            np.isclose(x[0], 1.0, atol=tol),
            np.isclose(x[1], 0.0, atol=tol),
            np.isclose(x[1], 1.0, atol=tol)
        ])
    
    tdim = domain.topology.dim
    fdim = tdim - 1
    boundary_facets = mesh.locate_entities_boundary(domain, fdim, boundary_marker)
    dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    
    # Interpolate exact solution for BC
    u_bc = fem.Function(V)
    u_bc.interpolate(lambda x: np.sin(np.pi * x[0]) * np.sin(np.pi * x[1]))
    bc = fem.dirichletbc(u_bc, dofs)
    
    # Variational problem
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    
    κ = 1.0  # coefficient
    a = κ * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = ufl.inner(f_expr, v) * ufl.dx
    
    # Solve using LinearProblem
    problem = petsc.LinearProblem(
        a, L, bcs=[bc],
        petsc_options={"ksp_type": ksp_type, "pc_type": pc_type, "ksp_rtol": rtol},
        petsc_options_prefix="poisson_"
    )
    u_sol = problem.solve()
    
    # Get solver iterations from KSP
    iterations = problem.solver.getIterationNumber()
    
    # Sample solution on 50x50 uniform grid
    nx, ny = 50, 50
    x_vals = np.linspace(0.0, 1.0, nx)
    y_vals = np.linspace(0.0, 1.0, ny)
    X, Y = np.meshgrid(x_vals, y_vals, indexing='ij')
    points = np.vstack([X.ravel(), Y.ravel(), np.zeros(nx * ny)])
    
    # Evaluate solution at points
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
    
    # In parallel, some points might not be found on any process
    # Gather all results to root and broadcast (simplest approach)
    if comm.size > 1:
        # Gather all u_values to root
        all_u_values = None
        if comm.rank == 0:
            all_u_values = np.empty((comm.size, points.shape[1]), dtype=np.float64)
        comm.Gather(u_values, all_u_values, root=0)
        
        if comm.rank == 0:
            # Combine: take first non-NaN value for each point
            combined = np.zeros(points.shape[1])
            for i in range(points.shape[1]):
                for p in range(comm.size):
                    val = all_u_values[p, i]
                    if not np.isnan(val):
                        combined[i] = val
                        break
            u_values = combined
        # Broadcast combined array to all processes
        u_values = comm.bcast(u_values, root=0)
    
    # Reshape to (nx, ny)
    u_grid = u_values.reshape((nx, ny))
    
    # Prepare solver_info
    solver_info = {
        "mesh_resolution": mesh_resolution,
        "element_degree": element_degree,
        "ksp_type": ksp_type,
        "pc_type": pc_type,
        "rtol": rtol,
        "iterations": iterations
    }
    
    return {"u": u_grid, "solver_info": solver_info}

if __name__ == "__main__":
    # Test the solver with a dummy case_spec
    case_spec = {
        "pde": {
            "type": "elliptic",
            "time": None  # Steady problem
        }
    }
    result = solve(case_spec)
    print("Solution shape:", result["u"].shape)
    print("Solver info:", result["solver_info"])
