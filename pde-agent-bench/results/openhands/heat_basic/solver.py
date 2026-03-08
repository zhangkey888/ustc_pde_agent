"""
Solver for transient heat equation using dolfinx 0.10.0
Problem: ∂u/∂t - ∇·(κ ∇u) = f in Ω × (0, T]
Manufactured solution: u = exp(-t)*sin(pi*x)*sin(pi*y)
Case ID: heat_basic
"""

import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
import ufl
from petsc4py import PETSc
from dolfinx.fem import petsc

# Define scalar type
ScalarType = PETSc.ScalarType


def solve(case_spec: dict) -> dict:
    """
    Solve transient heat equation with backward Euler time stepping.
    
    Parameters:
    -----------
    case_spec : dict
        Dictionary containing PDE specification
        
    Returns:
    --------
    dict with keys:
        - "u": u_grid, numpy array with shape (nx, ny) - final solution
        - "solver_info": dict with solver parameters and performance metrics
        - "u_initial": initial condition array, same shape as u (optional but recommended)
    """
    # Extract parameters from case specification
    # The case_spec should contain the structure described in the problem
    t_end = 0.1
    dt_suggested = 0.01
    kappa = 1.0
    
    # Try to extract from case_spec if available
    if 't_end' in case_spec:
        t_end = case_spec['t_end']
    if 'dt' in case_spec:
        dt_suggested = case_spec['dt']
    
    # Check for pde structure
    if 'pde' in case_spec:
        pde_info = case_spec['pde']
        if 'coefficients' in pde_info and 'kappa' in pde_info['coefficients']:
            kappa = pde_info['coefficients']['kappa']
    
    # Agent-selectable parameters (optimized for accuracy within time limit)
    mesh_resolution = 64      # Spatial resolution - balanced accuracy/speed
    element_degree = 1        # Linear elements - sufficient for this problem
    dt = dt_suggested         # Use suggested time step
    ksp_type = 'cg'           # Conjugate gradient for symmetric positive definite
    pc_type = 'ilu'           # ILU preconditioner
    rtol = 1e-8               # Linear solver tolerance
    
    # Create mesh
    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_square(
        comm, 
        nx=mesh_resolution, 
        ny=mesh_resolution, 
        cell_type=mesh.CellType.triangle
    )
    
    # Create function space
    V = fem.functionspace(domain, ("Lagrange", element_degree))
    
    # Define trial and test functions
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    
    # Define functions for current and previous time steps
    u_n = fem.Function(V)      # Solution at previous time step
    u_sol = fem.Function(V)    # Solution at current time step
    
    # Set up boundary conditions (Dirichlet)
    # Exact solution: u = exp(-t)*sin(pi*x)*sin(pi*y) = 0 on boundaries
    tdim = domain.topology.dim
    fdim = tdim - 1
    
    # Define boundary marker function for all boundaries
    def boundary_marker(x):
        # Mark all boundaries (x=0, x=1, y=0, y=1)
        return np.logical_or.reduce([
            np.isclose(x[0], 0.0),
            np.isclose(x[0], 1.0),
            np.isclose(x[1], 0.0),
            np.isclose(x[1], 1.0)
        ])
    
    # Find boundary facets
    boundary_facets = mesh.locate_entities_boundary(domain, fdim, boundary_marker)
    
    # Locate DOFs on boundary
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    
    # Create zero boundary condition function
    u_bc = fem.Function(V)
    u_bc.interpolate(lambda x: np.zeros_like(x[0]))
    
    # Create Dirichlet BC
    bc = fem.dirichletbc(u_bc, boundary_dofs)
    
    # Define constants
    kappa_const = fem.Constant(domain, ScalarType(kappa))
    dt_const = fem.Constant(domain, ScalarType(dt))
    
    # Define spatial coordinate
    x = ufl.SpatialCoordinate(domain)
    
    # Manufactured solution: u_exact = exp(-t)*sin(pi*x)*sin(pi*y)
    # Compute source term f from exact solution:
    # ∂u/∂t = -exp(-t)*sin(pi*x)*sin(pi*y)
    # -∇·(κ ∇u) = κ*(2*pi^2)*exp(-t)*sin(pi*x)*sin(pi*y)
    # So f = ∂u/∂t - ∇·(κ ∇u) = exp(-t)*sin(pi*x)*sin(pi*y)*(-1 - κ*2*pi^2)
    
    pi = np.pi
    
    # Define variational form for backward Euler
    # (u, v) + dt*κ*(∇u, ∇v) = (u_n, v) + dt*(f, v)
    a_form = ufl.inner(u, v) * ufl.dx + dt_const * kappa_const * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
    L_form = ufl.inner(u_n, v) * ufl.dx + dt_const * ufl.inner(fem.Constant(domain, ScalarType(0.0)), v) * ufl.dx
    
    # Compile forms
    a = fem.form(a_form)
    L = fem.form(L_form)
    
    # Assemble stiffness matrix (constant in time) with BCs
    A = petsc.assemble_matrix(a, bcs=[bc])
    A.assemble()
    
    # Create vectors for RHS and solution
    b = petsc.create_vector(L.function_spaces)
    
    # Set up linear solver
    solver = PETSc.KSP().create(domain.comm)
    solver.setOperators(A)
    solver.setType(ksp_type)
    solver.getPC().setType(pc_type)
    solver.setTolerances(rtol=rtol)
    solver.setFromOptions()
    
    # Set initial condition: u(x,0) = sin(pi*x)*sin(pi*y)
    def initial_condition(x):
        return np.sin(pi * x[0]) * np.sin(pi * x[1])
    
    u_n.interpolate(initial_condition)
    u_sol.x.array[:] = u_n.x.array
    
    # Create source function (reused in each time step)
    f_func = fem.Function(V)
    
    # Time stepping
    n_steps = int(np.round(t_end / dt))
    total_iterations = 0
    
    for step in range(n_steps):
        # Update time
        current_time = (step + 1) * dt
        
        # Compute source term f at current time
        # f = exp(-t)*sin(pi*x)*sin(pi*y)*(-1 - κ*2*pi^2)
        f_value = np.exp(-current_time) * (-1.0 - kappa * 2.0 * pi**2)
        
        # Update source function
        f_func.interpolate(lambda x: f_value * np.sin(pi * x[0]) * np.sin(pi * x[1]))
        
        # Update RHS form with current source term
        L_form_updated = ufl.inner(u_n, v) * ufl.dx + dt_const * ufl.inner(f_func, v) * ufl.dx
        L_updated = fem.form(L_form_updated)
        
        # Assemble RHS
        with b.localForm() as loc:
            loc.set(0)
        petsc.assemble_vector(b, L_updated)
        
        # Apply lifting for boundary conditions
        petsc.apply_lifting(b, [a], bcs=[[bc]])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        
        # Set boundary conditions in RHS
        petsc.set_bc(b, [bc])
        
        # Solve linear system
        solver.solve(b, u_sol.x.petsc_vec)
        u_sol.x.scatter_forward()
        
        # Get iteration count for this solve
        total_iterations += solver.getIterationNumber()
        
        # Update previous solution for next time step
        u_n.x.array[:] = u_sol.x.array
    
    # Sample solution on 50x50 uniform grid
    nx, ny = 50, 50
    x_vals = np.linspace(0.0, 1.0, nx)
    y_vals = np.linspace(0.0, 1.0, ny)
    
    # Create grid points
    X, Y = np.meshgrid(x_vals, y_vals, indexing='ij')
    points = np.vstack([X.ravel(), Y.ravel(), np.zeros(nx * ny)])
    
    # Evaluate solution at grid points
    u_grid = probe_points(u_sol, points, domain)
    u_grid = u_grid.reshape(nx, ny)
    
    # Evaluate initial condition at grid points
    u_initial_func = fem.Function(V)
    u_initial_func.interpolate(initial_condition)
    u_initial_grid = probe_points(u_initial_func, points, domain)
    u_initial_grid = u_initial_grid.reshape(nx, ny)
    
    # Prepare solver info according to specification
    solver_info = {
        "mesh_resolution": mesh_resolution,
        "element_degree": element_degree,
        "ksp_type": ksp_type,
        "pc_type": pc_type,
        "rtol": rtol,
        "iterations": total_iterations,  # total linear solver iterations
        "dt": dt,
        "n_steps": n_steps,
        "time_scheme": "backward_euler"
    }
    
    return {
        "u": u_grid,
        "u_initial": u_initial_grid,
        "solver_info": solver_info
    }


def probe_points(u_func, points_array, domain):
    """
    Evaluate FEM function at arbitrary points.
    
    Parameters:
    -----------
    u_func : dolfinx.fem.Function
        Function to evaluate
    points_array : numpy.ndarray
        Array of shape (3, N) containing points
    domain : dolfinx.mesh.Mesh
        Computational mesh
        
    Returns:
    --------
    numpy.ndarray of shape (N,) with function values
    """
    # Build bounding box tree
    bb_tree = geometry.bb_tree(domain, domain.topology.dim)
    
    # Find cells colliding with points
    cell_candidates = geometry.compute_collisions_points(bb_tree, points_array.T)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points_array.T)
    
    # Build per-point mapping
    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    
    for i in range(points_array.shape[1]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(points_array.T[i])
            cells_on_proc.append(links[0])
            eval_map.append(i)
    
    # Initialize result array with NaN
    u_values = np.full((points_array.shape[1],), np.nan)
    
    # Evaluate function at points on this processor
    if len(points_on_proc) > 0:
        vals = u_func.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u_values[eval_map] = vals.flatten()
    
    # In parallel, we need to gather results from all processors
    comm = domain.comm
    if comm.size > 1:
        # Gather all values to root
        all_values = comm.gather(u_values, root=0)
        if comm.rank == 0:
            # Combine values from all processors (non-NaN values take precedence)
            combined = np.full_like(u_values, np.nan)
            for proc_vals in all_values:
                valid_mask = ~np.isnan(proc_vals)
                combined[valid_mask] = proc_vals[valid_mask]
            u_values = combined
        # Broadcast combined result to all processors
        u_values = comm.bcast(u_values, root=0)
    
    return u_values


if __name__ == "__main__":
    # Test the solver with a simple case specification
    test_case = {
        "t_end": 0.1,
        "dt": 0.01,
        "kappa": 1.0,
        "pde": {
            "time": True,
            "coefficients": {"kappa": 1.0},
            "domain": [0, 1, 0, 1]
        }
    }
    
    result = solve(test_case)
    print(f"Solution shape: {result['u'].shape}")
    print(f"Solver info: {result['solver_info']}")