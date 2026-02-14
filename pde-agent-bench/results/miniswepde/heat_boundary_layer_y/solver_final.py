import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
import ufl
from dolfinx.fem import petsc
from petsc4py import PETSc
import time

ScalarType = PETSc.ScalarType

def solve(case_spec: dict) -> dict:
    start_time = time.time()
    
    # Parameters - use values from problem description
    t_end = 0.08
    dt_suggested = 0.008
    scheme = "backward_euler"
    
    if 'pde' in case_spec and 'time' in case_spec['pde']:
        time_params = case_spec['pde']['time']
        t_end = time_params.get('t_end', t_end)
        dt_suggested = time_params.get('dt', dt_suggested)
        scheme = time_params.get('scheme', scheme)
    
    # Exact solution and source
    def exact_solution(x, t):
        return np.exp(-t) * np.exp(5*x[1]) * np.sin(np.pi*x[0])
    
    def source_term(x, t):
        pi = np.pi
        return np.exp(-t) * np.exp(5*x[1]) * np.sin(pi*x[0]) * (pi**2 - 26)
    
    # Based on testing, use these parameters for accuracy
    # P2 elements for better accuracy, fine mesh, small dt
    element_degree = 2
    mesh_resolution = 96  # Between 64 and 128
    dt = dt_suggested / 4  # Smaller time step
    
    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_square(comm, mesh_resolution, mesh_resolution, 
                                     cell_type=mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", element_degree))
    
    # Time stepping
    n_steps = int(np.ceil(t_end / dt))
    dt_actual = t_end / n_steps
    
    # Functions
    u_n = fem.Function(V)
    u_n1 = fem.Function(V)
    
    # Initial condition
    def u0_expr(x):
        return exact_solution(x, 0.0)
    u_n.interpolate(u0_expr)
    
    # Boundary facets
    tdim = domain.topology.dim
    fdim = tdim - 1
    
    def boundary_marker(x):
        return np.logical_or.reduce([
            np.isclose(x[0], 0.0),
            np.isclose(x[0], 1.0),
            np.isclose(x[1], 0.0),
            np.isclose(x[1], 1.0)
        ])
    
    boundary_facets = mesh.locate_entities_boundary(domain, fdim, boundary_marker)
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    
    # BC function
    u_bc = fem.Function(V)
    
    # Source function
    f = fem.Function(V)
    
    # Variational form
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    
    κ = fem.Constant(domain, ScalarType(1.0))
    a = ufl.inner(u, v) * ufl.dx + dt_actual * κ * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = ufl.inner(u_n, v) * ufl.dx + dt_actual * ufl.inner(f, v) * ufl.dx
    
    a_form = fem.form(a)
    L_form = fem.form(L)
    
    # Assemble matrix once
    A = petsc.assemble_matrix(a_form)
    A.assemble()
    
    # Create RHS vector
    b = petsc.create_vector(L_form.function_spaces)
    
    # Set up solver - use direct for robustness
    ksp = PETSc.KSP().create(domain.comm)
    ksp.setOperators(A)
    ksp.setType(PETSc.KSP.Type.PREONLY)
    ksp.getPC().setType(PETSc.PC.Type.LU)
    
    # Time stepping
    current_time = 0.0
    linear_iterations = 0
    
    for step in range(n_steps):
        current_time += dt_actual
        
        # Update BC
        def bc_expr(x):
            return exact_solution(x, current_time)
        u_bc.interpolate(bc_expr)
        bc = fem.dirichletbc(u_bc, boundary_dofs)
        
        # Update source
        def f_expr(x):
            return source_term(x, current_time)
        f.interpolate(f_expr)
        
        # Assemble RHS
        with b.localForm() as loc:
            loc.set(0)
        petsc.assemble_vector(b, L_form)
        
        # Apply BCs
        petsc.apply_lifting(b, [a_form], bcs=[[bc]])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        petsc.set_bc(b, [bc])
        
        # Solve
        ksp.solve(b, u_n1.x.petsc_vec)
        u_n1.x.scatter_forward()
        
        # Get iteration count
        its = ksp.getIterationNumber()
        linear_iterations += its
        
        # Update for next step
        u_n.x.array[:] = u_n1.x.array
    
    # Sample solution on 50x50 grid
    nx = ny = 50
    x_vals = np.linspace(0.0, 1.0, nx)
    y_vals = np.linspace(0.0, 1.0, ny)
    X, Y = np.meshgrid(x_vals, y_vals, indexing='ij')
    
    points = np.zeros((3, nx * ny))
    points[0, :] = X.flatten()
    points[1, :] = Y.flatten()
    points[2, :] = 0.0
    
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
        vals = u_n1.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u_values[eval_map] = vals.flatten()
    
    u_grid = u_values.reshape((nx, ny))
    
    # Initial condition on same grid
    u0_func = fem.Function(V)
    u0_func.interpolate(u0_expr)
    
    u0_values = np.full((points.shape[1],), np.nan)
    if len(points_on_proc) > 0:
        vals0 = u0_func.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u0_values[eval_map] = vals0.flatten()
    
    u0_grid = u0_values.reshape((nx, ny))
    
    end_time = time.time()
    wall_time = end_time - start_time
    
    solver_info = {
        "mesh_resolution": mesh_resolution,
        "element_degree": element_degree,
        "ksp_type": "preonly",
        "pc_type": "lu",
        "rtol": 1e-8,
        "iterations": linear_iterations,
        "dt": dt_actual,
        "n_steps": n_steps,
        "time_scheme": scheme,
        "wall_time_sec": wall_time
    }
    
    return {
        "u": u_grid,
        "u_initial": u0_grid,
        "solver_info": solver_info
    }

if __name__ == "__main__":
    test_case = {
        "pde": {
            "time": {
                "t_end": 0.08,
                "dt": 0.008,
                "scheme": "backward_euler"
            }
        }
    }
    
    try:
        result = solve(test_case)
        print("Solver executed successfully!")
        print(f"Mesh resolution: {result['solver_info']['mesh_resolution']}")
        print(f"Element degree: {result['solver_info']['element_degree']}")
        print(f"Solution shape: {result['u'].shape}")
        print(f"Wall time: {result['solver_info']['wall_time_sec']:.3f}s")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
