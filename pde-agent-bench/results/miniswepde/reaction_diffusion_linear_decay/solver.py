import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc
import time as time_module

ScalarType = PETSc.ScalarType


def solve(case_spec: dict = None) -> dict:
    if case_spec is None:
        case_spec = {}

    pde_spec = case_spec.get("pde", {})
    
    # Extract parameters with defaults
    epsilon = case_spec.get("epsilon", 1.0)
    reaction_alpha = case_spec.get("reaction_alpha", 1.0)
    
    # Time parameters - hardcoded defaults as specified in problem
    time_spec = pde_spec.get("time", {})
    t_end = time_spec.get("t_end", 0.6)
    dt_val = time_spec.get("dt", 0.01)
    time_scheme = time_spec.get("scheme", "backward_euler")
    is_transient = True  # Force transient
    
    # Mesh resolution and element degree
    N = case_spec.get("mesh_resolution", 64)
    degree = case_spec.get("element_degree", 2)
    
    # Solver parameters
    ksp_type_str = case_spec.get("ksp_type", "gmres")
    pc_type_str = case_spec.get("pc_type", "hypre")
    rtol_val = case_spec.get("rtol", 1e-10)
    
    # Output grid
    nx_out, ny_out = 65, 65
    
    comm = MPI.COMM_WORLD
    
    # Create mesh
    domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
    
    # Function space
    V = fem.functionspace(domain, ("Lagrange", degree))
    
    # Spatial coordinates for UFL expressions
    x = ufl.SpatialCoordinate(domain)
    pi_val = np.pi
    
    # Time as a constant that we update
    t_const = fem.Constant(domain, ScalarType(0.0))
    
    # Exact solution as UFL expression
    u_exact_ufl = ufl.exp(-t_const) * ufl.cos(2 * pi_val * x[0]) * ufl.sin(pi_val * x[1])
    
    # Source term f derived from manufactured solution:
    # u = exp(-t)*cos(2*pi*x)*sin(pi*y)
    # du/dt = -u
    # laplacian(u) = -5*pi^2*u
    # f = du/dt - eps*laplacian(u) + alpha*u = u*(-1 + 5*eps*pi^2 + alpha)
    f_ufl = u_exact_ufl * (-1.0 + 5.0 * epsilon * pi_val**2 + reaction_alpha)
    
    # Trial and test functions
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    
    # Solution function
    u_sol = fem.Function(V, name="u")
    u_old = fem.Function(V, name="u_old")
    
    # Set initial condition: u(x, 0) = cos(2*pi*x)*sin(pi*y)
    u_sol.interpolate(lambda X: np.cos(2 * np.pi * X[0]) * np.sin(np.pi * X[1]))
    u_old.x.array[:] = u_sol.x.array[:]
    
    # Boundary conditions - all boundaries
    tdim = domain.topology.dim
    fdim = tdim - 1
    
    # Find all boundary facets
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda X: np.ones(X.shape[1], dtype=bool)
    )
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    
    # BC function that will be updated each time step
    u_bc_func = fem.Function(V)
    
    # Constants
    dt_const = fem.Constant(domain, ScalarType(dt_val))
    eps_const = fem.Constant(domain, ScalarType(epsilon))
    alpha_const = fem.Constant(domain, ScalarType(reaction_alpha))
    
    # Backward Euler: (u - u_old)/dt - eps*laplacian(u) + alpha*u = f
    # Bilinear form
    a_form = (u * v / dt_const + eps_const * ufl.inner(ufl.grad(u), ufl.grad(v)) + alpha_const * u * v) * ufl.dx
    
    # Linear form
    L_form = (u_old / dt_const * v + f_ufl * v) * ufl.dx
    
    # Compile forms
    a_compiled = fem.form(a_form)
    L_compiled = fem.form(L_form)
    
    # Create vector for RHS using function space
    b = petsc.create_vector(V)
    
    # Setup KSP solver
    solver = PETSc.KSP().create(domain.comm)
    solver.setType(ksp_type_str)
    pc = solver.getPC()
    pc.setType(pc_type_str)
    solver.setTolerances(rtol=rtol_val, atol=1e-12, max_it=2000)
    
    # Time stepping
    t = 0.0
    n_steps = int(np.round(t_end / dt_val))
    total_iterations = 0
    
    for step in range(n_steps):
        t += dt_val
        t_const.value = t
        
        # Update boundary condition
        t_current = t
        u_bc_func.interpolate(
            lambda X, tc=t_current: np.exp(-tc) * np.cos(2 * np.pi * X[0]) * np.sin(np.pi * X[1])
        )
        bc = fem.dirichletbc(u_bc_func, boundary_dofs)
        
        # Assemble matrix
        A = petsc.assemble_matrix(a_compiled, bcs=[bc])
        A.assemble()
        
        # Assemble RHS
        with b.localForm() as loc:
            loc.set(0)
        petsc.assemble_vector(b, L_compiled)
        petsc.apply_lifting(b, [a_compiled], bcs=[[bc]])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        petsc.set_bc(b, [bc])
        
        # Set operator and solve
        solver.setOperators(A)
        solver.setUp()
        solver.solve(b, u_sol.x.petsc_vec)
        u_sol.x.scatter_forward()
        
        total_iterations += solver.getIterationNumber()
        
        # Update old solution
        u_old.x.array[:] = u_sol.x.array[:]
        
        # Destroy matrix to avoid memory leak
        A.destroy()
    
    # Evaluate solution on 65x65 grid
    xs = np.linspace(0, 1, nx_out)
    ys = np.linspace(0, 1, ny_out)
    XX, YY = np.meshgrid(xs, ys, indexing='ij')
    
    # Points array: shape (N_points, 3)
    points_3d = np.zeros((nx_out * ny_out, 3))
    points_3d[:, 0] = XX.flatten()
    points_3d[:, 1] = YY.flatten()
    
    # Use geometry utilities for point evaluation
    bb_tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, points_3d)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points_3d)
    
    u_values = np.full(nx_out * ny_out, np.nan)
    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    
    for i in range(len(points_3d)):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(points_3d[i])
            cells_on_proc.append(links[0])
            eval_map.append(i)
    
    if len(points_on_proc) > 0:
        pts_arr = np.array(points_on_proc)
        cells_arr = np.array(cells_on_proc, dtype=np.int32)
        vals = u_sol.eval(pts_arr, cells_arr)
        u_values[eval_map] = vals.flatten()
    
    u_grid = u_values.reshape((nx_out, ny_out))
    
    # Compute exact solution for verification
    u_exact_grid = np.exp(-t_end) * np.cos(2 * np.pi * XX) * np.sin(np.pi * YY)
    error = np.sqrt(np.nanmean((u_grid - u_exact_grid)**2))
    print(f"L2 grid error: {error:.6e}")
    
    result = {
        "u": u_grid,
        "solver_info": {
            "mesh_resolution": N,
            "element_degree": degree,
            "ksp_type": ksp_type_str,
            "pc_type": pc_type_str,
            "rtol": rtol_val,
            "iterations": total_iterations,
            "dt": dt_val,
            "n_steps": n_steps,
            "time_scheme": time_scheme,
        }
    }
    
    # Cleanup
    solver.destroy()
    b.destroy()
    
    return result


if __name__ == "__main__":
    t0 = time_module.time()
    result = solve()
    elapsed = time_module.time() - t0
    print(f"Wall time: {elapsed:.2f}s")
    print(f"Grid shape: {result['u'].shape}")
    print(f"Solver info: {result['solver_info']}")
    print(f"NaN count: {np.isnan(result['u']).sum()}")
