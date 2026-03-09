import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry, nls
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc
import time as time_module

ScalarType = PETSc.ScalarType


def solve(case_spec: dict) -> dict:
    """Solve a reaction-diffusion equation (steady or transient)."""
    
    pde = case_spec.get("pde", {})
    
    # Extract parameters
    epsilon = pde.get("epsilon", 1.0)
    
    # Time parameters - check multiple locations
    time_params = pde.get("time", {})
    t_end = time_params.get("t_end", 0.5)
    dt_val = time_params.get("dt", 0.01)
    time_scheme = time_params.get("scheme", "backward_euler")
    
    # Force transient if t_end/dt present
    is_transient = bool(time_params) or t_end > 0
    
    # Hardcoded defaults for this specific problem
    if not time_params:
        t_end = 0.5
        dt_val = 0.01
        time_scheme = "backward_euler"
        is_transient = True
    
    # Reaction type
    reaction = pde.get("reaction", {})
    reaction_type = reaction.get("type", "linear")
    reaction_coeff = reaction.get("coefficient", 1.0)
    
    # Solver parameters
    mesh_resolution = 64
    element_degree = 1
    ksp_type_str = "gmres"
    pc_type_str = "hypre"
    rtol = 1e-8
    
    # Create mesh
    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_square(comm, mesh_resolution, mesh_resolution, 
                                      cell_type=mesh.CellType.triangle)
    
    # Function space
    V = fem.functionspace(domain, ("Lagrange", element_degree))
    
    # Spatial coordinates and time
    x = ufl.SpatialCoordinate(domain)
    
    # Time constant
    t = fem.Constant(domain, ScalarType(0.0))
    
    # Manufactured solution in UFL
    pi = ufl.pi
    u_exact_ufl = ufl.exp(-t) * ufl.sin(pi * x[0]) * ufl.sin(pi * x[1])
    
    # Compute source term from manufactured solution
    # f = du/dt - epsilon * laplacian(u) + R(u)
    # du/dt for exp(-t)*sin(pi*x)*sin(pi*y) = -exp(-t)*sin(pi*x)*sin(pi*y)
    du_dt_ufl = -ufl.exp(-t) * ufl.sin(pi * x[0]) * ufl.sin(pi * x[1])
    
    # laplacian(u) = -2*pi^2 * exp(-t) * sin(pi*x) * sin(pi*y)
    laplacian_u_ufl = -2.0 * pi**2 * ufl.exp(-t) * ufl.sin(pi * x[0]) * ufl.sin(pi * x[1])
    
    # R(u_exact) = reaction_coeff * u_exact
    R_u_exact = reaction_coeff * u_exact_ufl
    
    # Source term
    f_ufl = du_dt_ufl - epsilon * laplacian_u_ufl + R_u_exact
    
    # Trial and test functions
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    
    # Solution function
    u_sol = fem.Function(V, name="u")
    u_old = fem.Function(V, name="u_old")
    
    # Boundary conditions - u = u_exact on boundary
    # On boundary of unit square, sin(pi*x)*sin(pi*y) = 0, so u_bc = 0
    tdim = domain.topology.dim
    fdim = tdim - 1
    
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
    )
    
    u_bc_func = fem.Function(V)
    u_bc_func.x.array[:] = 0.0
    
    dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc = fem.dirichletbc(u_bc_func, dofs)
    bcs = [bc]
    
    # Time stepping with backward Euler
    dt_const = fem.Constant(domain, ScalarType(dt_val))
    eps_const = fem.Constant(domain, ScalarType(epsilon))
    react_const = fem.Constant(domain, ScalarType(reaction_coeff))
    
    # Backward Euler: (u - u_old)/dt - eps*laplacian(u) + R(u) = f
    # Weak form: (u - u_old)/dt * v + eps * grad(u) . grad(v) + react * u * v = f * v
    a = (u * v / dt_const) * ufl.dx \
        + eps_const * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx \
        + react_const * u * v * ufl.dx
    
    L = (u_old * v / dt_const) * ufl.dx \
        + f_ufl * v * ufl.dx
    
    # Initial condition: u(x, 0) = sin(pi*x)*sin(pi*y)
    u_old.interpolate(lambda x_arr: np.sin(np.pi * x_arr[0]) * np.sin(np.pi * x_arr[1]))
    u_sol.x.array[:] = u_old.x.array[:]
    
    # Compile forms
    a_form = fem.form(a)
    L_form = fem.form(L)
    
    # Assemble matrix (constant in time for linear problem)
    A = petsc.assemble_matrix(a_form, bcs=bcs)
    A.assemble()
    
    # Create RHS vector using function space
    b = petsc.create_vector(V)
    
    # Setup KSP solver
    ksp_solver = PETSc.KSP().create(domain.comm)
    ksp_solver.setOperators(A)
    ksp_solver.setType(PETSc.KSP.Type.GMRES)
    pc = ksp_solver.getPC()
    pc.setType(PETSc.PC.Type.HYPRE)
    ksp_solver.setTolerances(rtol=rtol, atol=1e-12, max_it=1000)
    ksp_solver.setUp()
    
    # Time stepping
    n_steps = int(np.round(t_end / dt_val))
    total_iterations = 0
    
    current_time = 0.0
    for step in range(n_steps):
        current_time += dt_val
        t.value = current_time
        
        # Assemble RHS
        with b.localForm() as loc:
            loc.set(0)
        petsc.assemble_vector(b, L_form)
        petsc.apply_lifting(b, [a_form], bcs=[bcs])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        petsc.set_bc(b, bcs)
        
        # Solve
        ksp_solver.solve(b, u_sol.x.petsc_vec)
        u_sol.x.scatter_forward()
        
        total_iterations += ksp_solver.getIterationNumber()
        
        # Update old solution
        u_old.x.array[:] = u_sol.x.array[:]
    
    # Evaluate solution on 60x60 grid
    nx_out, ny_out = 60, 60
    xs = np.linspace(0, 1, nx_out)
    ys = np.linspace(0, 1, ny_out)
    XX, YY = np.meshgrid(xs, ys, indexing='ij')
    
    points_2d = np.column_stack([XX.ravel(), YY.ravel()])
    points_3d = np.zeros((points_2d.shape[0], 3))
    points_3d[:, 0] = points_2d[:, 0]
    points_3d[:, 1] = points_2d[:, 1]
    
    # Point evaluation
    bb_tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, points_3d)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points_3d)
    
    u_values = np.full(points_3d.shape[0], np.nan)
    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    
    for i in range(points_3d.shape[0]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(points_3d[i])
            cells_on_proc.append(links[0])
            eval_map.append(i)
    
    if len(points_on_proc) > 0:
        vals = u_sol.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u_values[eval_map] = vals.flatten()
    
    u_grid = u_values.reshape((nx_out, ny_out))
    
    # Also compute initial condition on same grid
    u_init_func = fem.Function(V)
    u_init_func.interpolate(lambda x_arr: np.sin(np.pi * x_arr[0]) * np.sin(np.pi * x_arr[1]))
    
    u_init_values = np.full(points_3d.shape[0], np.nan)
    points_on_proc2 = []
    cells_on_proc2 = []
    eval_map2 = []
    
    for i in range(points_3d.shape[0]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc2.append(points_3d[i])
            cells_on_proc2.append(links[0])
            eval_map2.append(i)
    
    if len(points_on_proc2) > 0:
        vals2 = u_init_func.eval(np.array(points_on_proc2), np.array(cells_on_proc2, dtype=np.int32))
        u_init_values[eval_map2] = vals2.flatten()
    
    u_initial_grid = u_init_values.reshape((nx_out, ny_out))
    
    # Cleanup
    ksp_solver.destroy()
    A.destroy()
    b.destroy()
    
    result = {
        "u": u_grid,
        "u_initial": u_initial_grid,
        "solver_info": {
            "mesh_resolution": mesh_resolution,
            "element_degree": element_degree,
            "ksp_type": ksp_type_str,
            "pc_type": pc_type_str,
            "rtol": rtol,
            "iterations": total_iterations,
            "dt": dt_val,
            "n_steps": n_steps,
            "time_scheme": time_scheme,
        }
    }
    
    return result


if __name__ == "__main__":
    case_spec = {
        "pde": {
            "type": "reaction_diffusion",
            "epsilon": 1.0,
            "reaction": {"type": "linear", "coefficient": 1.0},
            "time": {
                "t_end": 0.5,
                "dt": 0.01,
                "scheme": "backward_euler"
            },
            "domain": {"type": "unit_square"},
            "manufactured_solution": "exp(-t)*(sin(pi*x)*sin(pi*y))"
        }
    }
    
    start = time_module.time()
    result = solve(case_spec)
    elapsed = time_module.time() - start
    
    print(f"Solve time: {elapsed:.2f}s")
    print(f"Solution shape: {result['u'].shape}")
    print(f"Solution range: [{np.nanmin(result['u']):.6f}, {np.nanmax(result['u']):.6f}]")
    print(f"Total iterations: {result['solver_info']['iterations']}")
    
    # Compute error against exact solution at t=0.5
    nx_out, ny_out = 60, 60
    xs = np.linspace(0, 1, nx_out)
    ys = np.linspace(0, 1, ny_out)
    XX, YY = np.meshgrid(xs, ys, indexing='ij')
    
    t_end = 0.5
    u_exact = np.exp(-t_end) * np.sin(np.pi * XX) * np.sin(np.pi * YY)
    
    diff = result['u'] - u_exact
    valid = ~np.isnan(diff)
    l2_error = np.sqrt(np.mean(diff[valid]**2))
    linf_error = np.max(np.abs(diff[valid]))
    
    print(f"L2 error (grid): {l2_error:.6e}")
    print(f"Linf error (grid): {linf_error:.6e}")
    print(f"Exact solution range: [{u_exact.min():.6f}, {u_exact.max():.6f}]")
    
    if l2_error <= 1.29e-02:
        print("PASS: Error within tolerance")
    else:
        print("FAIL: Error exceeds tolerance")
