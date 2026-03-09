import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry, nls
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc
import time as time_module

ScalarType = PETSc.ScalarType


def solve(case_spec: dict) -> dict:
    """Solve reaction-diffusion equation (steady or transient)."""
    
    pde_spec = case_spec.get("pde", {})
    domain_spec = case_spec.get("domain", {})
    
    # Extract time parameters
    time_spec = pde_spec.get("time", {})
    
    # Hardcoded defaults for this specific problem
    t_end = 0.4
    dt_val = 0.02
    time_scheme = "backward_euler"
    is_transient = True
    
    if time_spec:
        t_end = time_spec.get("t_end", t_end)
        dt_val = time_spec.get("dt", dt_val)
        time_scheme = time_spec.get("scheme", time_scheme)
        is_transient = True
    
    # Extract diffusion coefficient
    epsilon_val = pde_spec.get("epsilon", 1.0)
    if "coefficients" in pde_spec:
        epsilon_val = pde_spec["coefficients"].get("epsilon", epsilon_val)
    
    # Extract reaction parameters
    reaction_spec = pde_spec.get("reaction", {})
    reaction_type = reaction_spec.get("type", "linear")
    reaction_coeff_val = reaction_spec.get("coefficient", 0.0)
    
    # Output grid
    output_spec = case_spec.get("output", {})
    nx_out = output_spec.get("nx", 60)
    ny_out = output_spec.get("ny", 60)
    
    # Mesh resolution
    N = 64
    degree = 1
    
    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
    
    V = fem.functionspace(domain, ("Lagrange", degree))
    
    # Spatial coordinates
    x = ufl.SpatialCoordinate(domain)
    
    # Constants
    t_param = fem.Constant(domain, ScalarType(0.0))
    epsilon = fem.Constant(domain, ScalarType(epsilon_val))
    reaction_coeff = fem.Constant(domain, ScalarType(reaction_coeff_val))
    dt_const = fem.Constant(domain, ScalarType(dt_val))
    
    # Manufactured solution: u = exp(-t)*sin(pi*x)*sin(pi*y)
    u_exact_ufl = ufl.exp(-t_param) * ufl.sin(ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])
    
    # Source term computed analytically:
    # du/dt = -exp(-t)*sin(pi*x)*sin(pi*y)
    dudt_ufl = -ufl.exp(-t_param) * ufl.sin(ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])
    
    # laplacian(u) = -2*pi^2*exp(-t)*sin(pi*x)*sin(pi*y)
    laplacian_u = -2.0 * ufl.pi**2 * ufl.exp(-t_param) * ufl.sin(ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])
    
    # R(u) = reaction_coeff * u
    R_u_exact = reaction_coeff * u_exact_ufl
    
    # f = du/dt - epsilon * laplacian(u) + R(u)
    f_ufl = dudt_ufl - epsilon * laplacian_u + R_u_exact
    
    # Functions
    u_n = fem.Function(V, name="u_n")  # solution at previous time step
    u_h = fem.Function(V, name="u_h")  # solution at current time step
    v = ufl.TestFunction(V)
    u_trial = ufl.TrialFunction(V)
    
    # Set initial condition
    t_param.value = 0.0
    u_init_expr = fem.Expression(u_exact_ufl, V.element.interpolation_points)
    u_n.interpolate(u_init_expr)
    
    # Boundary conditions
    tdim = domain.topology.dim
    fdim = tdim - 1
    
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
    )
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    
    u_bc = fem.Function(V)
    
    def update_bc(t_val):
        t_param.value = t_val
        bc_expr = fem.Expression(u_exact_ufl, V.element.interpolation_points)
        u_bc.interpolate(bc_expr)
    
    update_bc(0.0)
    bc = fem.dirichletbc(u_bc, boundary_dofs)
    bcs = [bc]
    
    # Weak form for backward Euler:
    # (u - u_n)/dt * v + epsilon * grad(u)·grad(v) + reaction_coeff * u * v = f * v
    a = (u_trial / dt_const) * v * ufl.dx \
        + epsilon * ufl.inner(ufl.grad(u_trial), ufl.grad(v)) * ufl.dx \
        + reaction_coeff * u_trial * v * ufl.dx
    
    L = (u_n / dt_const) * v * ufl.dx + f_ufl * v * ufl.dx
    
    # Solver setup
    ksp_type = "gmres"
    pc_type = "ilu"
    rtol = 1e-10
    
    a_form = fem.form(a)
    L_form = fem.form(L)
    
    A = petsc.assemble_matrix(a_form, bcs=bcs)
    A.assemble()
    
    b_vec = petsc.create_vector(V)
    
    solver = PETSc.KSP().create(domain.comm)
    solver.setOperators(A)
    solver.setType(ksp_type)
    solver.getPC().setType(pc_type)
    solver.setTolerances(rtol=rtol, atol=1e-12, max_it=1000)
    
    # Time stepping
    t_current = 0.0
    n_steps = int(round(t_end / dt_val))
    total_iterations = 0
    
    for step in range(n_steps):
        t_current += dt_val
        
        # Update BC and source term for new time
        update_bc(t_current)
        t_param.value = t_current
        
        # Reassemble matrix (BCs might change - for zero BCs this is constant but let's be safe)
        A.zeroEntries()
        petsc.assemble_matrix(A, a_form, bcs=bcs)
        A.assemble()
        
        # Assemble RHS
        with b_vec.localForm() as loc:
            loc.set(0.0)
        petsc.assemble_vector(b_vec, L_form)
        petsc.apply_lifting(b_vec, [a_form], bcs=[bcs])
        b_vec.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        petsc.set_bc(b_vec, bcs)
        
        # Solve
        solver.solve(b_vec, u_h.x.petsc_vec)
        u_h.x.scatter_forward()
        
        total_iterations += solver.getIterationNumber()
        
        # Update previous solution
        u_n.x.array[:] = u_h.x.array[:]
    
    # Evaluate solution on output grid
    x_out = np.linspace(0, 1, nx_out)
    y_out = np.linspace(0, 1, ny_out)
    X, Y = np.meshgrid(x_out, y_out, indexing='ij')
    
    points = np.zeros((3, nx_out * ny_out))
    points[0, :] = X.flatten()
    points[1, :] = Y.flatten()
    
    bb_tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, points.T)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points.T)
    
    u_values = np.full(nx_out * ny_out, np.nan)
    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    
    for i in range(nx_out * ny_out):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(points[:, i])
            cells_on_proc.append(links[0])
            eval_map.append(i)
    
    if len(points_on_proc) > 0:
        pts_arr = np.array(points_on_proc)
        cells_arr = np.array(cells_on_proc, dtype=np.int32)
        vals = u_h.eval(pts_arr, cells_arr)
        u_values[eval_map] = vals.flatten()
    
    u_grid = u_values.reshape((nx_out, ny_out))
    
    # Evaluate initial condition on grid
    t_param.value = 0.0
    u_init_func = fem.Function(V)
    u_init_func.interpolate(fem.Expression(u_exact_ufl, V.element.interpolation_points))
    
    u_init_values = np.full(nx_out * ny_out, np.nan)
    if len(points_on_proc) > 0:
        vals_init = u_init_func.eval(pts_arr, cells_arr)
        u_init_values[eval_map] = vals_init.flatten()
    u_initial_grid = u_init_values.reshape((nx_out, ny_out))
    
    # Clean up
    solver.destroy()
    A.destroy()
    b_vec.destroy()
    
    result = {
        "u": u_grid,
        "u_initial": u_initial_grid,
        "solver_info": {
            "mesh_resolution": N,
            "element_degree": degree,
            "ksp_type": ksp_type,
            "pc_type": pc_type,
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
            "reaction": {"type": "linear", "coefficient": 0.0},
            "time": {
                "t_end": 0.4,
                "dt": 0.02,
                "scheme": "backward_euler"
            }
        },
        "domain": {
            "type": "unit_square",
            "dim": 2
        },
        "output": {
            "nx": 60,
            "ny": 60
        }
    }
    
    start = time_module.time()
    result = solve(case_spec)
    elapsed = time_module.time() - start
    
    u_grid = result["u"]
    print(f"Solution shape: {u_grid.shape}")
    print(f"Solution range: [{np.nanmin(u_grid):.6f}, {np.nanmax(u_grid):.6f}]")
    print(f"Wall time: {elapsed:.3f}s")
    print(f"Iterations: {result['solver_info']['iterations']}")
    
    # Compute error against exact solution at t_end
    t_end = 0.4
    nx_out, ny_out = 60, 60
    x_out = np.linspace(0, 1, nx_out)
    y_out = np.linspace(0, 1, ny_out)
    X, Y = np.meshgrid(x_out, y_out, indexing='ij')
    u_exact = np.exp(-t_end) * np.sin(np.pi * X) * np.sin(np.pi * Y)
    
    error = np.sqrt(np.nanmean((u_grid - u_exact)**2))
    max_error = np.nanmax(np.abs(u_grid - u_exact))
    rel_error = error / np.sqrt(np.nanmean(u_exact**2))
    
    print(f"L2 grid error: {error:.6e}")
    print(f"Max grid error: {max_error:.6e}")
    print(f"Relative L2 error: {rel_error:.6e}")
    print(f"NaN count: {np.sum(np.isnan(u_grid))}")
