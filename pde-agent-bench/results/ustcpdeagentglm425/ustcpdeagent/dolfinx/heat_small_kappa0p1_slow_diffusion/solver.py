import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc as fem_petsc
import ufl
from petsc4py import PETSc

ScalarType = PETSc.ScalarType

def solve(case_spec: dict) -> dict:
    # Extract parameters from case_spec
    pde = case_spec["pde"]
    kappa_val = pde["coefficients"]["kappa"]
    t0 = pde["time"]["t0"]
    t_end = pde["time"]["t_end"]
    dt_suggested = pde["time"]["dt"]
    scheme = pde["time"]["scheme"]
    
    output_spec = case_spec["output"]
    nx_out = output_spec["grid"]["nx"]
    ny_out = output_spec["grid"]["ny"]
    bbox = output_spec["grid"]["bbox"]  # [xmin, xmax, ymin, ymax]
    
    # Choose parameters for accuracy
    mesh_res = 128
    element_degree = 2
    dt = 0.01  # smaller than suggested 0.02 for better accuracy
    n_steps = int(round((t_end - t0) / dt))
    
    # Create mesh
    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_square(comm, mesh_res, mesh_res, cell_type=mesh.CellType.triangle)
    
    # Function space
    V = fem.functionspace(domain, ("Lagrange", element_degree))
    
    # Manufactured solution: u = exp(-0.5*t)*sin(2*pi*x)*sin(pi*y)
    x = ufl.SpatialCoordinate(domain)
    
    # Exact solution as UFL expression (for boundary condition interpolation)
    def u_exact_ufl(t):
        return ufl.exp(-0.5 * t) * ufl.sin(2 * ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])
    
    # Source term: f = du/dt - kappa*div(grad(u))
    # du/dt = -0.5*exp(-0.5*t)*sin(2*pi*x)*sin(pi*y)
    # div(kappa*grad(u)) = kappa*(-4*pi^2 - pi^2)*exp(-0.5*t)*sin(2*pi*x)*sin(pi*y)
    #                     = -5*kappa*pi^2*exp(-0.5*t)*sin(2*pi*x)*sin(pi*y)
    # f = (-0.5 + 5*kappa*pi^2)*exp(-0.5*t)*sin(2*pi*x)*sin(pi*y)
    def source_ufl(t):
        return (-0.5 + 5.0 * kappa_val * ufl.pi**2) * ufl.exp(-0.5 * t) * ufl.sin(2 * ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])
    
    # Define variational problem
    u_n = fem.Function(V)  # solution from previous time step
    u_h = fem.Function(V)  # current solution
    
    # Interpolate initial condition
    u_n.interpolate(
        fem.Expression(u_exact_ufl(0.0), V.element.interpolation_points)
    )
    
    # Save initial condition for output
    u_initial = fem.Function(V)
    u_initial.interpolate(
        fem.Expression(u_exact_ufl(0.0), V.element.interpolation_points)
    )
    
    # Trial and test functions
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    
    # Time-dependent source and boundary as UFL expressions (will be re-evaluated each step)
    # We use fem.Expression for interpolation at each time step
    
    # Boundary condition: all Dirichlet
    fdim = domain.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
    )
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    
    # BC value function (updated each step)
    u_bc_func = fem.Function(V)
    
    # Variational form: backward Euler
    # (u - u_n)/dt * v + kappa * inner(grad(u), grad(v)) = f * v
    f_expr = fem.Function(V)  # source term function, updated each step
    
    a_form = ufl.inner(u, v) * ufl.dx + dt * kappa_val * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
    L_form = ufl.inner(u_n, v) * ufl.dx + dt * ufl.inner(f_expr, v) * ufl.dx
    
    # Compile forms
    a = fem.form(a_form)
    L = fem.form(L_form)
    
    # Assemble matrix (constant across time steps since mesh and dt don't change)
    A = fem_petsc.assemble_matrix(a, bcs=[])
    A.assemble()
    
    # Create RHS vector
    b = fem_petsc.create_vector(L.function_spaces)
    
    # Setup solver
    ksp = PETSc.KSP().create(domain.comm)
    ksp.setOperators(A)
    ksp.setType(PETSc.KSP.Type.CG)
    ksp.getPC().setType(PETSc.PC.Type.HYPRE)
    ksp.getPC().setHYPREType("boomeramg")
    rtol = 1e-10
    ksp.setTolerances(rtol=rtol)
    ksp.setFromOptions()
    
    total_iterations = 0
    t = t0
    
    for step in range(n_steps):
        t += dt
        
        # Update source term
        f_expr.interpolate(
            fem.Expression(source_ufl(t), V.element.interpolation_points)
        )
        
        # Update boundary condition
        u_bc_func.interpolate(
            fem.Expression(u_exact_ufl(t), V.element.interpolation_points)
        )
        
        # Create BC
        bc = fem.dirichletbc(u_bc_func, boundary_dofs)
        
        # Assemble RHS
        with b.localForm() as loc:
            loc.set(0)
        fem_petsc.assemble_vector(b, L)
        
        # Apply lifting
        fem_petsc.apply_lifting(b, [a], bcs=[[bc]])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        
        # Set BC on RHS
        fem_petsc.set_bc(b, [bc])
        
        # Solve
        ksp.solve(b, u_h.x.petsc_vec)
        u_h.x.scatter_forward()
        
        total_iterations += ksp.getIterationNumber()
        
        # Update previous solution
        u_n.x.array[:] = u_h.x.array[:]
    
    # Compute error against exact solution
    u_exact_func = fem.Function(V)
    u_exact_func.interpolate(
        fem.Expression(u_exact_ufl(t_end), V.element.interpolation_points)
    )
    
    # L2 error
    L2_diff = fem.form(ufl.inner(u_h - u_exact_func, u_h - u_exact_func) * ufl.dx)
    L2_exact = fem.form(ufl.inner(u_exact_func, u_exact_func) * ufl.dx)
    error_L2_sq = fem.assemble_scalar(L2_diff)
    exact_L2_sq = fem.assemble_scalar(L2_exact)
    error_L2 = np.sqrt(float(error_L2_sq) / float(exact_L2_sq)) if float(exact_L2_sq) > 0 else np.sqrt(float(error_L2_sq))
    
    if comm.rank == 0:
        print(f"Relative L2 error at t={t_end}: {error_L2:.6e}")
        print(f"Total linear iterations: {total_iterations}")
    
    # Sample solution onto output grid
    xmin, xmax, ymin, ymax = bbox
    xs = np.linspace(xmin, xmax, nx_out)
    ys = np.linspace(ymin, ymax, ny_out)
    XX, YY = np.meshgrid(xs, ys)
    
    # Points array shape (3, N)
    points = np.zeros((3, nx_out * ny_out))
    points[0, :] = XX.ravel()
    points[1, :] = YY.ravel()
    points[2, :] = 0.0
    
    # Build bounding box tree
    bb_tree = geometry.bb_tree(domain, domain.topology.dim)
    
    # Find colliding cells
    cell_candidates = geometry.compute_collisions_points(bb_tree, points.T)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points.T)
    
    # Build mapping
    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    for i in range(points.shape[1]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(points.T[i])
            cells_on_proc.append(links[0])
            eval_map.append(i)
    
    u_values = np.full((nx_out * ny_out,), np.nan)
    if len(points_on_proc) > 0:
        pts_array = np.array(points_on_proc)
        cells_array = np.array(cells_on_proc, dtype=np.int32)
        vals = u_h.eval(pts_array, cells_array)
        u_values[eval_map] = vals.flatten()
    
    # Gather on all ranks
    from mpi4py import MPI as MPI_mod
    u_values_global = np.zeros_like(u_values)
    comm.Allreduce(u_values, u_values_global, op=MPI_mod.SUM)
    
    # Handle NaN - replace with 0 (boundary points outside domain shouldn't happen for unit square)
    u_values_global = np.nan_to_num(u_values_global, nan=0.0)
    
    u_grid = u_values_global.reshape(ny_out, nx_out)
    
    # Also sample initial condition
    u_initial_values = np.full((nx_out * ny_out,), np.nan)
    if len(points_on_proc) > 0:
        pts_array = np.array(points_on_proc)
        cells_array = np.array(cells_on_proc, dtype=np.int32)
        vals_init = u_initial.eval(pts_array, cells_array)
        u_initial_values[eval_map] = vals_init.flatten()
    
    u_initial_global = np.zeros_like(u_initial_values)
    comm.Allreduce(u_initial_values, u_initial_global, op=MPI_mod.SUM)
    u_initial_grid = u_initial_global.reshape(ny_out, nx_out)
    
    # Get solver info
    pc_type = ksp.getPC().getType()
    ksp_type = ksp.getType()
    
    result = {
        "u": u_grid,
        "solver_info": {
            "mesh_resolution": mesh_res,
            "element_degree": element_degree,
            "ksp_type": str(ksp_type),
            "pc_type": str(pc_type),
            "rtol": rtol,
            "iterations": total_iterations,
            "dt": dt,
            "n_steps": n_steps,
            "time_scheme": "backward_euler"
        },
        "u_initial": u_initial_grid
    }
    
    return result


if __name__ == "__main__":
    # Test with sample case_spec
    case_spec = {
        "pde": {
            "coefficients": {"kappa": 0.1},
            "time": {"t0": 0.0, "t_end": 0.2, "dt": 0.02, "scheme": "backward_euler"}
        },
        "output": {
            "grid": {"nx": 50, "ny": 50, "bbox": [0.0, 1.0, 0.0, 1.0]}
        }
    }
    result = solve(case_spec)
    print(f"Output shape: {result['u'].shape}")
    print(f"Solver info: {result['solver_info']}")
    print(f"Max value: {np.max(result['u']):.6e}")
