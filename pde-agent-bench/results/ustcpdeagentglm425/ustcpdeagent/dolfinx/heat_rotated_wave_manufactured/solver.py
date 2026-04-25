import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc

def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD
    
    # Extract parameters
    pde = case_spec["pde"]
    time_params = pde.get("time", {})
    t0 = time_params.get("t0", 0.0)
    t_end = time_params.get("t_end", 0.1)
    dt_suggested = time_params.get("dt", 0.01)
    
    # Adaptive: use smaller dt for better accuracy
    dt = dt_suggested / 2.0  # 0.005
    kappa = 1.0
    
    # Output grid
    output = case_spec["output"]
    grid = output["grid"]
    nx_out = grid["nx"]
    ny_out = grid["ny"]
    bbox = grid["bbox"]
    xmin, xmax, ymin, ymax = bbox
    
    # Solver parameters - push accuracy higher within time budget
    mesh_res = 120
    element_degree = 3
    rtol = 1e-12
    
    # Create mesh
    domain = mesh.create_unit_square(comm, mesh_res, mesh_res,
                                      cell_type=mesh.CellType.triangle)
    
    # Function space
    V = fem.functionspace(domain, ("Lagrange", element_degree))
    
    # Spatial coordinate
    x = ufl.SpatialCoordinate(domain)
    pi = np.pi
    
    # Time variable
    t_var = fem.Constant(domain, PETSc.ScalarType(t0))
    
    # Arguments for manufactured solution
    a_arg = 3 * pi * (x[0] + x[1])
    b_arg = pi * (x[0] - x[1])
    sin_ab = ufl.sin(a_arg) * ufl.sin(b_arg)
    
    # f = exp(-t) * sin(3pi(x+y)) * sin(pi(x-y)) * (20*pi^2 - 1)
    coeff = 20 * pi**2 - 1.0
    f_expr = ufl.exp(-t_var) * sin_ab * coeff
    
    # Boundary condition g = exp(-t) * sin(3pi(x+y)) * sin(pi(x-y))
    g_expr = ufl.exp(-t_var) * sin_ab
    
    # Initial condition u0 = sin(3pi(x+y)) * sin(pi(x-y)) (at t=0)
    u0_expr = sin_ab
    
    # Variational form: Backward Euler
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    
    a_form = ufl.inner(u, v) * ufl.dx + dt * kappa * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
    
    # Previous solution
    u_n = fem.Function(V)
    u_n.interpolate(fem.Expression(u0_expr, V.element.interpolation_points))
    
    # Source term function
    f_func = fem.Function(V)
    
    L_form = ufl.inner(u_n, v) * ufl.dx + dt * ufl.inner(f_func, v) * ufl.dx
    
    # Boundary conditions
    fdim = domain.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool))
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    
    u_bc_func = fem.Function(V)
    u_bc_func.interpolate(fem.Expression(g_expr, V.element.interpolation_points))
    bc = fem.dirichletbc(u_bc_func, boundary_dofs)
    
    # Compile forms
    a_compiled = fem.form(a_form)
    L_compiled = fem.form(L_form)
    
    # Assemble LHS matrix
    A = petsc.assemble_matrix(a_compiled, bcs=[bc])
    A.assemble()
    
    # Setup solver
    solver = PETSc.KSP().create(domain.comm)
    solver.setOperators(A)
    solver.setType(PETSc.KSP.Type.CG)
    solver.getPC().setType(PETSc.PC.Type.HYPRE)
    solver.setTolerances(rtol=rtol)
    
    # Solution
    u_sol = fem.Function(V)
    
    # Time stepping
    t_current = t0
    n_steps = 0
    total_iterations = 0
    
    b = petsc.create_vector(L_compiled.function_spaces)
    
    while t_current < t_end - 1e-12:
        t_current += dt
        n_steps += 1
        t_var.value = PETSc.ScalarType(t_current)
        
        # Update source and BC
        f_func.interpolate(fem.Expression(f_expr, V.element.interpolation_points))
        u_bc_func.interpolate(fem.Expression(g_expr, V.element.interpolation_points))
        
        # Assemble RHS
        with b.localForm() as loc:
            loc.set(0)
        petsc.assemble_vector(b, L_compiled)
        petsc.apply_lifting(b, [a_compiled], bcs=[[bc]])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        petsc.set_bc(b, [bc])
        
        # Solve
        solver.solve(b, u_sol.x.petsc_vec)
        u_sol.x.scatter_forward()
        
        total_iterations += solver.getIterationNumber()
        
        # Update previous
        u_n.x.array[:] = u_sol.x.array[:]
        u_n.x.scatter_forward()
    
    # Sample on output grid
    xs = np.linspace(xmin, xmax, nx_out)
    ys = np.linspace(ymin, ymax, ny_out)
    XX, YY = np.meshgrid(xs, ys)
    
    points = np.zeros((nx_out * ny_out, 3))
    points[:, 0] = XX.ravel()
    points[:, 1] = YY.ravel()
    
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
    
    u_values = np.full((points.shape[0],), np.nan)
    pts_arr = None
    cells_arr = None
    if len(points_on_proc) > 0:
        pts_arr = np.array(points_on_proc)
        cells_arr = np.array(cells_on_proc, dtype=np.int32)
        vals = u_sol.eval(pts_arr, cells_arr)
        u_values[eval_map] = vals.flatten()
    
    u_grid = u_values.reshape(ny_out, nx_out)
    
    # Initial condition on grid
    u_initial = fem.Function(V)
    u_initial.interpolate(fem.Expression(u0_expr, V.element.interpolation_points))
    
    u_init_values = np.full((points.shape[0],), np.nan)
    if len(points_on_proc) > 0:
        vals_init = u_initial.eval(pts_arr, cells_arr)
        u_init_values[eval_map] = vals_init.flatten()
    u_initial_grid = u_init_values.reshape(ny_out, nx_out)
    
    # Compute L2 error for verification
    t_var.value = PETSc.ScalarType(t_end)
    u_exact_final = ufl.exp(-t_var) * sin_ab
    error_form = fem.form(ufl.inner(u_sol - u_exact_final, u_sol - u_exact_final) * ufl.dx)
    error_sq = fem.assemble_scalar(error_form)
    l2_error = np.sqrt(comm.allreduce(error_sq, op=MPI.SUM))
    
    solver_info = {
        "mesh_resolution": mesh_res,
        "element_degree": element_degree,
        "ksp_type": "cg",
        "pc_type": "hypre",
        "rtol": rtol,
        "iterations": total_iterations,
        "dt": dt,
        "n_steps": n_steps,
        "time_scheme": "backward_euler",
    }
    
    if comm.rank == 0:
        print(f"L2 error: {l2_error:.6e}")
        print(f"Total iterations: {total_iterations}")
    
    return {
        "u": u_grid,
        "solver_info": solver_info,
        "u_initial": u_initial_grid,
    }
