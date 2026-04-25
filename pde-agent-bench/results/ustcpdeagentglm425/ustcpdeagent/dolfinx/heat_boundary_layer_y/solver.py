import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc

def solve(case_spec: dict) -> dict:
    pde = case_spec["pde"]
    kappa_val = pde["coefficients"]["kappa"]
    t0 = pde["time"]["t0"]
    t_end = pde["time"]["t_end"]
    
    output_spec = case_spec["output"]
    nx_out = output_spec["grid"]["nx"]
    ny_out = output_spec["grid"]["ny"]
    bbox = output_spec["grid"]["bbox"]
    
    mesh_res = 96
    element_degree = 2
    dt = 0.002
    theta = 0.5  # Crank-Nicolson
    
    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_square(comm, mesh_res, mesh_res, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", element_degree))
    
    x = ufl.SpatialCoordinate(domain)
    t_const = fem.Constant(domain, PETSc.ScalarType(t0))
    t_next = fem.Constant(domain, PETSc.ScalarType(t0))
    
    u_exact_ufl = ufl.exp(-t_const) * ufl.exp(5.0 * x[1]) * ufl.sin(ufl.pi * x[0])
    u_exact_next_ufl = ufl.exp(-t_next) * ufl.exp(5.0 * x[1]) * ufl.sin(ufl.pi * x[0])
    
    source_coeff = -1.0 + kappa_val * (float(np.pi**2) - 25.0)
    f_ufl = source_coeff * ufl.exp(-t_const) * ufl.exp(5.0 * x[1]) * ufl.sin(ufl.pi * x[0])
    f_next_ufl = source_coeff * ufl.exp(-t_next) * ufl.exp(5.0 * x[1]) * ufl.sin(ufl.pi * x[0])
    
    # Compile expressions ONCE
    bc_expr = fem.Expression(u_exact_next_ufl, V.element.interpolation_points)
    f_curr_expr = fem.Expression(f_ufl, V.element.interpolation_points)
    f_next_expr = fem.Expression(f_next_ufl, V.element.interpolation_points)
    
    u_bc_func = fem.Function(V)
    f_func = fem.Function(V)
    f_next_func = fem.Function(V)
    
    # Set initial values
    t_const.value = PETSc.ScalarType(t0)
    t_next.value = PETSc.ScalarType(t0)
    u_bc_func.interpolate(bc_expr)
    f_func.interpolate(f_curr_expr)
    f_next_func.interpolate(f_next_expr)
    
    # Boundary DOFs
    tdim = domain.topology.dim
    fdim = tdim - 1
    boundary_facets = mesh.locate_entities_boundary(domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool))
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc = fem.dirichletbc(u_bc_func, boundary_dofs)
    
    # Initial condition
    u_n = fem.Function(V)
    t_const.value = PETSc.ScalarType(t0)
    u_n.interpolate(fem.Expression(u_exact_ufl, V.element.interpolation_points))
    
    # Variational form for theta-method
    # (u^{n+1} - u^n)/dt + kappa*[theta*inner(grad(u^{n+1}),grad(v)) + (1-theta)*inner(grad(u^n),grad(v))] = theta*f^{n+1} + (1-theta)*f^n
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    kappa_c = fem.Constant(domain, PETSc.ScalarType(kappa_val))
    dt_c = fem.Constant(domain, PETSc.ScalarType(dt))
    theta_c = fem.Constant(domain, PETSc.ScalarType(theta))
    
    # Bilinear form (LHS)
    a = fem.form(u * v / dt_c * ufl.dx + theta_c * kappa_c * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx)
    
    # RHS with theta-method
    f_avg = theta_c * f_next_func + (1.0 - theta_c) * f_func
    L = fem.form(u_n * v / dt_c * ufl.dx 
                 - (1.0 - theta_c) * kappa_c * ufl.inner(ufl.grad(u_n), ufl.grad(v)) * ufl.dx
                 + f_avg * v * ufl.dx)
    
    A = petsc.assemble_matrix(a, bcs=[bc])
    A.assemble()
    
    solver = PETSc.KSP().create(domain.comm)
    solver.setOperators(A)
    solver.setType(PETSc.KSP.Type.CG)
    solver.getPC().setType(PETSc.PC.Type.HYPRE)
    solver.setTolerances(rtol=1e-10, atol=1e-14)
    
    u_sol = fem.Function(V)
    b = petsc.create_vector(L.function_spaces)
    
    n_steps = int(round((t_end - t0) / dt))
    total_iterations = 0
    
    for step in range(n_steps):
        t_current = t0 + step * dt
        t_next_val = t0 + (step + 1) * dt
        
        # Update time constants
        t_const.value = PETSc.ScalarType(t_current)
        t_next.value = PETSc.ScalarType(t_next_val)
        
        # Update BC and source terms
        u_bc_func.interpolate(bc_expr)
        f_func.interpolate(f_curr_expr)
        f_next_func.interpolate(f_next_expr)
        
        # Assemble RHS
        with b.localForm() as loc:
            loc.set(0)
        petsc.assemble_vector(b, L)
        petsc.apply_lifting(b, [a], bcs=[[bc]])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        petsc.set_bc(b, [bc])
        
        solver.solve(b, u_sol.x.petsc_vec)
        u_sol.x.scatter_forward()
        total_iterations += solver.getIterationNumber()
        
        u_n.x.array[:] = u_sol.x.array[:]
        u_n.x.scatter_forward()
    
    # L2 error verification
    t_next.value = PETSc.ScalarType(t_end)
    u_exact_func = fem.Function(V)
    u_exact_func.interpolate(bc_expr)
    
    error_form = fem.form(ufl.inner(u_sol - u_exact_func, u_sol - u_exact_func) * ufl.dx)
    error_sq = fem.assemble_scalar(error_form)
    l2_error = np.sqrt(domain.comm.allreduce(error_sq, op=MPI.SUM))
    if comm.rank == 0:
        print(f"L2 error: {l2_error:.6e}")
    
    # Sample on output grid
    xmin, xmax, ymin, ymax = bbox
    xs = np.linspace(xmin, xmax, nx_out)
    ys = np.linspace(ymin, ymax, ny_out)
    XX, YY = np.meshgrid(xs, ys)
    
    N = nx_out * ny_out
    points = np.zeros((N, 3))
    points[:, 0] = XX.ravel()
    points[:, 1] = YY.ravel()
    
    bb_tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, points)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points)
    
    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    for i in range(N):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(points[i])
            cells_on_proc.append(links[0])
            eval_map.append(i)
    
    u_values = np.zeros(N)
    if len(points_on_proc) > 0:
        vals = u_sol.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u_values[eval_map] = vals.flatten()
    
    u_global = np.zeros(N)
    comm.Allreduce(u_values, u_global, op=MPI.SUM)
    u_grid = u_global.reshape(ny_out, nx_out)
    
    # Initial condition
    t_next.value = PETSc.ScalarType(t0)
    u_init_func = fem.Function(V)
    u_init_func.interpolate(bc_expr)
    
    u_init_values = np.zeros(N)
    if len(points_on_proc) > 0:
        vals_init = u_init_func.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u_init_values[eval_map] = vals_init.flatten()
    
    u_init_global = np.zeros(N)
    comm.Allreduce(u_init_values, u_init_global, op=MPI.SUM)
    u_initial_grid = u_init_global.reshape(ny_out, nx_out)
    
    return {
        "u": u_grid,
        "u_initial": u_initial_grid,
        "solver_info": {
            "mesh_resolution": mesh_res,
            "element_degree": element_degree,
            "ksp_type": "cg",
            "pc_type": "hypre",
            "rtol": 1e-10,
            "iterations": total_iterations,
            "dt": dt,
            "n_steps": n_steps,
            "time_scheme": "crank_nicolson"
        }
    }
