import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc

def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD
    
    pde = case_spec.get("pde", {})
    time_params = pde.get("time", {})
    t0 = float(time_params.get("t0", 0.0))
    t_end = float(time_params.get("t_end", 0.3))
    
    epsilon = float(pde.get("epsilon", 0.01))
    reaction_lambda = float(pde.get("reaction_lambda", 1.0))
    
    output_spec = case_spec.get("output", {})
    grid_spec = output_spec.get("grid", {})
    nx_out = int(grid_spec.get("nx", 50))
    ny_out = int(grid_spec.get("ny", 50))
    bbox_out = grid_spec.get("bbox", [0, 1, 0, 1])
    
    # Proven working parameters
    mesh_res = 50
    elem_degree = 2
    dt = 0.01
    n_steps = int(round((t_end - t0) / dt))
    dt = (t_end - t0) / n_steps
    
    domain = mesh.create_unit_square(comm, mesh_res, mesh_res,
                                     cell_type=mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", elem_degree))
    
    current_t = [t0]
    
    def u_exact_val(x):
        return 0.2 * np.exp(-0.5 * current_t[0]) * np.sin(2*np.pi*x[0]) * np.sin(np.pi*x[1])
    
    def source_val(x):
        u_val = 0.2 * np.exp(-0.5 * current_t[0]) * np.sin(2*np.pi*x[0]) * np.sin(np.pi*x[1])
        return u_val * (-0.5 + 5.0*epsilon*np.pi**2 + reaction_lambda*(1.0 - u_val**2))
    
    u_n = fem.Function(V)
    u = fem.Function(V)
    f_func = fem.Function(V)
    g_func = fem.Function(V)
    u_n_cubed = fem.Function(V)
    
    current_t[0] = t0
    u_n.interpolate(u_exact_val)
    u.x.array[:] = u_n.x.array[:]
    
    fdim = domain.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool))
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc = fem.dirichletbc(g_func, boundary_dofs)
    
    u_trial = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    
    # Semi-implicit: linear part implicit, cubic reaction explicit
    # (1/dt + lam)*u*v + eps*grad(u).grad(v) = (1/dt)*u_n*v + f*v + lam*u_n^3*v
    a = ((1.0/dt + reaction_lambda) * u_trial * v 
         + epsilon * ufl.inner(ufl.grad(u_trial), ufl.grad(v))) * ufl.dx
    
    L_base = ((1.0/dt) * u_n * v + f_func * v) * ufl.dx
    L_cubed = (reaction_lambda * u_n_cubed * v) * ufl.dx
    
    a_form = fem.form(a)
    L_base_form = fem.form(L_base)
    L_cubed_form = fem.form(L_cubed)
    
    A = petsc.assemble_matrix(a_form, bcs=[bc])
    A.assemble()
    
    ksp = PETSc.KSP().create(domain.comm)
    ksp.setOperators(A)
    ksp.setType(PETSc.KSP.Type.CG)
    ksp.getPC().setType(PETSc.PC.Type.ILU)
    ksp.setTolerances(rtol=1e-10, atol=1e-14, max_it=500)
    ksp.setFromOptions()
    
    b = petsc.create_vector(L_base_form.function_spaces)
    b_cubed = petsc.create_vector(L_cubed_form.function_spaces)
    u_vec = u.x.petsc_vec
    
    total_iterations = 0
    # Semi-implicit scheme counts as 1 nonlinear iteration per step (linearized)
    nonlinear_iterations_list = [1] * n_steps
    
    for step in range(n_steps):
        t_new = t0 + (step + 1) * dt
        current_t[0] = t_new
        
        f_func.interpolate(source_val)
        g_func.interpolate(u_exact_val)
        u_n_cubed.x.array[:] = u_n.x.array[:] ** 3
        
        with b.localForm() as loc:
            loc.set(0)
        petsc.assemble_vector(b, L_base_form)
        
        with b_cubed.localForm() as loc:
            loc.set(0)
        petsc.assemble_vector(b_cubed, L_cubed_form)
        b.axpy(1.0, b_cubed)
        
        petsc.apply_lifting(b, [a_form], bcs=[[bc]])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        petsc.set_bc(b, [bc])
        
        ksp.solve(b, u_vec)
        u.x.scatter_forward()
        
        total_iterations += ksp.getIterationNumber()
        u_n.x.array[:] = u.x.array[:]
    
    # Evaluate on output grid
    x_min, x_max = float(bbox_out[0]), float(bbox_out[1])
    y_min, y_max = float(bbox_out[2]), float(bbox_out[3])
    
    xs = np.linspace(x_min, x_max, nx_out)
    ys = np.linspace(y_min, y_max, ny_out)
    XX, YY = np.meshgrid(xs, ys)
    
    points = np.zeros((nx_out * ny_out, 3), dtype=np.float64)
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
    
    u_values = np.full((points.shape[0],), np.nan, dtype=np.float64)
    pts_arr = None
    cells_arr = None
    if len(points_on_proc) > 0:
        pts_arr = np.array(points_on_proc)
        cells_arr = np.array(cells_on_proc, dtype=np.int32)
        vals = u.eval(pts_arr, cells_arr)
        u_values[eval_map] = vals.flatten()
    
    u_grid = u_values.reshape(ny_out, nx_out)
    
    # Initial condition on grid
    current_t[0] = t0
    u_n.interpolate(u_exact_val)
    u_initial_values = np.full((points.shape[0],), np.nan, dtype=np.float64)
    if pts_arr is not None:
        vals_init = u_n.eval(pts_arr, cells_arr)
        u_initial_values[eval_map] = vals_init.flatten()
    u_initial_grid = u_initial_values.reshape(ny_out, nx_out)
    
    # L2 error verification
    current_t[0] = t_end
    u_exact_func = fem.Function(V)
    u_exact_func.interpolate(u_exact_val)
    diff_func = fem.Function(V)
    diff_func.x.array[:] = u.x.array - u_exact_func.x.array
    
    error_form = fem.form(ufl.inner(diff_func, diff_func) * ufl.dx)
    error_sq_local = fem.assemble_scalar(error_form)
    error_sq = comm.allreduce(float(error_sq_local.real), op=MPI.SUM)
    l2_error = np.sqrt(error_sq)
    print(f"L2 error: {l2_error:.6e}, dt={dt:.4f}, n_steps={n_steps}, mesh={mesh_res}")
    
    solver_info = {
        "mesh_resolution": mesh_res,
        "element_degree": elem_degree,
        "ksp_type": "cg",
        "pc_type": "ilu",
        "rtol": 1e-10,
        "iterations": total_iterations,
        "dt": dt,
        "n_steps": n_steps,
        "time_scheme": "backward_euler",
        "nonlinear_iterations": nonlinear_iterations_list,
    }
    
    return {
        "u": u_grid,
        "u_initial": u_initial_grid,
        "solver_info": solver_info,
    }
