import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc

def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD
    
    pde = case_spec["pde"]
    time_info = pde["time"]
    output_info = case_spec["output"]
    grid_info = output_info["grid"]
    
    nx_out = grid_info["nx"]
    ny_out = grid_info["ny"]
    bbox = grid_info["bbox"]
    
    t0 = time_info["t0"]
    t_end = time_info["t_end"]
    dt_suggested = time_info.get("dt", 0.02)
    
    dt = dt_suggested / 7.0
    n_steps = int(round((t_end - t0) / dt))
    dt = (t_end - t0) / n_steps
    
    mesh_res = 192
    
    domain = mesh.create_rectangle(
        comm,
        [np.array([bbox[0], bbox[2]]), np.array([bbox[1], bbox[3]])],
        [mesh_res, mesh_res],
        cell_type=mesh.CellType.triangle,
    )
    
    V = fem.functionspace(domain, ("Lagrange", 2))
    
    x = ufl.SpatialCoordinate(domain)
    
    kappa_expr = 1.0 + 0.6 * ufl.sin(2.0 * ufl.pi * x[0]) * ufl.sin(2.0 * ufl.pi * x[1])
    kappa = fem.Function(V)
    kappa.interpolate(fem.Expression(kappa_expr, V.element.interpolation_points))
    
    f_expr = (ufl.sin(4.0 * ufl.pi * x[0]) * ufl.sin(3.0 * ufl.pi * x[1])
              + 0.3 * ufl.sin(10.0 * ufl.pi * x[0]) * ufl.sin(9.0 * ufl.pi * x[1]))
    f = fem.Function(V)
    f.interpolate(fem.Expression(f_expr, V.element.interpolation_points))
    
    u0_expr = ufl.sin(ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])
    u_n = fem.Function(V)
    u_n.interpolate(fem.Expression(u0_expr, V.element.interpolation_points))
    
    u_sol = fem.Function(V)
    u_sol.x.array[:] = u_n.x.array[:]
    
    fdim = domain.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
    )
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc = fem.dirichletbc(PETSc.ScalarType(0.0), boundary_dofs, V)
    
    u_trial = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    
    dt_const = fem.Constant(domain, PETSc.ScalarType(dt))
    
    a_form = (ufl.inner(u_trial, v) * ufl.dx
              + dt_const * ufl.inner(kappa * ufl.grad(u_trial), ufl.grad(v)) * ufl.dx)
    L_form = ufl.inner(u_n, v) * ufl.dx + dt_const * ufl.inner(f, v) * ufl.dx
    
    a_compiled = fem.form(a_form)
    L_compiled = fem.form(L_form)
    
    A = petsc.assemble_matrix(a_compiled, bcs=[bc])
    A.assemble()
    
    solver = PETSc.KSP().create(comm)
    solver.setOperators(A)
    solver.setType(PETSc.KSP.Type.CG)
    solver.getPC().setType(PETSc.PC.Type.HYPRE)
    solver.getPC().setHYPREType("boomeramg")
    rtol = 1e-10
    solver.setTolerances(rtol=rtol, atol=1e-14, max_it=1000)
    
    b = petsc.create_vector(L_compiled.function_spaces)
    
    total_iterations = 0
    
    for step in range(n_steps):
        with b.localForm() as loc:
            loc.set(0)
        petsc.assemble_vector(b, L_compiled)
        petsc.apply_lifting(b, [a_compiled], bcs=[[bc]])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        petsc.set_bc(b, [bc])
        
        solver.solve(b, u_sol.x.petsc_vec)
        u_sol.x.scatter_forward()
        
        total_iterations += solver.getIterationNumber()
        
        u_n.x.array[:] = u_sol.x.array[:]
    
    # Sample on output grid
    xs = np.linspace(bbox[0], bbox[1], nx_out)
    ys = np.linspace(bbox[2], bbox[3], ny_out)
    XX, YY = np.meshgrid(xs, ys)
    pts = np.column_stack([XX.ravel(), YY.ravel(), np.zeros(nx_out * ny_out)])
    
    bb_tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, pts)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, pts)
    
    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    for i in range(pts.shape[0]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(pts[i])
            cells_on_proc.append(links[0])
            eval_map.append(i)
    
    u_values = np.full((pts.shape[0],), np.nan)
    if len(points_on_proc) > 0:
        vals = u_sol.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u_values[eval_map] = vals.flatten()
    
    if comm.size > 1:
        all_values = np.zeros_like(u_values)
        comm.Allreduce(u_values, all_values, op=MPI.SUM)
        u_grid = all_values.reshape(ny_out, nx_out)
    else:
        u_grid = u_values.reshape(ny_out, nx_out)
    
    # Initial condition on output grid
    u_init_func = fem.Function(V)
    u_init_func.interpolate(fem.Expression(u0_expr, V.element.interpolation_points))
    
    u_init_values = np.full((pts.shape[0],), np.nan)
    if len(points_on_proc) > 0:
        vals_init = u_init_func.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u_init_values[eval_map] = vals_init.flatten()
    
    if comm.size > 1:
        all_init = np.zeros_like(u_init_values)
        comm.Allreduce(u_init_values, all_init, op=MPI.SUM)
        u_init_grid = all_init.reshape(ny_out, nx_out)
    else:
        u_init_grid = u_init_values.reshape(ny_out, nx_out)
    
    return {
        "u": u_grid,
        "u_initial": u_init_grid,
        "solver_info": {
            "mesh_resolution": mesh_res,
            "element_degree": 2,
            "ksp_type": "cg",
            "pc_type": "hypre",
            "rtol": rtol,
            "iterations": total_iterations,
            "dt": dt,
            "n_steps": n_steps,
            "time_scheme": "backward_euler",
        }
    }
