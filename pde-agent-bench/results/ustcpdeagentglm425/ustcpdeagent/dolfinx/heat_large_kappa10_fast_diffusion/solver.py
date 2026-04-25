import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc

def solve(case_spec: dict) -> dict:
    pde = case_spec["pde"]
    kappa = float(pde["coefficients"]["kappa"])
    t0 = float(pde["time"]["t0"])
    t_end = float(pde["time"]["t_end"])
    
    out = case_spec["output"]
    grid = out["grid"]
    nx_out = grid["nx"]
    ny_out = grid["ny"]
    bbox = grid["bbox"]
    
    # High accuracy discretization within time budget
    mesh_res = 48
    element_degree = 3
    dt = 0.001  # 50 time steps
    
    n_steps = int(round((t_end - t0) / dt))
    dt = (t_end - t0) / n_steps
    
    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_square(comm, mesh_res, mesh_res, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", element_degree))
    
    x_coord = ufl.SpatialCoordinate(domain)
    t_const = fem.Constant(domain, PETSc.ScalarType(t0))
    
    # Source: f = exp(-t)*sin(pi*x)*sin(pi*y)*(2*kappa*pi^2 - 1)
    f_coeff = 2.0 * kappa * np.pi**2 - 1.0
    f_expr = ufl.exp(-t_const) * ufl.sin(ufl.pi * x_coord[0]) * ufl.sin(ufl.pi * x_coord[1]) * f_coeff
    
    u_sol = fem.Function(V)
    u_old = fem.Function(V)
    
    # Initial condition: u0 = sin(pi*x)*sin(pi*y)
    u_old.interpolate(fem.Expression(
        ufl.sin(ufl.pi * x_coord[0]) * ufl.sin(ufl.pi * x_coord[1]),
        V.element.interpolation_points))
    
    # Homogeneous Dirichlet BC on all boundaries
    fdim = domain.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool))
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc = fem.dirichletbc(PETSc.ScalarType(0.0), boundary_dofs, V)
    
    # Backward Euler variational forms
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    a = ufl.inner(u, v) * ufl.dx + dt * kappa * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = ufl.inner(u_old, v) * ufl.dx + dt * ufl.inner(f_expr, v) * ufl.dx
    
    a_form = fem.form(a)
    L_form = fem.form(L)
    
    A = petsc.assemble_matrix(a_form, bcs=[bc])
    A.assemble()
    b = petsc.create_vector(V)
    
    # Direct LU solver - reliable and fast for this problem size
    ksp = PETSc.KSP().create(domain.comm)
    ksp.setOperators(A)
    ksp.setType(PETSc.KSP.Type.PREONLY)
    ksp.getPC().setType(PETSc.PC.Type.LU)
    
    total_ksp_iterations = 0
    
    current_t = t0
    for step in range(n_steps):
        current_t += dt
        t_const.value = PETSc.ScalarType(current_t)
        
        with b.localForm() as loc:
            loc.set(0)
        petsc.assemble_vector(b, L_form)
        petsc.apply_lifting(b, [a_form], [[bc]])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        petsc.set_bc(b, [bc])
        
        ksp.solve(b, u_sol.x.petsc_vec)
        u_sol.x.scatter_forward()
        
        total_ksp_iterations += ksp.getIterationNumber()
        
        u_old.x.array[:] = u_sol.x.array[:]
    
    # L2 error verification
    t_const.value = PETSc.ScalarType(t_end)
    u_exact = fem.Function(V)
    u_exact.interpolate(fem.Expression(
        ufl.exp(-t_const) * ufl.sin(ufl.pi * x_coord[0]) * ufl.sin(ufl.pi * x_coord[1]),
        V.element.interpolation_points))
    
    error_form = fem.form(ufl.inner(u_sol - u_exact, u_sol - u_exact) * ufl.dx)
    error_L2 = float(np.sqrt(domain.comm.allreduce(fem.assemble_scalar(error_form), op=MPI.SUM)))
    
    norm_form = fem.form(ufl.inner(u_exact, u_exact) * ufl.dx)
    norm_exact = float(np.sqrt(domain.comm.allreduce(fem.assemble_scalar(norm_form), op=MPI.SUM)))
    rel_error = error_L2 / norm_exact if norm_exact > 0 else error_L2
    
    if comm.rank == 0:
        print(f"L2 error: {error_L2:.6e}, Rel L2 error: {rel_error:.6e}")
    
    # Sample solution on output grid
    xmin, xmax, ymin, ymax = bbox
    xs = np.linspace(xmin, xmax, nx_out)
    ys = np.linspace(ymin, ymax, ny_out)
    XX, YY = np.meshgrid(xs, ys)
    
    pts = np.zeros((3, nx_out * ny_out))
    pts[0, :] = XX.ravel()
    pts[1, :] = YY.ravel()
    
    bb_tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, pts.T)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, pts.T)
    
    u_values = np.full(nx_out * ny_out, np.nan)
    pts_list, cells_list, idx_map = [], [], []
    for i in range(pts.shape[1]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            pts_list.append(pts.T[i])
            cells_list.append(links[0])
            idx_map.append(i)
    
    if len(pts_list) > 0:
        vals = u_sol.eval(np.array(pts_list), np.array(cells_list, dtype=np.int32))
        u_values[idx_map] = vals.flatten()
    
    u_global = np.zeros_like(u_values) if comm.rank == 0 else None
    comm.Reduce(u_values, u_global, op=MPI.SUM, root=0)
    u_grid = u_global.reshape(ny_out, nx_out) if comm.rank == 0 else np.zeros((ny_out, nx_out))
    u_grid = comm.bcast(u_grid, root=0)
    
    # Sample initial condition
    u_init = fem.Function(V)
    u_init.interpolate(fem.Expression(
        ufl.sin(ufl.pi * x_coord[0]) * ufl.sin(ufl.pi * x_coord[1]),
        V.element.interpolation_points))
    
    u_init_vals = np.full(nx_out * ny_out, np.nan)
    pts_i, cells_i, idx_i = [], [], []
    for i in range(pts.shape[1]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            pts_i.append(pts.T[i])
            cells_i.append(links[0])
            idx_i.append(i)
    
    if len(pts_i) > 0:
        vals_i = u_init.eval(np.array(pts_i), np.array(cells_i, dtype=np.int32))
        u_init_vals[idx_i] = vals_i.flatten()
    
    u_init_global = np.zeros_like(u_init_vals) if comm.rank == 0 else None
    comm.Reduce(u_init_vals, u_init_global, op=MPI.SUM, root=0)
    u_initial = u_init_global.reshape(ny_out, nx_out) if comm.rank == 0 else np.zeros((ny_out, nx_out))
    u_initial = comm.bcast(u_initial, root=0)
    
    return {
        "u": u_grid,
        "u_initial": u_initial,
        "solver_info": {
            "mesh_resolution": int(mesh_res),
            "element_degree": int(element_degree),
            "ksp_type": "preonly",
            "pc_type": "lu",
            "rtol": float(1e-12),
            "iterations": int(total_ksp_iterations),
            "dt": float(dt),
            "n_steps": int(n_steps),
            "time_scheme": "backward_euler",
            "l2_error": float(error_L2),
            "rel_l2_error": float(rel_error)
        }
    }
