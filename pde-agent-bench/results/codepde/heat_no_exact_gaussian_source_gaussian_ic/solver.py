import numpy as np
from dolfinx import mesh, fem, default_scalar_type, geometry
from dolfinx.fem import petsc
from mpi4py import MPI
import ufl
from petsc4py import PETSc


def solve(case_spec: dict) -> dict:
    # 1. Parse case_spec
    pde = case_spec.get("pde", {})
    if not pde:
        pde = case_spec.get("oracle_config", {}).get("pde", {})
    
    # Extract parameters
    kappa = float(pde.get("coefficients", {}).get("kappa", 1.0))
    
    time_params = pde.get("time", {})
    t_end = float(time_params.get("t_end", 0.1))
    dt_suggested = float(time_params.get("dt", 0.02))
    scheme = time_params.get("scheme", "backward_euler")
    
    # 2. Create mesh
    nx = ny = 80
    domain = mesh.create_unit_square(MPI.COMM_WORLD, nx, ny, cell_type=mesh.CellType.triangle)
    
    # 3. Function space
    degree = 1
    V = fem.functionspace(domain, ("Lagrange", degree))
    
    # Spatial coordinate
    x = ufl.SpatialCoordinate(domain)
    
    # 4. Source term: f = exp(-200*((x-0.3)**2 + (y-0.7)**2))
    f_expr = ufl.exp(-200.0 * ((x[0] - 0.3)**2 + (x[1] - 0.7)**2))
    
    # 5. Initial condition: u0 = exp(-120*((x-0.6)**2 + (y-0.4)**2))
    u_n = fem.Function(V, name="u_n")
    u_n.interpolate(lambda X: np.exp(-120.0 * ((X[0] - 0.6)**2 + (X[1] - 0.4)**2)))
    
    # Store initial condition for output
    u_initial_func = fem.Function(V)
    u_initial_func.x.array[:] = u_n.x.array[:]
    
    # 6. Time stepping parameters
    dt = dt_suggested
    n_steps = int(np.ceil(t_end / dt))
    dt = t_end / n_steps  # adjust to hit t_end exactly
    
    dt_const = fem.Constant(domain, PETSc.ScalarType(dt))
    kappa_const = fem.Constant(domain, PETSc.ScalarType(kappa))
    
    # 7. Variational forms - Backward Euler
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    
    # (u - u_n)/dt - kappa * laplacian(u) = f
    # Weak form: u*v*dx + dt*kappa*inner(grad(u), grad(v))*dx = u_n*v*dx + dt*f*v*dx
    a = u * v * ufl.dx + dt_const * kappa_const * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = u_n * v * ufl.dx + dt_const * f_expr * v * ufl.dx
    
    # 8. Boundary conditions: u = g on boundary
    # g = 0 (homogeneous Dirichlet - the Gaussians decay to ~0 at boundary)
    tdim = domain.topology.dim
    fdim = tdim - 1
    
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda X: np.ones(X.shape[1], dtype=bool)
    )
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc = fem.dirichletbc(PETSc.ScalarType(0.0), boundary_dofs, V)
    
    # 9. Assemble and set up solver for time loop
    a_form = fem.form(a)
    L_form = fem.form(L)
    
    A = petsc.assemble_matrix(a_form, bcs=[bc])
    A.assemble()
    
    b = fem.Function(V)
    b_vec = b.x.petsc_vec
    
    solver = PETSc.KSP().create(domain.comm)
    solver.setOperators(A)
    solver.setType(PETSc.KSP.Type.CG)
    pc = solver.getPC()
    pc.setType(PETSc.PC.Type.ILU)
    solver.setTolerances(rtol=1e-10, atol=1e-12, max_it=1000)
    solver.setUp()
    
    u_sol = fem.Function(V)
    u_sol.x.array[:] = u_n.x.array[:]
    
    # 10. Time loop
    total_iterations = 0
    t = 0.0
    
    for step in range(n_steps):
        t += dt
        
        # Assemble RHS
        with b_vec.localForm() as loc:
            loc.set(0.0)
        petsc.assemble_vector(b_vec, L_form)
        petsc.apply_lifting(b_vec, [a_form], bcs=[[bc]])
        b_vec.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        petsc.set_bc(b_vec, [bc])
        
        # Solve
        solver.solve(b_vec, u_sol.x.petsc_vec)
        u_sol.x.scatter_forward()
        
        total_iterations += solver.getIterationNumber()
        
        # Update previous solution
        u_n.x.array[:] = u_sol.x.array[:]
    
    # 11. Extract solution on 50x50 uniform grid
    n_grid = 50
    xs = np.linspace(0.0, 1.0, n_grid)
    ys = np.linspace(0.0, 1.0, n_grid)
    XX, YY = np.meshgrid(xs, ys, indexing='ij')
    
    points_2d = np.stack([XX.ravel(), YY.ravel()], axis=0)
    points_3d = np.vstack([points_2d, np.zeros((1, points_2d.shape[1]))])
    
    bb_tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, points_3d.T)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points_3d.T)
    
    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    
    for i in range(points_3d.shape[1]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(points_3d[:, i])
            cells_on_proc.append(links[0])
            eval_map.append(i)
    
    u_values = np.zeros(points_3d.shape[1])
    u_initial_values = np.zeros(points_3d.shape[1])
    
    if len(points_on_proc) > 0:
        pts_arr = np.array(points_on_proc)
        cells_arr = np.array(cells_on_proc, dtype=np.int32)
        vals = u_sol.eval(pts_arr, cells_arr)
        u_values[eval_map] = vals.flatten()
        
        vals_init = u_initial_func.eval(pts_arr, cells_arr)
        u_initial_values[eval_map] = vals_init.flatten()
    
    u_grid = u_values.reshape((n_grid, n_grid))
    u_initial_grid = u_initial_values.reshape((n_grid, n_grid))
    
    # Cleanup
    solver.destroy()
    A.destroy()
    
    return {
        "u": u_grid,
        "u_initial": u_initial_grid,
        "solver_info": {
            "mesh_resolution": nx,
            "element_degree": degree,
            "ksp_type": "cg",
            "pc_type": "ilu",
            "rtol": 1e-10,
            "iterations": total_iterations,
            "dt": dt,
            "n_steps": n_steps,
            "time_scheme": "backward_euler",
        }
    }