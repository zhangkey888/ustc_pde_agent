import numpy as np
from dolfinx import mesh, fem, default_scalar_type, geometry
from dolfinx.fem import petsc
from mpi4py import MPI
import ufl
from petsc4py import PETSc


def solve(case_spec: dict) -> dict:
    # 1. Parse case_spec
    pde = case_spec.get("pde", case_spec.get("oracle_config", {}).get("pde", {}))
    
    # Time parameters
    time_params = pde.get("time", {})
    t_end = time_params.get("t_end", 0.1)
    dt_suggested = time_params.get("dt", 0.02)
    scheme = time_params.get("scheme", "backward_euler")
    
    # Use a smaller dt for better accuracy
    dt = 0.005
    n_steps = int(round(t_end / dt))
    dt = t_end / n_steps  # adjust to hit t_end exactly
    
    # Mesh resolution
    nx = ny = 80
    
    # 2. Create mesh
    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_square(comm, nx, ny, cell_type=mesh.CellType.triangle)
    
    # 3. Function space
    degree = 1
    V = fem.functionspace(domain, ("Lagrange", degree))
    
    # Spatial coordinate
    x = ufl.SpatialCoordinate(domain)
    
    # 4. Define kappa
    kappa_expr = 1.0 + 0.6 * ufl.sin(2 * ufl.pi * x[0]) * ufl.sin(2 * ufl.pi * x[1])
    
    # 5. Source term
    f_expr = 1.0 + ufl.sin(2 * ufl.pi * x[0]) * ufl.cos(2 * ufl.pi * x[1])
    
    # 6. Initial condition
    u_n = fem.Function(V, name="u_n")
    u_n.interpolate(lambda X: np.sin(np.pi * X[0]) * np.sin(np.pi * X[1]))
    
    # 7. Boundary conditions (u=0 on boundary since sin(pi*x)*sin(pi*y)=0 on boundary,
    # and the problem evolves; for the heat equation with g on boundary,
    # we check if g is specified. Here g=0 since IC is 0 on boundary)
    tdim = domain.topology.dim
    fdim = tdim - 1
    
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda X: np.ones(X.shape[1], dtype=bool)
    )
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    
    # BC value: g = 0 (since sin(pi*x)*sin(pi*y) = 0 on boundary and no other BC specified)
    bc_val = fem.Function(V)
    bc_val.x.array[:] = 0.0
    bc = fem.dirichletbc(bc_val, boundary_dofs)
    
    # 8. Variational form for backward Euler
    # (u - u_n)/dt - div(kappa * grad(u)) = f
    # Weak form: (u - u_n)/dt * v dx + kappa * grad(u) . grad(v) dx = f * v dx
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    
    dt_const = fem.Constant(domain, default_scalar_type(dt))
    
    a = (u * v * ufl.dx 
         + dt_const * kappa_expr * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx)
    L = (u_n * v * ufl.dx 
         + dt_const * f_expr * v * ufl.dx)
    
    # 9. Compile forms
    a_form = fem.form(a)
    L_form = fem.form(L)
    
    # 10. Assemble matrix (constant in time since kappa doesn't depend on t)
    A = petsc.assemble_matrix(a_form, bcs=[bc])
    A.assemble()
    
    # Create RHS vector
    b = fem.Function(V)
    
    # 11. Setup KSP solver
    solver = PETSc.KSP().create(domain.comm)
    solver.setOperators(A)
    solver.setType(PETSc.KSP.Type.CG)
    pc = solver.getPC()
    pc.setType(PETSc.PC.Type.HYPRE)
    solver.setTolerances(rtol=1e-10, atol=1e-12, max_it=1000)
    solver.setUp()
    
    # Solution function
    u_sol = fem.Function(V)
    u_sol.x.array[:] = u_n.x.array[:]
    
    total_iterations = 0
    
    # 12. Time stepping
    for step in range(n_steps):
        # Assemble RHS
        with b.x.petsc_vec.localForm() as loc:
            loc.set(0.0)
        petsc.assemble_vector(b.x.petsc_vec, L_form)
        
        # Apply lifting for BCs
        petsc.apply_lifting(b.x.petsc_vec, [a_form], bcs=[[bc]])
        b.x.petsc_vec.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        petsc.set_bc(b.x.petsc_vec, [bc])
        
        # Solve
        solver.solve(b.x.petsc_vec, u_sol.x.petsc_vec)
        u_sol.x.scatter_forward()
        
        total_iterations += solver.getIterationNumber()
        
        # Update u_n
        u_n.x.array[:] = u_sol.x.array[:]
    
    # 13. Extract initial condition for output
    u_initial_func = fem.Function(V)
    u_initial_func.interpolate(lambda X: np.sin(np.pi * X[0]) * np.sin(np.pi * X[1]))
    
    # 14. Evaluate on 50x50 grid
    nx_out, ny_out = 50, 50
    xs = np.linspace(0.0, 1.0, nx_out)
    ys = np.linspace(0.0, 1.0, ny_out)
    XX, YY = np.meshgrid(xs, ys, indexing='ij')
    points_2d = np.column_stack([XX.ravel(), YY.ravel()])
    points_3d = np.zeros((points_2d.shape[0], 3))
    points_3d[:, :2] = points_2d
    
    # Build bounding box tree
    bb_tree = geometry.bb_tree(domain, domain.topology.dim)
    
    cell_candidates = geometry.compute_collisions_points(bb_tree, points_3d)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points_3d)
    
    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    for i in range(points_3d.shape[0]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(points_3d[i])
            cells_on_proc.append(links[0])
            eval_map.append(i)
    
    u_values = np.full(points_3d.shape[0], np.nan)
    u_init_values = np.full(points_3d.shape[0], np.nan)
    
    if len(points_on_proc) > 0:
        pts_arr = np.array(points_on_proc)
        cells_arr = np.array(cells_on_proc, dtype=np.int32)
        vals = u_sol.eval(pts_arr, cells_arr)
        u_values[eval_map] = vals.flatten()
        vals_init = u_initial_func.eval(pts_arr, cells_arr)
        u_init_values[eval_map] = vals_init.flatten()
    
    u_grid = u_values.reshape((nx_out, ny_out))
    u_init_grid = u_init_values.reshape((nx_out, ny_out))
    
    # Cleanup
    solver.destroy()
    A.destroy()
    
    return {
        "u": u_grid,
        "u_initial": u_init_grid,
        "solver_info": {
            "mesh_resolution": nx,
            "element_degree": degree,
            "ksp_type": "cg",
            "pc_type": "hypre",
            "rtol": 1e-10,
            "iterations": total_iterations,
            "dt": dt,
            "n_steps": n_steps,
            "time_scheme": "backward_euler",
        }
    }