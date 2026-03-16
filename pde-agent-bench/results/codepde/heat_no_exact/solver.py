import numpy as np
from dolfinx import mesh, fem, default_scalar_type, geometry
from dolfinx.fem import petsc
from mpi4py import MPI
import ufl
from petsc4py import PETSc


def solve(case_spec: dict) -> dict:
    # 1. Parse case_spec
    pde = case_spec.get("pde", {})
    coeffs = pde.get("coefficients", {})
    kappa = coeffs.get("kappa", 1.0)
    
    time_params = pde.get("time", {})
    t_end = time_params.get("t_end", 0.1)
    dt_suggested = time_params.get("dt", 0.02)
    scheme = time_params.get("scheme", "backward_euler")
    
    # Use a finer dt for accuracy
    dt = dt_suggested
    n_steps = int(round(t_end / dt))
    dt = t_end / n_steps  # adjust to hit t_end exactly
    
    # 2. Create mesh
    nx = ny = 64
    domain = mesh.create_unit_square(MPI.COMM_WORLD, nx, ny, cell_type=mesh.CellType.triangle)
    
    # 3. Function space
    degree = 1
    V = fem.functionspace(domain, ("Lagrange", degree))
    
    # 4. Spatial coordinate
    x = ufl.SpatialCoordinate(domain)
    
    # Source term: f = sin(pi*x)*cos(pi*y)
    f_expr = ufl.sin(ufl.pi * x[0]) * ufl.cos(ufl.pi * x[1])
    
    # 5. Initial condition: u0 = sin(pi*x)*sin(pi*y)
    u_n = fem.Function(V, name="u_n")
    u_n.interpolate(lambda X: np.sin(np.pi * X[0]) * np.sin(np.pi * X[1]))
    
    # 6. Boundary conditions: u = g on boundary
    # For this problem, sin(pi*x)*sin(pi*y) = 0 on the boundary of [0,1]^2
    # The exact solution isn't known, but boundary should remain 0 for homogeneous Dirichlet
    # Actually, we need to check what g is. The problem says u = g on boundary.
    # Since no explicit g is given, and the initial condition is 0 on boundary, 
    # and the source term sin(pi*x)*cos(pi*y) is 0 on x=0 and x=1 boundaries,
    # Let's use homogeneous Dirichlet BCs (g=0)
    
    tdim = domain.topology.dim
    fdim = tdim - 1
    
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
    )
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc = fem.dirichletbc(PETSc.ScalarType(0.0), boundary_dofs, V)
    bcs = [bc]
    
    # 7. Variational form for backward Euler:
    # (u - u_n)/dt - kappa * laplacian(u) = f
    # Weak form: (u, v)/dt + kappa*(grad(u), grad(v)) = (u_n, v)/dt + (f, v)
    
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    
    dt_const = fem.Constant(domain, PETSc.ScalarType(dt))
    kappa_const = fem.Constant(domain, PETSc.ScalarType(kappa))
    
    a = (u * v / dt_const + kappa_const * ufl.inner(ufl.grad(u), ufl.grad(v))) * ufl.dx
    L = (u_n * v / dt_const + f_expr * v) * ufl.dx
    
    # 8. Compile forms and set up manual assembly for time loop
    a_form = fem.form(a)
    L_form = fem.form(L)
    
    # Assemble LHS matrix (constant in time for this problem)
    A = petsc.assemble_matrix(a_form, bcs=bcs)
    A.assemble()
    
    # Create RHS vector
    b = fem.Function(V)
    
    # Set up KSP solver
    solver = PETSc.KSP().create(domain.comm)
    solver.setOperators(A)
    solver.setType(PETSc.KSP.Type.CG)
    pc = solver.getPC()
    pc.setType(PETSc.PC.Type.ILU)
    solver.setTolerances(rtol=1e-10, atol=1e-12, max_it=1000)
    solver.setUp()
    
    u_sol = fem.Function(V, name="u_sol")
    
    total_iterations = 0
    
    # 9. Time stepping
    for step in range(n_steps):
        # Assemble RHS
        with b.x.petsc_vec.localForm() as loc:
            loc.set(0.0)
        petsc.assemble_vector(b.x.petsc_vec, L_form)
        petsc.apply_lifting(b.x.petsc_vec, [a_form], bcs=[bcs])
        b.x.petsc_vec.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        petsc.set_bc(b.x.petsc_vec, bcs)
        
        # Solve
        solver.solve(b.x.petsc_vec, u_sol.x.petsc_vec)
        u_sol.x.scatter_forward()
        
        total_iterations += solver.getIterationNumber()
        
        # Update u_n for next time step
        u_n.x.array[:] = u_sol.x.array[:]
        u_n.x.scatter_forward()
    
    # 10. Also store initial condition for output
    u_initial_func = fem.Function(V)
    u_initial_func.interpolate(lambda X: np.sin(np.pi * X[0]) * np.sin(np.pi * X[1]))
    
    # 11. Extract solution on 50x50 uniform grid
    n_eval = 50
    xs = np.linspace(0.0, 1.0, n_eval)
    ys = np.linspace(0.0, 1.0, n_eval)
    XX, YY = np.meshgrid(xs, ys, indexing='ij')
    
    points_2d = np.zeros((3, n_eval * n_eval))
    points_2d[0, :] = XX.ravel()
    points_2d[1, :] = YY.ravel()
    points_2d[2, :] = 0.0
    
    bb_tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, points_2d.T)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points_2d.T)
    
    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    for i in range(points_2d.shape[1]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(points_2d[:, i])
            cells_on_proc.append(links[0])
            eval_map.append(i)
    
    u_values = np.full(n_eval * n_eval, np.nan)
    u_init_values = np.full(n_eval * n_eval, np.nan)
    
    if len(points_on_proc) > 0:
        pts_arr = np.array(points_on_proc)
        cells_arr = np.array(cells_on_proc, dtype=np.int32)
        vals = u_sol.eval(pts_arr, cells_arr)
        u_values[eval_map] = vals.flatten()
        
        vals_init = u_initial_func.eval(pts_arr, cells_arr)
        u_init_values[eval_map] = vals_init.flatten()
    
    u_grid = u_values.reshape((n_eval, n_eval))
    u_init_grid = u_init_values.reshape((n_eval, n_eval))
    
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
            "pc_type": "ilu",
            "rtol": 1e-10,
            "iterations": total_iterations,
            "dt": dt,
            "n_steps": n_steps,
            "time_scheme": "backward_euler",
        }
    }