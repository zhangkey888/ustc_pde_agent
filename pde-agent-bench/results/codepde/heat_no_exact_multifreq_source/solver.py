import numpy as np
from dolfinx import mesh, fem, default_scalar_type, geometry
from dolfinx.fem import petsc
from mpi4py import MPI
import ufl
from petsc4py import PETSc


def solve(case_spec: dict) -> dict:
    # 1. Parse case_spec
    pde = case_spec.get("pde", {})
    
    kappa = 1.0
    t_end = 0.12
    dt = 0.02
    
    # Try to extract from case_spec
    coeffs = pde.get("coefficients", {})
    if "kappa" in coeffs:
        kappa = float(coeffs["kappa"])
    
    time_params = pde.get("time", {})
    if "t_end" in time_params:
        t_end = float(time_params["t_end"])
    if "dt" in time_params:
        dt = float(time_params["dt"])
    
    # Use a finer dt for accuracy
    dt_use = 0.005
    n_steps = int(round(t_end / dt_use))
    dt_use = t_end / n_steps  # adjust to hit t_end exactly
    
    # 2. Create mesh - use fine enough mesh for multi-frequency source
    nx, ny = 80, 80
    domain = mesh.create_unit_square(MPI.COMM_WORLD, nx, ny, cell_type=mesh.CellType.triangle)
    
    # 3. Function space - use P2 for better accuracy with high-frequency terms
    degree = 2
    V = fem.functionspace(domain, ("Lagrange", degree))
    
    # 4. Define functions
    u_n = fem.Function(V)  # solution at previous time step
    u_n.x.array[:] = 0.0  # initial condition u0 = 0
    
    # Store initial condition for output
    u_initial_func = fem.Function(V)
    u_initial_func.x.array[:] = 0.0
    
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    
    # Source term: f = sin(5*pi*x)*sin(3*pi*y) + 0.5*sin(9*pi*x)*sin(7*pi*y)
    x = ufl.SpatialCoordinate(domain)
    pi = np.pi
    f = (ufl.sin(5 * pi * x[0]) * ufl.sin(3 * pi * x[1]) 
         + 0.5 * ufl.sin(9 * pi * x[0]) * ufl.sin(7 * pi * x[1]))
    
    # Backward Euler: (u - u_n)/dt - kappa * laplacian(u) = f
    # Weak form: (u, v)/dt + kappa*(grad(u), grad(v)) = (u_n, v)/dt + (f, v)
    dt_const = fem.Constant(domain, PETSc.ScalarType(dt_use))
    kappa_const = fem.Constant(domain, PETSc.ScalarType(kappa))
    
    a = (ufl.inner(u, v) / dt_const + kappa_const * ufl.inner(ufl.grad(u), ufl.grad(v))) * ufl.dx
    L = (ufl.inner(u_n, v) / dt_const + ufl.inner(f, v)) * ufl.dx
    
    # 5. Boundary conditions: u = g on boundary
    # g = 0 (homogeneous Dirichlet, since initial condition is 0 and source has sin terms that vanish on boundary)
    tdim = domain.topology.dim
    fdim = tdim - 1
    
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
    )
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc = fem.dirichletbc(PETSc.ScalarType(0.0), boundary_dofs, V)
    
    # 6. Assemble and solve with manual time-stepping
    a_form = fem.form(a)
    L_form = fem.form(L)
    
    A = petsc.assemble_matrix(a_form, bcs=[bc])
    A.assemble()
    
    b = fem.Function(V)
    
    solver = PETSc.KSP().create(domain.comm)
    solver.setOperators(A)
    solver.setType(PETSc.KSP.Type.CG)
    pc = solver.getPC()
    pc.setType(PETSc.PC.Type.HYPRE)
    solver.setTolerances(rtol=1e-10, atol=1e-12, max_it=2000)
    solver.setUp()
    
    uh = fem.Function(V)
    
    total_iterations = 0
    
    for step in range(n_steps):
        # Assemble RHS
        b_vec = petsc.create_vector(L_form)
        with b_vec.localForm() as loc:
            loc.set(0)
        petsc.assemble_vector(b_vec, L_form)
        petsc.apply_lifting(b_vec, [a_form], bcs=[[bc]])
        b_vec.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        petsc.set_bc(b_vec, [bc])
        
        # Solve
        solver.solve(b_vec, uh.x.petsc_vec)
        uh.x.scatter_forward()
        
        total_iterations += solver.getIterationNumber()
        
        # Update previous solution
        u_n.x.array[:] = uh.x.array[:]
        
        b_vec.destroy()
    
    # 7. Extract solution on 50x50 uniform grid
    n_grid = 50
    xs = np.linspace(0.0, 1.0, n_grid)
    ys = np.linspace(0.0, 1.0, n_grid)
    X, Y = np.meshgrid(xs, ys, indexing='ij')
    
    points_2d = np.zeros((3, n_grid * n_grid))
    points_2d[0, :] = X.ravel()
    points_2d[1, :] = Y.ravel()
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
    
    u_values = np.zeros(n_grid * n_grid)
    u_initial_values = np.zeros(n_grid * n_grid)
    
    if len(points_on_proc) > 0:
        pts = np.array(points_on_proc)
        cls = np.array(cells_on_proc, dtype=np.int32)
        vals = uh.eval(pts, cls)
        u_values[eval_map] = vals.flatten()
        
        vals_init = u_initial_func.eval(pts, cls)
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
            "pc_type": "hypre",
            "rtol": 1e-10,
            "iterations": total_iterations,
            "dt": dt_use,
            "n_steps": n_steps,
            "time_scheme": "backward_euler",
        }
    }