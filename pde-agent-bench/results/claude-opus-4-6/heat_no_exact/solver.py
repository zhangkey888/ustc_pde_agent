import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc


def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD
    
    # Parameters
    nx = ny = 64
    degree = 1
    kappa = 1.0
    t_end = 0.1
    dt = 0.02
    n_steps = int(round(t_end / dt))
    
    # Create mesh
    domain = mesh.create_unit_square(comm, nx, ny, cell_type=mesh.CellType.triangle)
    tdim = domain.topology.dim
    fdim = tdim - 1
    
    # Function space
    V = fem.functionspace(domain, ("Lagrange", degree))
    
    # Trial and test functions
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    
    # Previous solution
    u_n = fem.Function(V)
    
    # Spatial coordinate
    x = ufl.SpatialCoordinate(domain)
    
    # Source term: f = sin(pi*x)*cos(pi*y)
    f = ufl.sin(ufl.pi * x[0]) * ufl.cos(ufl.pi * x[1])
    
    # Initial condition: u0 = sin(pi*x)*sin(pi*y)
    u_n.interpolate(lambda X: np.sin(np.pi * X[0]) * np.sin(np.pi * X[1]))
    
    # Store initial condition for output
    u_initial_func = fem.Function(V)
    u_initial_func.interpolate(lambda X: np.sin(np.pi * X[0]) * np.sin(np.pi * X[1]))
    
    # Boundary conditions: u = g on ∂Ω
    # g = 0 (since sin(pi*x)*sin(pi*y) = 0 on boundary of unit square for homogeneous Dirichlet)
    # Actually, the problem says u = g on boundary. For heat_no_exact, g is typically 0.
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
    )
    dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc = fem.dirichletbc(PETSc.ScalarType(0.0), dofs, V)
    bcs = [bc]
    
    # Backward Euler: (u - u_n)/dt - kappa * laplacian(u) = f
    # Weak form: (u, v)/dt + kappa*(grad(u), grad(v)) = (u_n, v)/dt + (f, v)
    dt_const = fem.Constant(domain, PETSc.ScalarType(dt))
    kappa_const = fem.Constant(domain, PETSc.ScalarType(kappa))
    
    a = (u * v / dt_const + kappa_const * ufl.inner(ufl.grad(u), ufl.grad(v))) * ufl.dx
    L = (u_n * v / dt_const + f * v) * ufl.dx
    
    # Compile forms
    a_form = fem.form(a)
    L_form = fem.form(L)
    
    # Assemble matrix (constant in time for this problem)
    A = petsc.assemble_matrix(a_form, bcs=bcs)
    A.assemble()
    
    # Create RHS vector
    b = petsc.create_vector(L_form)
    
    # Solution function
    u_sol = fem.Function(V)
    
    # Setup KSP solver
    solver = PETSc.KSP().create(domain.comm)
    solver.setOperators(A)
    solver.setType(PETSc.KSP.Type.CG)
    pc = solver.getPC()
    pc.setType(PETSc.PC.Type.ILU)
    solver.setTolerances(rtol=1e-8, atol=1e-12, max_it=1000)
    solver.setUp()
    
    total_iterations = 0
    
    # Time stepping
    for step in range(n_steps):
        # Assemble RHS
        with b.localForm() as loc:
            loc.set(0.0)
        petsc.assemble_vector(b, L_form)
        petsc.apply_lifting(b, [a_form], bcs=[bcs])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        petsc.set_bc(b, bcs)
        
        # Solve
        solver.solve(b, u_sol.x.petsc_vec)
        u_sol.x.scatter_forward()
        
        total_iterations += solver.getIterationNumber()
        
        # Update previous solution
        u_n.x.array[:] = u_sol.x.array[:]
    
    # Sample solution on 50x50 grid
    nx_out, ny_out = 50, 50
    xs = np.linspace(0, 1, nx_out)
    ys = np.linspace(0, 1, ny_out)
    XX, YY = np.meshgrid(xs, ys, indexing='ij')
    points = np.zeros((3, nx_out * ny_out))
    points[0, :] = XX.ravel()
    points[1, :] = YY.ravel()
    
    # Point evaluation
    bb_tree = geometry.bb_tree(domain, tdim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, points.T)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points.T)
    
    def eval_on_grid(func):
        points_on_proc = []
        cells_on_proc = []
        eval_map = []
        for i in range(points.shape[1]):
            links = colliding_cells.links(i)
            if len(links) > 0:
                points_on_proc.append(points[:, i])
                cells_on_proc.append(links[0])
                eval_map.append(i)
        
        values = np.full(points.shape[1], np.nan)
        if len(points_on_proc) > 0:
            pts = np.array(points_on_proc)
            cls = np.array(cells_on_proc, dtype=np.int32)
            vals = func.eval(pts, cls)
            values[eval_map] = vals.flatten()
        return values.reshape((nx_out, ny_out))
    
    u_grid = eval_on_grid(u_sol)
    u_initial_grid = eval_on_grid(u_initial_func)
    
    # Cleanup
    solver.destroy()
    A.destroy()
    b.destroy()
    
    return {
        "u": u_grid,
        "u_initial": u_initial_grid,
        "solver_info": {
            "mesh_resolution": nx,
            "element_degree": degree,
            "ksp_type": "cg",
            "pc_type": "ilu",
            "rtol": 1e-8,
            "iterations": total_iterations,
            "dt": dt,
            "n_steps": n_steps,
            "time_scheme": "backward_euler",
        }
    }