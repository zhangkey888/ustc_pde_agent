import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc


def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD
    
    # Parameters
    kappa = 0.5
    t_end = 0.2
    dt = 0.02
    n_steps = int(round(t_end / dt))
    mesh_resolution = 64
    element_degree = 2
    
    # Create mesh
    domain = mesh.create_unit_square(comm, mesh_resolution, mesh_resolution, cell_type=mesh.CellType.triangle)
    
    # Function space
    V = fem.functionspace(domain, ("Lagrange", element_degree))
    
    # Spatial coordinates
    x = ufl.SpatialCoordinate(domain)
    
    # Time parameter
    t_param = fem.Constant(domain, PETSc.ScalarType(0.0))
    
    # Exact solution: u = exp(-2*t)*cos(pi*x)*cos(pi*y)
    pi = np.pi
    u_exact_ufl = ufl.exp(-2.0 * t_param) * ufl.cos(ufl.pi * x[0]) * ufl.cos(ufl.pi * x[1])
    
    # Source term: f = du/dt - kappa * laplacian(u)
    # du/dt = -2*exp(-2t)*cos(pi*x)*cos(pi*y)
    # laplacian(u) = -2*pi^2*exp(-2t)*cos(pi*x)*cos(pi*y)
    # f = -2*exp(-2t)*cos(pi*x)*cos(pi*y) - kappa*(-2*pi^2)*exp(-2t)*cos(pi*x)*cos(pi*y)
    # f = exp(-2t)*cos(pi*x)*cos(pi*y)*(-2 + kappa*2*pi^2)
    # f = exp(-2t)*cos(pi*x)*cos(pi*y)*(-2 + pi^2)  since kappa=0.5
    f_ufl = ufl.exp(-2.0 * t_param) * ufl.cos(ufl.pi * x[0]) * ufl.cos(ufl.pi * x[1]) * (-2.0 + kappa * 2.0 * ufl.pi**2)
    
    # Trial and test functions
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    
    # Previous solution
    u_n = fem.Function(V)
    
    # Set initial condition: u(x,0) = cos(pi*x)*cos(pi*y)
    t_param.value = 0.0
    u_n.interpolate(fem.Expression(u_exact_ufl, V.element.interpolation_points))
    
    # Store initial condition for output
    u_initial_func = fem.Function(V)
    u_initial_func.x.array[:] = u_n.x.array[:]
    
    # Backward Euler: (u - u_n)/dt - kappa*laplacian(u) = f(t_{n+1})
    # Weak form: (u/dt)*v*dx + kappa*grad(u)·grad(v)*dx = (u_n/dt)*v*dx + f*v*dx
    dt_const = fem.Constant(domain, PETSc.ScalarType(dt))
    kappa_const = fem.Constant(domain, PETSc.ScalarType(kappa))
    
    a = (u * v / dt_const) * ufl.dx + kappa_const * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = (u_n * v / dt_const) * ufl.dx + f_ufl * v * ufl.dx
    
    # Boundary conditions
    tdim = domain.topology.dim
    fdim = tdim - 1
    
    # All boundary
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
    )
    
    u_bc = fem.Function(V)
    dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc = fem.dirichletbc(u_bc, dofs)
    
    # Compile forms
    a_form = fem.form(a)
    L_form = fem.form(L)
    
    # Assemble matrix (constant in time for backward Euler with constant kappa)
    A = petsc.assemble_matrix(a_form, bcs=[bc])
    A.assemble()
    
    # Create RHS vector
    b = fem.petsc.create_vector(L_form)
    
    # Solution function
    u_sol = fem.Function(V)
    
    # Setup KSP solver
    solver = PETSc.KSP().create(domain.comm)
    solver.setOperators(A)
    solver.setType(PETSc.KSP.Type.CG)
    pc = solver.getPC()
    pc.setType(PETSc.PC.Type.HYPRE)
    solver.setTolerances(rtol=1e-10, atol=1e-12, max_it=1000)
    solver.setUp()
    
    total_iterations = 0
    
    # Time stepping
    for step in range(n_steps):
        t_current = (step + 1) * dt
        t_param.value = t_current
        
        # Update boundary condition
        u_bc.interpolate(fem.Expression(u_exact_ufl, V.element.interpolation_points))
        
        # Assemble RHS
        with b.localForm() as loc:
            loc.set(0.0)
        petsc.assemble_vector(b, L_form)
        petsc.apply_lifting(b, [a_form], bcs=[[bc]])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        petsc.set_bc(b, [bc])
        
        # Solve
        solver.solve(b, u_sol.x.petsc_vec)
        u_sol.x.scatter_forward()
        
        total_iterations += solver.getIterationNumber()
        
        # Update previous solution
        u_n.x.array[:] = u_sol.x.array[:]
    
    # Evaluate on 50x50 grid
    nx_out, ny_out = 50, 50
    xs = np.linspace(0, 1, nx_out)
    ys = np.linspace(0, 1, ny_out)
    XX, YY = np.meshgrid(xs, ys, indexing='ij')
    points = np.zeros((3, nx_out * ny_out))
    points[0, :] = XX.ravel()
    points[1, :] = YY.ravel()
    
    bb_tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, points.T)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points.T)
    
    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    for i in range(points.shape[1]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(points[:, i])
            cells_on_proc.append(links[0])
            eval_map.append(i)
    
    u_values = np.full(nx_out * ny_out, np.nan)
    if len(points_on_proc) > 0:
        pts_arr = np.array(points_on_proc)
        cells_arr = np.array(cells_on_proc, dtype=np.int32)
        vals = u_sol.eval(pts_arr, cells_arr)
        u_values[eval_map] = vals.flatten()
    
    u_grid = u_values.reshape((nx_out, ny_out))
    
    # Also evaluate initial condition
    u_init_values = np.full(nx_out * ny_out, np.nan)
    if len(points_on_proc) > 0:
        vals_init = u_initial_func.eval(pts_arr, cells_arr)
        u_init_values[eval_map] = vals_init.flatten()
    
    u_init_grid = u_init_values.reshape((nx_out, ny_out))
    
    # Cleanup
    solver.destroy()
    A.destroy()
    b.destroy()
    
    return {
        "u": u_grid,
        "u_initial": u_init_grid,
        "solver_info": {
            "mesh_resolution": mesh_resolution,
            "element_degree": element_degree,
            "ksp_type": "cg",
            "pc_type": "hypre",
            "rtol": 1e-10,
            "iterations": total_iterations,
            "dt": dt,
            "n_steps": n_steps,
            "time_scheme": "backward_euler",
        }
    }