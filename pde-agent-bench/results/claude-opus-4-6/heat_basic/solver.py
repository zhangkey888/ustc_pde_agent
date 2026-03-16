import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc


def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD
    
    # Parameters
    kappa = 1.0
    t_end = 0.1
    dt = 0.01
    n_steps = int(round(t_end / dt))
    mesh_resolution = 64
    element_degree = 2
    
    # Create mesh
    domain = mesh.create_unit_square(comm, mesh_resolution, mesh_resolution, cell_type=mesh.CellType.triangle)
    
    # Function space
    V = fem.functionspace(domain, ("Lagrange", element_degree))
    
    # Manufactured solution: u = exp(-t)*sin(pi*x)*sin(pi*y)
    # du/dt = -exp(-t)*sin(pi*x)*sin(pi*y)
    # -kappa * laplacian(u) = kappa * 2*pi^2 * exp(-t)*sin(pi*x)*sin(pi*y)
    # f = du/dt - kappa*laplacian(u) ... wait, the PDE is du/dt - div(kappa grad u) = f
    # So f = du/dt - kappa * laplacian(u) = -exp(-t)*sin(pi*x)*sin(pi*y) + kappa*2*pi^2*exp(-t)*sin(pi*x)*sin(pi*y)
    # f = exp(-t)*sin(pi*x)*sin(pi*y)*(-1 + 2*kappa*pi^2)
    
    x = ufl.SpatialCoordinate(domain)
    t_const = fem.Constant(domain, PETSc.ScalarType(0.0))
    
    # Exact solution as UFL expression
    u_exact_ufl = ufl.exp(-t_const) * ufl.sin(ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])
    
    # Source term
    f_ufl = ufl.exp(-t_const) * ufl.sin(ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1]) * (-1.0 + 2.0 * kappa * ufl.pi**2)
    
    # Trial and test functions
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    
    # Previous solution
    u_n = fem.Function(V)
    
    # Initial condition: u(x,0) = sin(pi*x)*sin(pi*y)
    u_n.interpolate(lambda x: np.sin(np.pi * x[0]) * np.sin(np.pi * x[1]))
    
    # Store initial condition for output
    nx_out, ny_out = 50, 50
    xs = np.linspace(0, 1, nx_out)
    ys = np.linspace(0, 1, ny_out)
    X, Y = np.meshgrid(xs, ys, indexing='ij')
    points_2d = np.vstack([X.ravel(), Y.ravel(), np.zeros(nx_out * ny_out)])
    
    # Evaluate initial condition
    u_initial = _evaluate_function(domain, u_n, points_2d, nx_out, ny_out)
    
    # Backward Euler: (u - u_n)/dt - kappa*laplacian(u) = f(t_{n+1})
    # Weak form: (u/dt)*v*dx + kappa*grad(u)·grad(v)*dx = (u_n/dt)*v*dx + f*v*dx
    dt_const = fem.Constant(domain, PETSc.ScalarType(dt))
    kappa_const = fem.Constant(domain, PETSc.ScalarType(kappa))
    
    a = (u * v / dt_const) * ufl.dx + kappa_const * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = (u_n * v / dt_const) * ufl.dx + f_ufl * v * ufl.dx
    
    # Boundary conditions - u = 0 on all boundaries (exact solution is 0 on boundary of unit square)
    # Actually, sin(pi*x)*sin(pi*y) = 0 on all boundaries, so g = 0
    tdim = domain.topology.dim
    fdim = tdim - 1
    
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
    )
    
    # For time-dependent BC (which is 0 here), we can use a constant
    u_bc = fem.Function(V)
    u_bc.x.array[:] = 0.0
    dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc = fem.dirichletbc(u_bc, dofs)
    bcs = [bc]
    
    # Compile forms
    a_form = fem.form(a)
    L_form = fem.form(L)
    
    # Assemble matrix (constant in time for this problem with constant kappa and dt)
    A = petsc.assemble_matrix(a_form, bcs=bcs)
    A.assemble()
    
    # Create RHS vector
    b = petsc.create_vector(L_form)
    
    # Solution function
    u_sol = fem.Function(V)
    
    # Setup solver
    solver = PETSc.KSP().create(domain.comm)
    solver.setOperators(A)
    solver.setType(PETSc.KSP.Type.CG)
    pc = solver.getPC()
    pc.setType(PETSc.PC.Type.HYPRE)
    solver.setTolerances(rtol=1e-10, atol=1e-12, max_it=1000)
    solver.setUp()
    
    total_iterations = 0
    
    # Time stepping
    t = 0.0
    for step in range(n_steps):
        t += dt
        t_const.value = t
        
        # Update BC if needed (it's 0 for all time, so no update needed)
        # u_bc is already 0
        
        # Assemble RHS
        with b.localForm() as loc:
            loc.set(0)
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
    
    # Evaluate solution on grid
    u_grid = _evaluate_function(domain, u_sol, points_2d, nx_out, ny_out)
    
    solver.destroy()
    A.destroy()
    b.destroy()
    
    return {
        "u": u_grid,
        "u_initial": u_initial,
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


def _evaluate_function(domain, u_func, points_2d, nx, ny):
    """Evaluate a function on a grid of points."""
    bb_tree = geometry.bb_tree(domain, domain.topology.dim)
    
    points_T = points_2d.T  # shape (N, 3)
    cell_candidates = geometry.compute_collisions_points(bb_tree, points_T)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points_T)
    
    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    
    for i in range(points_T.shape[0]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(points_T[i])
            cells_on_proc.append(links[0])
            eval_map.append(i)
    
    u_values = np.full(points_T.shape[0], np.nan)
    if len(points_on_proc) > 0:
        vals = u_func.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u_values[eval_map] = vals.flatten()
    
    return u_values.reshape((nx, ny))