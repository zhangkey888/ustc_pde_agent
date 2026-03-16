import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc

ScalarType = PETSc.ScalarType


def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD

    # Parse case spec
    pde = case_spec.get("pde", {})
    time_params = pde.get("time", {})
    t_end = time_params.get("t_end", 0.06)
    dt_suggested = time_params.get("dt", 0.003)
    scheme = time_params.get("scheme", "backward_euler")

    kappa = 1.0
    
    # Choose mesh resolution and element degree
    # The manufactured solution has sin(4*pi*x)*sin(4*pi*y) which oscillates 4 times
    # Need sufficient resolution. With P2 elements, N=40 should be fine.
    N = 40
    degree = 2
    dt = dt_suggested

    # Create mesh
    domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
    
    # Function space
    V = fem.functionspace(domain, ("Lagrange", degree))
    
    # Time
    t = fem.Constant(domain, ScalarType(0.0))
    dt_const = fem.Constant(domain, ScalarType(dt))
    kappa_const = fem.Constant(domain, ScalarType(kappa))
    
    # Spatial coordinates
    x = ufl.SpatialCoordinate(domain)
    pi = np.pi
    
    # Exact solution: u = exp(-t)*sin(4*pi*x)*sin(4*pi*y)
    u_exact_ufl = ufl.exp(-t) * ufl.sin(4 * pi * x[0]) * ufl.sin(4 * pi * x[1])
    
    # Source term: f = du/dt - kappa * laplacian(u)
    # du/dt = -exp(-t)*sin(4*pi*x)*sin(4*pi*y)
    # laplacian(u) = exp(-t)*(-16*pi^2 - 16*pi^2)*sin(4*pi*x)*sin(4*pi*y) = -32*pi^2*exp(-t)*sin(4*pi*x)*sin(4*pi*y)
    # -kappa*laplacian(u) = 32*pi^2*kappa*exp(-t)*sin(4*pi*x)*sin(4*pi*y)
    # f = du/dt - kappa*laplacian(u) = exp(-t)*sin(4*pi*x)*sin(4*pi*y)*(-1 + 32*pi^2*kappa)
    f_ufl = ufl.exp(-t) * ufl.sin(4 * pi * x[0]) * ufl.sin(4 * pi * x[1]) * (-1.0 + 32.0 * pi**2 * kappa)
    
    # Trial and test functions
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    
    # Previous solution
    u_n = fem.Function(V)
    
    # Initial condition: u(x,0) = sin(4*pi*x)*sin(4*pi*y)
    u_n.interpolate(lambda x: np.sin(4 * np.pi * x[0]) * np.sin(4 * np.pi * x[1]))
    
    # Store initial condition for output
    # Create grid for evaluation
    nx_out, ny_out = 50, 50
    xs = np.linspace(0, 1, nx_out)
    ys = np.linspace(0, 1, ny_out)
    XX, YY = np.meshgrid(xs, ys, indexing='ij')
    points_2d = np.vstack([XX.ravel(), YY.ravel(), np.zeros(nx_out * ny_out)])
    
    # Evaluate initial condition on grid
    bb_tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, points_2d.T)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points_2d.T)
    
    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    for i in range(points_2d.shape[1]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(points_2d.T[i])
            cells_on_proc.append(links[0])
            eval_map.append(i)
    
    u_initial_flat = np.full(nx_out * ny_out, np.nan)
    if len(points_on_proc) > 0:
        vals = u_n.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u_initial_flat[eval_map] = vals.flatten()
    u_initial = u_initial_flat.reshape((nx_out, ny_out))
    
    # Backward Euler: (u - u_n)/dt - kappa*laplacian(u) = f
    # Weak form: (u/dt)*v*dx + kappa*grad(u)*grad(v)*dx = (u_n/dt)*v*dx + f*v*dx
    a = (u / dt_const) * v * ufl.dx + kappa_const * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = (u_n / dt_const) * v * ufl.dx + f_ufl * v * ufl.dx
    
    # Boundary conditions (time-dependent)
    # On unit square boundary, exact solution = exp(-t)*sin(4*pi*x)*sin(4*pi*y)
    # On boundaries x=0,1 or y=0,1: sin(4*pi*0)=0, sin(4*pi*1)=0 => BC = 0
    # So homogeneous Dirichlet BCs
    tdim = domain.topology.dim
    fdim = tdim - 1
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
    )
    dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc = fem.dirichletbc(ScalarType(0.0), dofs, V)
    bcs = [bc]
    
    # Compile forms
    a_form = fem.form(a)
    L_form = fem.form(L)
    
    # Assemble matrix (constant in time for backward Euler with constant kappa and dt)
    A = petsc.assemble_matrix(a_form, bcs=bcs)
    A.assemble()
    
    # Solution function
    u_sol = fem.Function(V)
    
    # Create RHS vector
    b = fem.petsc.create_vector(L_form)
    
    # Setup solver
    solver = PETSc.KSP().create(domain.comm)
    solver.setOperators(A)
    solver.setType(PETSc.KSP.Type.CG)
    pc = solver.getPC()
    pc.setType(PETSc.PC.Type.HYPRE)
    solver.setTolerances(rtol=1e-10, atol=1e-12, max_it=1000)
    solver.setUp()
    
    # Time stepping
    n_steps = int(np.round(t_end / dt))
    total_iterations = 0
    
    for step in range(n_steps):
        t.value = (step + 1) * dt
        
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
    
    # Evaluate final solution on grid
    u_final_flat = np.full(nx_out * ny_out, np.nan)
    if len(points_on_proc) > 0:
        vals = u_sol.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u_final_flat[eval_map] = vals.flatten()
    u_grid = u_final_flat.reshape((nx_out, ny_out))
    
    # Cleanup
    solver.destroy()
    A.destroy()
    b.destroy()
    
    return {
        "u": u_grid,
        "u_initial": u_initial,
        "solver_info": {
            "mesh_resolution": N,
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