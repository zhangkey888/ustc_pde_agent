import numpy as np
from dolfinx import mesh, fem, default_scalar_type, geometry
from dolfinx.fem import petsc
from mpi4py import MPI
import ufl
from petsc4py import PETSc


def solve(case_spec: dict) -> dict:
    # 1. Parse case_spec
    pde = case_spec.get("pde", {})
    
    kappa = 0.8
    t_end = 0.2
    dt_suggested = 0.02
    scheme = "backward_euler"
    
    # Try to parse from case_spec
    if "coefficients" in pde:
        kappa = float(pde["coefficients"].get("kappa", kappa))
    if "time" in pde:
        t_end = float(pde["time"].get("t_end", t_end))
        dt_suggested = float(pde["time"].get("dt", dt_suggested))
        scheme = pde["time"].get("scheme", scheme)
    
    # 2. Mesh and function space
    nx, ny = 80, 80
    degree = 1
    dt = 0.005  # Use smaller dt for accuracy
    n_steps = int(round(t_end / dt))
    dt = t_end / n_steps  # Adjust to hit t_end exactly
    
    domain = mesh.create_unit_square(MPI.COMM_WORLD, nx, ny, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", degree))
    
    # 3. Define source term and initial condition using UFL
    x = ufl.SpatialCoordinate(domain)
    pi = np.pi
    
    f_expr = ufl.cos(2 * pi * x[0]) * ufl.sin(pi * x[1])
    u0_expr = ufl.sin(2 * pi * x[0]) * ufl.sin(pi * x[1])
    
    # 4. Initial condition
    u_n = fem.Function(V, name="u_n")
    u0_fem_expr = fem.Expression(u0_expr, V.element.interpolation_points)
    u_n.interpolate(u0_fem_expr)
    
    # Store initial condition for output
    u_initial_func = fem.Function(V)
    u_initial_func.interpolate(u0_fem_expr)
    
    # 5. Boundary conditions: u = g on boundary
    # For this problem, the boundary condition should be consistent.
    # Since sin(2*pi*x)*sin(pi*y) = 0 on the boundary of [0,1]^2,
    # and the exact solution evolves from that, we use homogeneous Dirichlet BCs
    # (the source term cos(2*pi*x)*sin(pi*y) is zero at y=0 and y=1 but not at x=0,x=1)
    # Actually, let's think: the problem says u = g on boundary. 
    # Since there's no exact solution given, we need to figure out g.
    # For a general heat equation without exact solution, the BC is typically
    # specified. Let's use g = 0 (homogeneous Dirichlet) since the initial condition
    # vanishes on the boundary.
    
    tdim = domain.topology.dim
    fdim = tdim - 1
    
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
    )
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc = fem.dirichletbc(PETSc.ScalarType(0.0), boundary_dofs, V)
    bcs = [bc]
    
    # 6. Variational form (Backward Euler)
    # (u - u_n)/dt - kappa * laplacian(u) = f
    # Weak form: (u, v)/dt + kappa*(grad(u), grad(v)) = (u_n, v)/dt + (f, v)
    
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    
    kappa_c = fem.Constant(domain, PETSc.ScalarType(kappa))
    dt_c = fem.Constant(domain, PETSc.ScalarType(dt))
    
    a = (u * v / dt_c + kappa_c * ufl.inner(ufl.grad(u), ufl.grad(v))) * ufl.dx
    L = (u_n * v / dt_c + f_expr * v) * ufl.dx
    
    # 7. Assemble and set up solver for time stepping
    a_form = fem.form(a)
    L_form = fem.form(L)
    
    A = petsc.assemble_matrix(a_form, bcs=bcs)
    A.assemble()
    
    b = fem.Function(V)
    b_vec = b.x.petsc_vec
    
    u_sol = fem.Function(V, name="u")
    
    solver = PETSc.KSP().create(domain.comm)
    solver.setOperators(A)
    solver.setType(PETSc.KSP.Type.CG)
    pc = solver.getPC()
    pc.setType(PETSc.PC.Type.HYPRE)
    solver.setTolerances(rtol=1e-10, atol=1e-12, max_it=1000)
    solver.setUp()
    
    # 8. Time stepping
    total_iterations = 0
    t = 0.0
    
    for step in range(n_steps):
        t += dt
        
        # Assemble RHS
        with b_vec.localForm() as loc:
            loc.set(0.0)
        petsc.assemble_vector(b_vec, L_form)
        petsc.apply_lifting(b_vec, [a_form], bcs=[bcs])
        b_vec.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        petsc.set_bc(b_vec, bcs)
        
        # Solve
        solver.solve(b_vec, u_sol.x.petsc_vec)
        u_sol.x.scatter_forward()
        total_iterations += solver.getIterationNumber()
        
        # Update u_n
        u_n.x.array[:] = u_sol.x.array[:]
        u_n.x.scatter_forward()
    
    # 9. Extract solution on 50x50 grid
    nx_out, ny_out = 50, 50
    xs = np.linspace(0.0, 1.0, nx_out)
    ys = np.linspace(0.0, 1.0, ny_out)
    XX, YY = np.meshgrid(xs, ys, indexing='ij')
    points_2d = np.zeros((3, nx_out * ny_out))
    points_2d[0, :] = XX.ravel()
    points_2d[1, :] = YY.ravel()
    
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
    
    u_values = np.full(nx_out * ny_out, np.nan)
    if len(points_on_proc) > 0:
        pts_arr = np.array(points_on_proc)
        cells_arr = np.array(cells_on_proc, dtype=np.int32)
        vals = u_sol.eval(pts_arr, cells_arr)
        u_values[eval_map] = vals.flatten()
    
    u_grid = u_values.reshape((nx_out, ny_out))
    
    # Also extract initial condition on same grid
    u0_values = np.full(nx_out * ny_out, np.nan)
    if len(points_on_proc) > 0:
        vals0 = u_initial_func.eval(pts_arr, cells_arr)
        u0_values[eval_map] = vals0.flatten()
    u0_grid = u0_values.reshape((nx_out, ny_out))
    
    # Cleanup
    solver.destroy()
    A.destroy()
    
    return {
        "u": u_grid,
        "u_initial": u0_grid,
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