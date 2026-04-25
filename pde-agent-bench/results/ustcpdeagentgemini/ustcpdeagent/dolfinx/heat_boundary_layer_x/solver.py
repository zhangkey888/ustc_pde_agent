import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
import ufl
from dolfinx import mesh, fem, geometry

def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD
    
    # Mesh and function space
    nx = 64
    ny = 64
    domain = mesh.create_unit_square(comm, nx, ny, cell_type=mesh.CellType.triangle)
    degree = 2
    V = fem.functionspace(domain, ("Lagrange", degree))
    
    # Time parameters
    t = 0.0
    t_end = 0.08
    dt = 0.002
    n_steps = int(np.round((t_end - t) / dt))
    
    # Define exact solution expression
    x = ufl.SpatialCoordinate(domain)
    t_const = fem.Constant(domain, PETSc.ScalarType(t))
    u_ex = ufl.exp(-t_const) * ufl.exp(5 * x[0]) * ufl.sin(ufl.pi * x[1])
    
    # Source term f = du/dt - div(kappa grad(u))
    # kappa = 1.0
    # f = -u_ex - div(grad(u_ex)) = -u_ex - (25*u_ex - pi^2*u_ex) = -u_ex * (1 + 25 - pi^2)
    f = -u_ex * (26.0 - ufl.pi**2)
    
    # Boundary conditions
    fdim = domain.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(domain, fdim, lambda x_coord: np.ones(x_coord.shape[1], dtype=bool))
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    
    u_bc = fem.Function(V)
    u_bc_expr = fem.Expression(u_ex, V.element.interpolation_points())
    u_bc.interpolate(u_bc_expr)
    bc = fem.dirichletbc(u_bc, boundary_dofs)
    
    # Initial condition
    u_n = fem.Function(V)
    u_n.interpolate(u_bc_expr)
    
    # Store initial for output
    u_initial_copy = fem.Function(V)
    u_initial_copy.x.array[:] = u_n.x.array[:]
    
    # Variational problem (Backward Euler)
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    dt_const = fem.Constant(domain, PETSc.ScalarType(dt))
    
    F = ufl.inner(u - u_n, v) * ufl.dx + dt_const * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx - dt_const * ufl.inner(f, v) * ufl.dx
    a = ufl.lhs(F)
    L = ufl.rhs(F)
    
    # Assemble linear system
    a_form = fem.form(a)
    L_form = fem.form(L)
    
    A = fem.petsc.assemble_matrix(a_form, bcs=[bc])
    A.assemble()
    b = fem.petsc.create_vector(L_form)
    
    # Solver setup
    ksp = PETSc.KSP().create(comm)
    ksp.setOperators(A)
    ksp.setType("preonly")
    ksp.getPC().setType("lu")
    ksp.getPC().setFactorSolverType("mumps")
    
    u_sol = fem.Function(V)
    
    total_iterations = 0
    # Time loop
    for n in range(n_steps):
        t += dt
        t_const.value = t
        
        # Update BCs
        u_bc.interpolate(u_bc_expr)
        
        # Assemble RHS
        with b.localForm() as loc_b:
            loc_b.set(0)
        fem.petsc.assemble_vector(b, L_form)
        fem.petsc.apply_lifting(b, [a_form], bcs=[[bc]])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        fem.petsc.set_bc(b, [bc])
        
        # Solve
        ksp.solve(b, u_sol.x.petsc_vec)
        u_sol.x.scatter_forward()
        
        # Update previous solution
        u_n.x.array[:] = u_sol.x.array[:]
        total_iterations += ksp.getIterationNumber()
    
    # Output interpolation
    out_grid = case_spec["output"]["grid"]
    nx_out = out_grid["nx"]
    ny_out = out_grid["ny"]
    bbox = out_grid["bbox"]
    
    xs = np.linspace(bbox[0], bbox[1], nx_out)
    ys = np.linspace(bbox[2], bbox[3], ny_out)
    X, Y = np.meshgrid(xs, ys)
    points = np.vstack((X.flatten(), Y.flatten(), np.zeros_like(X.flatten())))
    
    bb_tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, points.T)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points.T)
    
    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    for i in range(points.shape[1]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(points.T[i])
            cells_on_proc.append(links[0])
            eval_map.append(i)
    
    u_grid = np.full((points.shape[1],), np.nan)
    u_init_grid = np.full((points.shape[1],), np.nan)
    if len(points_on_proc) > 0:
        pts_arr = np.array(points_on_proc)
        cells_arr = np.array(cells_on_proc, dtype=np.int32)
        vals = u_sol.eval(pts_arr, cells_arr).flatten()
        vals_init = u_initial_copy.eval(pts_arr, cells_arr).flatten()
        u_grid[eval_map] = vals
        u_init_grid[eval_map] = vals_init
    
    u_grid = u_grid.reshape((ny_out, nx_out))
    u_init_grid = u_init_grid.reshape((ny_out, nx_out))
    
    solver_info = {
        "mesh_resolution": nx,
        "element_degree": degree,
        "ksp_type": "preonly",
        "pc_type": "lu",
        "rtol": 1e-8,
        "iterations": total_iterations,
        "dt": dt,
        "n_steps": n_steps,
        "time_scheme": "backward_euler"
    }
    
    return {
        "u": u_grid,
        "u_initial": u_init_grid,
        "solver_info": solver_info
    }

