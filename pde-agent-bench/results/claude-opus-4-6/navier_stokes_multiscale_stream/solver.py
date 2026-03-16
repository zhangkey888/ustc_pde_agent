import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, nls, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc


def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD
    
    nu_val = 0.12
    N = 40  # mesh resolution
    degree_u = 3
    degree_p = 2
    
    msh = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
    
    # Create mixed function space (Taylor-Hood)
    V_el = ("Lagrange", degree_u, (msh.geometry.dim,))
    Q_el = ("Lagrange", degree_p)
    
    V = fem.functionspace(msh, V_el)
    Q = fem.functionspace(msh, Q_el)
    
    # Mixed element
    mel = ufl.MixedElement([V.ufl_element(), Q.ufl_element()])
    W = fem.functionspace(msh, mel)
    
    # Current solution
    w = fem.Function(W)
    (u, p) = ufl.split(w)
    (v, q) = ufl.TestFunctions(W)
    
    x = ufl.SpatialCoordinate(msh)
    pi = ufl.pi
    
    # Exact solution
    u_exact_0 = pi * ufl.cos(pi * x[1]) * ufl.sin(pi * x[0]) + (3 * pi / 5) * ufl.cos(2 * pi * x[1]) * ufl.sin(3 * pi * x[0])
    u_exact_1 = -pi * ufl.cos(pi * x[0]) * ufl.sin(pi * x[1]) - (9 * pi / 10) * ufl.cos(3 * pi * x[0]) * ufl.sin(2 * pi * x[1])
    u_exact = ufl.as_vector([u_exact_0, u_exact_1])
    p_exact = ufl.cos(2 * pi * x[0]) * ufl.cos(pi * x[1])
    
    nu = fem.Constant(msh, PETSc.ScalarType(nu_val))
    
    # Compute source term from manufactured solution
    # f = u_exact · ∇u_exact - ν ∇²u_exact + ∇p_exact
    f = (ufl.grad(u_exact) * u_exact 
         - nu_val * ufl.div(ufl.grad(u_exact)) 
         + ufl.grad(p_exact))
    
    # Residual form
    F = (
        nu * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
        + ufl.inner(ufl.grad(u) * u, v) * ufl.dx
        - p * ufl.div(v) * ufl.dx
        + q * ufl.div(u) * ufl.dx
        - ufl.inner(f, v) * ufl.dx
    )
    
    # Boundary conditions
    tdim = msh.topology.dim
    fdim = tdim - 1
    
    # All boundary facets
    boundary_facets = mesh.locate_entities_boundary(
        msh, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
    )
    
    # Velocity BC
    u_bc_func = fem.Function(V)
    u_bc_func.interpolate(fem.Expression(
        u_exact, V.element.interpolation_points
    ))
    
    dofs_u = fem.locate_dofs_topological((W.sub(0), V), fdim, boundary_facets)
    bc_u = fem.dirichletbc(u_bc_func, dofs_u, W.sub(0))
    
    # Pin pressure at one point to remove nullspace
    # Find a vertex at (0,0)
    p_bc_dofs = fem.locate_dofs_geometrical(
        (W.sub(1), Q),
        lambda x: np.isclose(x[0], 0.0) & np.isclose(x[1], 0.0)
    )
    p_bc_func = fem.Function(Q)
    # p_exact at (0,0) = cos(0)*cos(0) = 1.0
    p_bc_func.interpolate(fem.Expression(p_exact, Q.element.interpolation_points))
    bc_p = fem.dirichletbc(p_bc_func, p_bc_dofs, W.sub(1))
    
    bcs = [bc_u, bc_p]
    
    # Initial guess: interpolate exact solution (helps convergence)
    # Actually, let's use a Stokes-like initial guess or just zero
    # For manufactured solution with good BC, Newton from zero should work
    w.x.array[:] = 0.0
    
    # Set up nonlinear problem
    problem = petsc.NonlinearProblem(F, w, bcs=bcs)
    solver = nls.petsc.NewtonSolver(comm, problem)
    
    solver.convergence_criterion = "incremental"
    solver.rtol = 1e-10
    solver.atol = 1e-12
    solver.max_it = 50
    
    ksp = solver.krylov_solver
    ksp.setType(PETSc.KSP.Type.GMRES)
    ksp.setTolerances(rtol=1e-10, atol=1e-12, max_it=2000)
    pc = ksp.getPC()
    pc.setType(PETSc.PC.Type.LU)
    pc.setFactorSolverType("mumps")
    
    n_newton, converged = solver.solve(w)
    assert converged, f"Newton solver did not converge after {n_newton} iterations"
    w.x.scatter_forward()
    
    # Extract velocity sub-function
    u_sol = w.sub(0).collapse()
    
    # Evaluate on 50x50 grid
    nx_out, ny_out = 50, 50
    xs = np.linspace(0, 1, nx_out)
    ys = np.linspace(0, 1, ny_out)
    XX, YY = np.meshgrid(xs, ys, indexing='ij')
    
    points = np.zeros((3, nx_out * ny_out))
    points[0, :] = XX.ravel()
    points[1, :] = YY.ravel()
    
    bb_tree = geometry.bb_tree(msh, tdim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, points.T)
    colliding_cells = geometry.compute_colliding_cells(msh, cell_candidates, points.T)
    
    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    for i in range(points.shape[1]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(points[:, i])
            cells_on_proc.append(links[0])
            eval_map.append(i)
    
    vel_mag = np.full(nx_out * ny_out, np.nan)
    if len(points_on_proc) > 0:
        pts_arr = np.array(points_on_proc)
        cells_arr = np.array(cells_on_proc, dtype=np.int32)
        vals = u_sol.eval(pts_arr, cells_arr)
        # vals shape: (n_points, 2)
        mag = np.sqrt(vals[:, 0]**2 + vals[:, 1]**2)
        for idx, global_idx in enumerate(eval_map):
            vel_mag[global_idx] = mag[idx]
    
    u_grid = vel_mag.reshape((nx_out, ny_out))
    
    solver_info = {
        "mesh_resolution": N,
        "element_degree": degree_u,
        "ksp_type": "gmres",
        "pc_type": "lu",
        "rtol": 1e-10,
        "nonlinear_iterations": [int(n_newton)],
    }
    
    return {
        "u": u_grid,
        "solver_info": solver_info,
    }