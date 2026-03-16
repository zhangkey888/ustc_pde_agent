import numpy as np
from dolfinx import mesh, fem, nls, geometry
from dolfinx.fem import petsc
from mpi4py import MPI
import ufl
from petsc4py import PETSc


def solve(case_spec: dict) -> dict:
    # 1. Parse case_spec
    pde_config = case_spec.get("pde", {})
    nu_val = pde_config.get("viscosity", 0.01)
    
    # Mesh resolution - use high resolution for accuracy with low viscosity
    N = 80
    degree_u = 2
    degree_p = 1
    
    comm = MPI.COMM_WORLD
    
    # 2. Create mesh
    domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
    tdim = domain.topology.dim
    fdim = tdim - 1
    
    # 3. Mixed function spaces (Taylor-Hood P2/P1)
    V = fem.functionspace(domain, ("Lagrange", degree_u, (domain.geometry.dim,)))
    Q = fem.functionspace(domain, ("Lagrange", degree_p))
    
    # Create mixed element
    vel_elem = ufl.VectorElement("Lagrange", domain.ufl_cell(), degree_u)
    pres_elem = ufl.FiniteElement("Lagrange", domain.ufl_cell(), degree_p)
    mixed_elem = ufl.MixedElement([vel_elem, pres_elem])
    W = fem.functionspace(domain, mixed_elem)
    
    # 4. Define exact solution for BCs and source term
    x = ufl.SpatialCoordinate(domain)
    pi = ufl.pi
    
    # Exact velocity: u = [0.2*pi*cos(pi*y)*sin(2*pi*x), -0.4*pi*cos(2*pi*x)*sin(pi*y)]
    u_exact_0 = 0.2 * pi * ufl.cos(pi * x[1]) * ufl.sin(2.0 * pi * x[0])
    u_exact_1 = -0.4 * pi * ufl.cos(2.0 * pi * x[0]) * ufl.sin(pi * x[1])
    u_exact = ufl.as_vector([u_exact_0, u_exact_1])
    
    # Exact pressure: p = 0
    p_exact = fem.Constant(domain, PETSc.ScalarType(0.0))
    
    nu = fem.Constant(domain, PETSc.ScalarType(nu_val))
    
    # Compute source term from manufactured solution
    # f = u·∇u - ν ∇²u + ∇p
    # Since p=0, ∇p = 0
    # f = (u·∇)u - ν Δu
    # In weak form: (u·∇u, v) + ν(∇u, ∇v) - (p, ∇·v) + (∇·u, q) = (f, v)
    # f = (u_exact · ∇)u_exact - ν Δu_exact
    
    grad_u_exact = ufl.grad(u_exact)
    f_expr = ufl.grad(u_exact) * u_exact - nu * ufl.div(ufl.grad(u_exact))
    
    # 5. Define variational problem
    w = fem.Function(W)
    (u_test, p_test) = ufl.TestFunctions(W)
    (u_sol, p_sol) = ufl.split(w)
    
    # Residual: ν(∇u, ∇v) + ((u·∇)u, v) - (p, ∇·v) + (q, ∇·u) - (f, v) = 0
    F = (
        nu * ufl.inner(ufl.grad(u_sol), ufl.grad(u_test)) * ufl.dx
        + ufl.inner(ufl.grad(u_sol) * u_sol, u_test) * ufl.dx
        - p_sol * ufl.div(u_test) * ufl.dx
        + ufl.div(u_sol) * p_test * ufl.dx
        - ufl.inner(f_expr, u_test) * ufl.dx
    )
    
    # 6. Boundary conditions - apply exact solution on all boundaries
    # Velocity BC
    u_bc_func = fem.Function(V)
    
    # Interpolate exact solution
    u_exact_expr = fem.Expression(
        u_exact,
        V.element.interpolation_points
    )
    u_bc_func.interpolate(u_exact_expr)
    
    # Find all boundary facets
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
    )
    
    # Velocity DOFs
    dofs_u = fem.locate_dofs_topological((W.sub(0), V), fdim, boundary_facets)
    bc_u = fem.dirichletbc(u_bc_func, dofs_u, W.sub(0))
    
    # Pin pressure at one point to remove nullspace
    # Find a vertex at (0,0)
    def corner(x):
        return np.isclose(x[0], 0.0) & np.isclose(x[1], 0.0)
    
    dofs_p = fem.locate_dofs_geometrical((W.sub(1), Q), corner)
    p_bc_func = fem.Function(Q)
    p_bc_func.x.array[:] = 0.0
    bc_p = fem.dirichletbc(p_bc_func, dofs_p, W.sub(1))
    
    bcs = [bc_u, bc_p]
    
    # 7. Initial guess - use Stokes solution first for robustness
    # Set initial guess to zero (or interpolate exact for faster convergence)
    # For better convergence, interpolate the exact solution as initial guess
    W0_sub, W0_map = W.sub(0).collapse()
    W1_sub, W1_map = W.sub(1).collapse()
    
    u_init = fem.Function(W0_sub)
    u_init.interpolate(u_exact_expr)
    w.x.array[W0_map] = u_init.x.array[:]
    w.x.array[W1_map] = 0.0
    w.x.scatter_forward()
    
    # 8. Solve nonlinear problem
    problem = petsc.NonlinearProblem(F, w, bcs=bcs)
    solver = nls.petsc.NewtonSolver(comm, problem)
    
    solver.convergence_criterion = "incremental"
    solver.rtol = 1e-10
    solver.atol = 1e-12
    solver.max_it = 50
    solver.report = True
    
    ksp = solver.krylov_solver
    ksp.setType(PETSc.KSP.Type.GMRES)
    ksp.setTolerances(rtol=1e-10, atol=1e-12, max_it=2000)
    pc = ksp.getPC()
    pc.setType(PETSc.PC.Type.LU)
    pc.setFactorSolverType("mumps")
    
    n_newton, converged = solver.solve(w)
    assert converged, f"Newton solver did not converge after {n_newton} iterations"
    w.x.scatter_forward()
    
    # 9. Extract velocity on 50x50 grid
    nx_out, ny_out = 50, 50
    xs = np.linspace(0.0, 1.0, nx_out)
    ys = np.linspace(0.0, 1.0, ny_out)
    XX, YY = np.meshgrid(xs, ys, indexing='ij')
    
    points_2d = np.zeros((3, nx_out * ny_out))
    points_2d[0, :] = XX.ravel()
    points_2d[1, :] = YY.ravel()
    
    # Get velocity sub-function
    u_h = w.sub(0).collapse()
    
    # Point evaluation
    bb_tree = geometry.bb_tree(domain, tdim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, points_2d.T)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points_2d.T)
    
    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    
    for i in range(nx_out * ny_out):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(points_2d[:, i])
            cells_on_proc.append(links[0])
            eval_map.append(i)
    
    u_values = np.full((nx_out * ny_out, 2), np.nan)
    if len(points_on_proc) > 0:
        pts_arr = np.array(points_on_proc)
        cells_arr = np.array(cells_on_proc, dtype=np.int32)
        vals = u_h.eval(pts_arr, cells_arr)
        for idx, global_idx in enumerate(eval_map):
            u_values[global_idx, :] = vals[idx, :]
    
    # Compute velocity magnitude
    vel_mag = np.sqrt(u_values[:, 0]**2 + u_values[:, 1]**2)
    u_grid = vel_mag.reshape((nx_out, ny_out))
    
    # Get total linear iterations
    total_linear_its = 0
    try:
        total_linear_its = int(ksp.getIterationNumber()) * n_newton
    except:
        total_linear_its = n_newton  # LU is 1 iteration per Newton step
    
    return {
        "u": u_grid,
        "solver_info": {
            "mesh_resolution": N,
            "element_degree": degree_u,
            "ksp_type": "gmres",
            "pc_type": "lu",
            "rtol": 1e-10,
            "iterations": n_newton,  # For direct solver, each Newton step = 1 linear solve
            "nonlinear_iterations": [int(n_newton)],
        }
    }