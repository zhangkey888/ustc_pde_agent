import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, nls, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc


def solve(case_spec: dict) -> dict:
    # 1. Parse case_spec
    pde_config = case_spec.get("pde", {})
    nu_val = pde_config.get("viscosity", 0.14)
    
    # Mesh resolution and element degrees
    N = 80
    degree_u = 2
    degree_p = 1
    
    # 2. Create mesh
    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
    tdim = domain.topology.dim
    fdim = tdim - 1
    
    # 3. Mixed function space (Taylor-Hood P2/P1)
    V = fem.functionspace(domain, ("Lagrange", degree_u, (domain.geometry.dim,)))
    Q = fem.functionspace(domain, ("Lagrange", degree_p))
    
    # Create mixed element
    vel_elem = ufl.VectorElement("Lagrange", domain.ufl_cell(), degree_u)
    pres_elem = ufl.FiniteElement("Lagrange", domain.ufl_cell(), degree_p)
    mixed_elem = ufl.MixedElement([vel_elem, pres_elem])
    W = fem.functionspace(domain, mixed_elem)
    
    # 4. Define exact solution for BCs and source term
    x = ufl.SpatialCoordinate(domain)
    
    # Exact velocity
    u_exact_0 = (-60*(x[1]-0.7)*ufl.exp(-30*((x[0]-0.3)**2 + (x[1]-0.7)**2))
                 + 60*(x[1]-0.3)*ufl.exp(-30*((x[0]-0.7)**2 + (x[1]-0.3)**2)))
    u_exact_1 = (60*(x[0]-0.3)*ufl.exp(-30*((x[0]-0.3)**2 + (x[1]-0.7)**2))
                 - 60*(x[0]-0.7)*ufl.exp(-30*((x[0]-0.7)**2 + (x[1]-0.3)**2)))
    u_exact = ufl.as_vector([u_exact_0, u_exact_1])
    p_exact = ufl.Constant(domain, PETSc.ScalarType(0.0))
    
    nu = fem.Constant(domain, PETSc.ScalarType(nu_val))
    
    # Compute source term f = u·∇u - ν∇²u + ∇p
    # For manufactured solution: f = (u_exact · ∇)u_exact - ν Δu_exact + ∇p_exact
    f_expr = (ufl.grad(u_exact) * u_exact 
              - nu * ufl.div(ufl.grad(u_exact)) 
              + ufl.grad(p_exact))
    
    # 5. Define variational problem
    w = fem.Function(W)
    (u, p) = ufl.split(w)
    (v, q) = ufl.TestFunctions(W)
    
    F = (nu * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
         + ufl.inner(ufl.grad(u) * u, v) * ufl.dx
         - p * ufl.div(v) * ufl.dx
         + q * ufl.div(u) * ufl.dx
         - ufl.inner(f_expr, v) * ufl.dx)
    
    # 6. Boundary conditions
    # Velocity BC on all boundaries
    def all_boundary(x):
        return (np.isclose(x[0], 0.0) | np.isclose(x[0], 1.0) |
                np.isclose(x[1], 0.0) | np.isclose(x[1], 1.0))
    
    boundary_facets = mesh.locate_entities_boundary(domain, fdim, all_boundary)
    
    # Interpolate exact solution for BC
    u_bc_func = fem.Function(V)
    
    # Create expression for exact solution
    u_exact_expr = fem.Expression(u_exact, V.element.interpolation_points)
    u_bc_func.interpolate(u_exact_expr)
    
    dofs_u = fem.locate_dofs_topological((W.sub(0), V), fdim, boundary_facets)
    bc_u = fem.dirichletbc(u_bc_func, dofs_u, W.sub(0))
    
    # Pin pressure at one point to remove nullspace
    # Find a vertex near (0,0)
    def corner_point(x):
        return np.isclose(x[0], 0.0) & np.isclose(x[1], 0.0)
    
    corner_facets = mesh.locate_entities_boundary(domain, fdim, corner_point)
    if len(corner_facets) > 0:
        dofs_p = fem.locate_dofs_topological((W.sub(1), Q), fdim, corner_facets)
        p_bc_func = fem.Function(Q)
        p_bc_func.x.array[:] = 0.0
        bc_p = fem.dirichletbc(p_bc_func, dofs_p, W.sub(1))
        bcs = [bc_u, bc_p]
    else:
        bcs = [bc_u]
    
    # 7. Initial guess: interpolate exact solution (helps Newton converge)
    # Set initial guess from exact solution
    W0_sub, W0_map = W.sub(0).collapse()
    W1_sub, W1_map = W.sub(1).collapse()
    
    u_init = fem.Function(W0_sub)
    u_init.interpolate(u_exact_expr)
    w.x.array[W0_map] = u_init.x.array[:]
    
    # pressure initial guess = 0 (already zero)
    w.x.scatter_forward()
    
    # 8. Solve nonlinear problem
    problem = petsc.NonlinearProblem(F, w, bcs=bcs)
    solver = nls.petsc.NewtonSolver(comm, problem)
    
    solver.convergence_criterion = "incremental"
    solver.rtol = 1e-10
    solver.atol = 1e-12
    solver.max_it = 25
    
    ksp = solver.krylov_solver
    ksp.setType(PETSc.KSP.Type.GMRES)
    ksp.setTolerances(rtol=1e-9, max_it=2000)
    pc = ksp.getPC()
    pc.setType(PETSc.PC.Type.LU)
    pc.setFactorSolverType("mumps")
    
    n_newton, converged = solver.solve(w)
    assert converged, f"Newton solver did not converge after {n_newton} iterations"
    w.x.scatter_forward()
    
    # 9. Extract velocity on evaluation grid
    nx_eval = 50
    ny_eval = 50
    xs = np.linspace(0.0, 1.0, nx_eval)
    ys = np.linspace(0.0, 1.0, ny_eval)
    XX, YY = np.meshgrid(xs, ys, indexing='ij')
    
    points = np.zeros((3, nx_eval * ny_eval))
    points[0, :] = XX.ravel()
    points[1, :] = YY.ravel()
    
    # Get velocity sub-function
    u_sol = w.sub(0).collapse()
    
    # Point evaluation
    bb_tree = geometry.bb_tree(domain, tdim)
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
    
    # Evaluate velocity (2 components)
    vel_magnitude = np.full(nx_eval * ny_eval, np.nan)
    if len(points_on_proc) > 0:
        pts_arr = np.array(points_on_proc)
        cells_arr = np.array(cells_on_proc, dtype=np.int32)
        vals = u_sol.eval(pts_arr, cells_arr)
        # vals shape: (n_points, 2)
        for idx, global_idx in enumerate(eval_map):
            ux = vals[idx, 0]
            uy = vals[idx, 1]
            vel_magnitude[global_idx] = np.sqrt(ux**2 + uy**2)
    
    u_grid = vel_magnitude.reshape((nx_eval, ny_eval))
    
    return {
        "u": u_grid,
        "solver_info": {
            "mesh_resolution": N,
            "element_degree": degree_u,
            "ksp_type": "gmres",
            "pc_type": "lu",
            "rtol": 1e-9,
            "nonlinear_iterations": [int(n_newton)],
        }
    }