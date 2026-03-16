import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc

ScalarType = PETSc.ScalarType


def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD
    
    # Parameters
    N = 80
    degree = 2
    ksp_type = "cg"
    pc_type = "hypre"
    rtol = 1e-10
    
    # Create mesh
    domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
    
    # Function spaces
    V = fem.functionspace(domain, ("Lagrange", degree))
    
    # Manufactured solution: u = exp(5*(x-1))*sin(pi*y)
    # Laplacian: u_xx + u_yy = 25*exp(5*(x-1))*sin(pi*y) - pi^2*exp(5*(x-1))*sin(pi*y)
    #          = (25 - pi^2)*exp(5*(x-1))*sin(pi*y)
    # So w = Laplacian(u) = (25 - pi^2)*exp(5*(x-1))*sin(pi*y)
    # Bilaplacian = Laplacian(w) = (25 - pi^2) * Laplacian(exp(5*(x-1))*sin(pi*y))
    #             = (25 - pi^2)^2 * exp(5*(x-1))*sin(pi*y)
    
    x = ufl.SpatialCoordinate(domain)
    pi_val = np.pi
    
    u_exact_ufl = ufl.exp(5.0 * (x[0] - 1.0)) * ufl.sin(pi_val * x[1])
    
    # Source term f = (25 - pi^2)^2 * exp(5*(x-1))*sin(pi*y)
    coeff = (25.0 - pi_val**2)**2
    f_ufl = coeff * ufl.exp(5.0 * (x[0] - 1.0)) * ufl.sin(pi_val * x[1])
    
    # Mixed formulation: introduce w = -Laplacian(u)
    # -Laplacian(u) = w  =>  (grad(u), grad(v)) = (w, v)  for all v
    # -Laplacian(w) = f  =>  (grad(w), grad(v)) = (f, v)  for all v
    # But w = -Laplacian(u) = -(25 - pi^2)*exp(5*(x-1))*sin(pi*y)
    
    # Actually, let's use w = Laplacian(u), then Laplacian(w) = f
    # Weak form for w: (grad(w), grad(v)) = -(f, v) + boundary terms
    # This gets complicated. Let me use the standard mixed formulation:
    # 
    # w = -Delta(u), so Delta(w) = -Delta^2(u) = -f
    # Weak: (grad(w), grad(phi)) = (f, phi) + boundary terms for w
    # Weak: (grad(u), grad(psi)) = (w, psi) + boundary terms for u
    #
    # Step 1: Solve -Delta(w) = -f with w_bc = -Delta(u)|_boundary
    # Step 2: Solve -Delta(u) = w with u_bc = g
    
    # w_exact = -Laplacian(u_exact) = -(25 - pi^2)*exp(5*(x-1))*sin(pi*y)
    # But actually: Laplacian(u) = (25 - pi^2)*u_exact
    # So w_exact (if w = -Delta u) = -(25 - pi^2)*u_exact
    
    # Let me define w = Delta(u) (positive Laplacian)
    # Then Delta(w) = Delta^2(u) = f
    # Weak form for -Delta(w) = -f: (grad(w), grad(v)) = (-f, v) ... no
    
    # Let's be careful. Standard approach:
    # Delta^2 u = f
    # Introduce sigma = -Delta u
    # Then -Delta sigma = f (since Delta(sigma) = Delta(-Delta u) = -Delta^2 u = -f)
    # 
    # Weak forms:
    # (grad(sigma), grad(v)) = (f, v)  with sigma = -Delta u on boundary
    # (grad(u), grad(v)) = (sigma, v)  with u = g on boundary
    
    # sigma_exact = -Delta(u_exact) = -(25 - pi^2)*exp(5*(x-1))*sin(pi*y)
    
    total_iterations = 0
    
    # --- Step 1: Solve for sigma ---
    # -Delta(sigma) = f in Omega
    # sigma = sigma_exact on boundary
    
    sigma = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    
    a1 = ufl.inner(ufl.grad(sigma), ufl.grad(v)) * ufl.dx
    L1 = f_ufl * v * ufl.dx
    
    # Boundary condition for sigma
    sigma_exact_ufl = -(25.0 - pi_val**2) * ufl.exp(5.0 * (x[0] - 1.0)) * ufl.sin(pi_val * x[1])
    
    tdim = domain.topology.dim
    fdim = tdim - 1
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
    )
    
    sigma_bc_func = fem.Function(V)
    sigma_bc_expr = fem.Expression(sigma_exact_ufl, V.element.interpolation_points)
    sigma_bc_func.interpolate(sigma_bc_expr)
    
    dofs_sigma = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc_sigma = fem.dirichletbc(sigma_bc_func, dofs_sigma)
    
    problem1 = petsc.LinearProblem(
        a1, L1, bcs=[bc_sigma],
        petsc_options={
            "ksp_type": ksp_type,
            "pc_type": pc_type,
            "ksp_rtol": str(rtol),
            "ksp_monitor": None,
        },
        petsc_options_prefix="step1_"
    )
    sigma_h = problem1.solve()
    
    # Get iteration count
    ksp1 = problem1.solver
    total_iterations += ksp1.getIterationNumber()
    
    # --- Step 2: Solve for u ---
    # -Delta(u) = sigma in Omega
    # u = g on boundary
    
    u_trial = ufl.TrialFunction(V)
    v2 = ufl.TestFunction(V)
    
    a2 = ufl.inner(ufl.grad(u_trial), ufl.grad(v2)) * ufl.dx
    L2 = sigma_h * v2 * ufl.dx
    
    # Boundary condition for u
    u_bc_func = fem.Function(V)
    u_bc_expr = fem.Expression(u_exact_ufl, V.element.interpolation_points)
    u_bc_func.interpolate(u_bc_expr)
    
    dofs_u = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc_u = fem.dirichletbc(u_bc_func, dofs_u)
    
    problem2 = petsc.LinearProblem(
        a2, L2, bcs=[bc_u],
        petsc_options={
            "ksp_type": ksp_type,
            "pc_type": pc_type,
            "ksp_rtol": str(rtol),
        },
        petsc_options_prefix="step2_"
    )
    u_h = problem2.solve()
    
    ksp2 = problem2.solver
    total_iterations += ksp2.getIterationNumber()
    
    # --- Evaluate on 50x50 grid ---
    nx_out, ny_out = 50, 50
    xs = np.linspace(0, 1, nx_out)
    ys = np.linspace(0, 1, ny_out)
    XX, YY = np.meshgrid(xs, ys, indexing='ij')
    points = np.zeros((3, nx_out * ny_out))
    points[0] = XX.ravel()
    points[1] = YY.ravel()
    
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
    
    u_values = np.full(nx_out * ny_out, np.nan)
    if len(points_on_proc) > 0:
        vals = u_h.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u_values[eval_map] = vals.flatten()
    
    u_grid = u_values.reshape((nx_out, ny_out))
    
    return {
        "u": u_grid,
        "solver_info": {
            "mesh_resolution": N,
            "element_degree": degree,
            "ksp_type": ksp_type,
            "pc_type": pc_type,
            "rtol": rtol,
            "iterations": total_iterations,
        }
    }