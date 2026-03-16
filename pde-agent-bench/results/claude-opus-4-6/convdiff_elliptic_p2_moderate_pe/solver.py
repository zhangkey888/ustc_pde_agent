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
    epsilon = 0.03
    beta = np.array([5.0, 2.0])
    
    # Mesh and element
    mesh_resolution = 80
    element_degree = 2
    
    domain = mesh.create_unit_square(comm, mesh_resolution, mesh_resolution, cell_type=mesh.CellType.triangle)
    
    V = fem.functionspace(domain, ("Lagrange", element_degree))
    
    x = ufl.SpatialCoordinate(domain)
    
    # Exact solution
    u_exact_ufl = ufl.sin(ufl.pi * x[0]) * ufl.sin(2 * ufl.pi * x[1])
    
    # Compute source term from manufactured solution
    # -eps * laplacian(u) + beta . grad(u) = f
    grad_u_exact = ufl.grad(u_exact_ufl)
    laplacian_u_exact = ufl.div(ufl.grad(u_exact_ufl))
    
    beta_ufl = ufl.as_vector([fem.Constant(domain, ScalarType(beta[0])),
                               fem.Constant(domain, ScalarType(beta[1]))])
    eps = fem.Constant(domain, ScalarType(epsilon))
    
    f_expr = -eps * laplacian_u_exact + ufl.dot(beta_ufl, grad_u_exact)
    
    # Boundary conditions
    u_bc_func = fem.Function(V)
    u_exact_expr = fem.Expression(u_exact_ufl, V.element.interpolation_points)
    u_bc_func.interpolate(u_exact_expr)
    
    tdim = domain.topology.dim
    fdim = tdim - 1
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
    )
    dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc = fem.dirichletbc(u_bc_func, dofs)
    
    # Variational form with SUPG stabilization
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    
    # Standard Galerkin
    a_gal = eps * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx + ufl.dot(beta_ufl, ufl.grad(u)) * v * ufl.dx
    L_gal = f_expr * v * ufl.dx
    
    # SUPG stabilization
    h = ufl.CellDiameter(domain)
    beta_norm = ufl.sqrt(ufl.dot(beta_ufl, beta_ufl))
    
    # Stabilization parameter
    Pe_cell = beta_norm * h / (2.0 * eps)
    # tau = h / (2 * |beta|) * (coth(Pe) - 1/Pe) ≈ h/(2|beta|) for large Pe
    # Use a simpler formula that works well:
    tau = h / (2.0 * beta_norm) * (1.0 - 1.0 / Pe_cell)
    # Clamp tau to be non-negative (for safety)
    # Actually use a standard formula:
    tau = h * h / (4.0 * eps + 2.0 * beta_norm * h)
    
    # SUPG: add tau * (beta . grad(v)) * R to both sides
    # R = -eps * laplacian(u) + beta . grad(u) - f
    # For linear elements, laplacian(u) = 0, but for P2 it's not zero in general
    # However, in the weak form with trial functions, we use the strong-form residual operator
    # applied to the trial function. For the bilinear form:
    # SUPG test function modification: v_supg = v + tau * beta . grad(v)
    
    # For the bilinear form, the SUPG residual of the operator applied to u is:
    # -eps * div(grad(u)) + beta . grad(u)
    # For P2 elements, div(grad(u)) is piecewise constant (not zero)
    
    # SUPG additional terms
    Lop_u = -eps * ufl.div(ufl.grad(u)) + ufl.dot(beta_ufl, ufl.grad(u))
    supg_test = tau * ufl.dot(beta_ufl, ufl.grad(v))
    
    a_supg = ufl.inner(Lop_u, supg_test) * ufl.dx
    L_supg = f_expr * supg_test * ufl.dx
    
    a = a_gal + a_supg
    L = L_gal + L_supg
    
    # Solve
    ksp_type = "gmres"
    pc_type = "ilu"
    rtol = 1e-10
    
    problem = petsc.LinearProblem(
        a, L, bcs=[bc],
        petsc_options={
            "ksp_type": ksp_type,
            "pc_type": pc_type,
            "ksp_rtol": str(rtol),
            "ksp_max_it": "2000",
            "ksp_monitor": None,
        },
        petsc_options_prefix="convdiff_"
    )
    u_sol = problem.solve()
    
    # Get iteration count
    ksp = problem.solver
    iterations = ksp.getIterationNumber()
    
    # Evaluate on 50x50 grid
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
        vals = u_sol.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u_values[eval_map] = vals.flatten()
    
    u_grid = u_values.reshape((nx_out, ny_out))
    
    solver_info = {
        "mesh_resolution": mesh_resolution,
        "element_degree": element_degree,
        "ksp_type": ksp_type,
        "pc_type": pc_type,
        "rtol": rtol,
        "iterations": iterations,
    }
    
    return {"u": u_grid, "solver_info": solver_info}