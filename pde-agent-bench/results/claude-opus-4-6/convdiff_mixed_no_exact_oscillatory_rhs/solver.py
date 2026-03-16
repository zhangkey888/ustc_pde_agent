import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc

ScalarType = PETSc.ScalarType


def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD

    # Parse parameters
    pde = case_spec.get("pde", {})
    params = pde.get("params", {})
    epsilon = params.get("epsilon", 0.005)
    beta_vec = params.get("beta", [15.0, 7.0])
    
    domain_spec = case_spec.get("domain", {})
    extents = domain_spec.get("extents", [[0, 1], [0, 1]])
    
    # Mesh resolution - use fine mesh for accuracy with high Peclet number
    N = 128
    element_degree = 1
    
    domain = mesh.create_rectangle(
        comm,
        [np.array([extents[0][0], extents[1][0]]),
         np.array([extents[0][1], extents[1][1]])],
        [N, N],
        cell_type=mesh.CellType.triangle
    )
    
    V = fem.functionspace(domain, ("Lagrange", element_degree))
    
    # Trial and test functions
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    
    x = ufl.SpatialCoordinate(domain)
    
    # Convection velocity
    beta = ufl.as_vector([ScalarType(beta_vec[0]), ScalarType(beta_vec[1])])
    
    # Source term
    f_expr = ufl.sin(10 * ufl.pi * x[0]) * ufl.sin(8 * ufl.pi * x[1])
    
    # Bilinear form: diffusion + convection
    a_standard = (epsilon * ufl.inner(ufl.grad(u), ufl.grad(v)) +
                  ufl.inner(ufl.dot(beta, ufl.grad(u)), v)) * ufl.dx
    L_standard = f_expr * v * ufl.dx
    
    # SUPG stabilization
    h = ufl.CellDiameter(domain)
    beta_norm = ufl.sqrt(ufl.dot(beta, beta))
    Pe_cell = beta_norm * h / (2.0 * epsilon)
    
    # SUPG stabilization parameter (optimal formula)
    tau = h / (2.0 * beta_norm) * (ufl.cosh(Pe_cell) / ufl.sinh(Pe_cell) - 1.0 / Pe_cell)
    # Simpler alternative that's more robust:
    # tau = h / (2.0 * beta_norm) * min(1, Pe_cell/3)
    # Use the doubly asymptotic formula:
    tau = h / (2.0 * beta_norm + 1e-16) * (1.0 / ufl.tanh(Pe_cell + 1e-16) - 1.0 / (Pe_cell + 1e-16))
    
    # SUPG residual: strong form residual tested with beta·grad(v)
    # Strong form: -eps*laplacian(u) + beta·grad(u) = f
    # For linear elements, laplacian(u) = 0 on each element
    # So strong residual ≈ beta·grad(u) - f
    # SUPG adds: tau * (beta·grad(u) - f) * (beta·grad(v))
    
    r_supg = ufl.dot(beta, ufl.grad(u)) - f_expr  # strong residual (no laplacian for P1)
    supg_test = tau * ufl.dot(beta, ufl.grad(v))
    
    a_supg = a_standard + r_supg * supg_test * ufl.dx
    # Wait, r_supg contains u (trial), need to split:
    # r_supg = beta·grad(u) - f
    # So: r_supg * supg_test = (beta·grad(u)) * supg_test - f * supg_test
    # The first part goes to bilinear form, second to RHS
    
    # Let me redo properly:
    a_form = (epsilon * ufl.inner(ufl.grad(u), ufl.grad(v)) +
              ufl.dot(beta, ufl.grad(u)) * v +
              tau * ufl.dot(beta, ufl.grad(u)) * ufl.dot(beta, ufl.grad(v))) * ufl.dx
    
    L_form = (f_expr * v +
              tau * f_expr * ufl.dot(beta, ufl.grad(v))) * ufl.dx
    
    # Boundary conditions: g = 0 (default for mixed_type with no exact solution)
    # Check if there's a BC specification
    bcs_spec = pde.get("bcs", [])
    
    tdim = domain.topology.dim
    fdim = tdim - 1
    
    # Apply zero Dirichlet BC on entire boundary
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
    )
    dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc = fem.dirichletbc(ScalarType(0.0), dofs, V)
    bcs = [bc]
    
    # Solve
    ksp_type = "gmres"
    pc_type = "ilu"
    rtol = 1e-10
    
    problem = petsc.LinearProblem(
        a_form, L_form, bcs=bcs,
        petsc_options={
            "ksp_type": ksp_type,
            "pc_type": pc_type,
            "ksp_rtol": str(rtol),
            "ksp_max_it": "5000",
            "ksp_monitor": None,
        },
        petsc_options_prefix="convdiff_"
    )
    u_sol = problem.solve()
    
    # Get iteration count
    ksp = problem.solver
    iterations = ksp.getIterationNumber()
    
    # Sample solution on 50x50 grid
    nx_out, ny_out = 50, 50
    xs = np.linspace(extents[0][0], extents[0][1], nx_out)
    ys = np.linspace(extents[1][0], extents[1][1], ny_out)
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
    
    return {
        "u": u_grid,
        "solver_info": {
            "mesh_resolution": N,
            "element_degree": element_degree,
            "ksp_type": ksp_type,
            "pc_type": pc_type,
            "rtol": rtol,
            "iterations": int(iterations),
        }
    }