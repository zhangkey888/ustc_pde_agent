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
    epsilon = params.get("epsilon", 0.01)
    beta_vec = params.get("beta", [12.0, 6.0])
    
    domain_spec = case_spec.get("domain", {})
    nx_out = domain_spec.get("nx", 50)
    ny_out = domain_spec.get("ny", 50)

    # Mesh resolution - use fine mesh for high Peclet number
    N = 128
    element_degree = 1

    domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
    
    V = fem.functionspace(domain, ("Lagrange", element_degree))

    # Trial and test functions
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    # Spatial coordinates
    x = ufl.SpatialCoordinate(domain)

    # Source term
    pi = ufl.pi
    f_expr = ufl.sin(8 * pi * x[0]) * ufl.sin(6 * pi * x[1]) + \
             0.3 * ufl.sin(12 * pi * x[0]) * ufl.sin(10 * pi * x[1])

    # Convection velocity
    beta = ufl.as_vector([ScalarType(beta_vec[0]), ScalarType(beta_vec[1])])

    # Diffusion constant
    eps_c = fem.Constant(domain, ScalarType(epsilon))

    # Standard Galerkin bilinear form
    a_std = eps_c * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx + \
            ufl.inner(ufl.dot(beta, ufl.grad(u)), v) * ufl.dx
    L_std = f_expr * v * ufl.dx

    # SUPG stabilization
    h = ufl.CellDiameter(domain)
    beta_norm = ufl.sqrt(ufl.dot(beta, beta))
    
    # SUPG stabilization parameter
    Pe_cell = beta_norm * h / (2.0 * eps_c)
    # Classical formula: tau = h / (2 * |beta|) * (coth(Pe) - 1/Pe)
    # Simplified: for high Pe, tau ~ h / (2*|beta|)
    tau = h / (2.0 * beta_norm) * (1.0 - 1.0 / Pe_cell)
    # Clamp tau to be non-negative (use ufl conditional or simpler approach)
    # For high Peclet, Pe_cell >> 1, so tau ~ h/(2*|beta|) which is fine
    # Use a simpler robust formula:
    tau = h * h / (4.0 * eps_c + 2.0 * beta_norm * h)

    # SUPG residual: -eps*laplacian(u) + beta.grad(u) - f
    # For linear elements, laplacian of u_h = 0 within each element
    # So the strong residual applied to trial function is just beta.grad(u)
    # SUPG test function modification: v_supg = tau * beta.grad(v)
    r_supg = ufl.dot(beta, ufl.grad(u)) - f_expr  # strong residual (no diffusion for P1)
    
    a_supg = tau * ufl.inner(ufl.dot(beta, ufl.grad(u)), ufl.dot(beta, ufl.grad(v))) * ufl.dx
    L_supg = tau * ufl.inner(f_expr, ufl.dot(beta, ufl.grad(v))) * ufl.dx

    a_total = a_std + a_supg
    L_total = L_std + L_supg

    # Boundary conditions: u = 0 on all boundaries (g=0 by default)
    tdim = domain.topology.dim
    fdim = tdim - 1

    def boundary_all(x):
        return (np.isclose(x[0], 0.0) | np.isclose(x[0], 1.0) |
                np.isclose(x[1], 0.0) | np.isclose(x[1], 1.0))

    boundary_facets = mesh.locate_entities_boundary(domain, fdim, boundary_all)
    dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc = fem.dirichletbc(ScalarType(0.0), dofs, V)

    # Solve
    ksp_type = "gmres"
    pc_type = "ilu"
    rtol = 1e-8

    problem = petsc.LinearProblem(
        a_total, L_total, bcs=[bc],
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

    # Evaluate on output grid
    nx_eval = nx_out
    ny_eval = ny_out
    xs = np.linspace(0.0, 1.0, nx_eval)
    ys = np.linspace(0.0, 1.0, ny_eval)
    XX, YY = np.meshgrid(xs, ys, indexing='ij')
    points = np.zeros((3, nx_eval * ny_eval))
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

    u_values = np.full(nx_eval * ny_eval, np.nan)
    if len(points_on_proc) > 0:
        pts_arr = np.array(points_on_proc)
        cells_arr = np.array(cells_on_proc, dtype=np.int32)
        vals = u_sol.eval(pts_arr, cells_arr)
        u_values[eval_map] = vals.flatten()

    u_grid = u_values.reshape((nx_eval, ny_eval))

    solver_info = {
        "mesh_resolution": N,
        "element_degree": element_degree,
        "ksp_type": ksp_type,
        "pc_type": pc_type,
        "rtol": rtol,
        "iterations": int(iterations),
    }

    return {
        "u": u_grid,
        "solver_info": solver_info,
    }