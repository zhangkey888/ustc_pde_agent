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
    epsilon = 0.2
    beta = [0.8, 0.3]
    
    # Mesh and element settings
    N = 64
    degree = 2

    domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", degree))

    x = ufl.SpatialCoordinate(domain)
    
    # Exact solution
    u_exact = ufl.cos(ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])
    
    # Compute source term from manufactured solution
    # -eps * laplacian(u) + beta . grad(u) = f
    # u = cos(pi*x)*sin(pi*y)
    # grad(u) = (-pi*sin(pi*x)*sin(pi*y), pi*cos(pi*x)*cos(pi*y))
    # laplacian(u) = -pi^2*cos(pi*x)*sin(pi*y) - pi^2*cos(pi*x)*sin(pi*y) = -2*pi^2*cos(pi*x)*sin(pi*y)
    # -eps * (-2*pi^2*cos(pi*x)*sin(pi*y)) + beta_x*(-pi*sin(pi*x)*sin(pi*y)) + beta_y*(pi*cos(pi*x)*cos(pi*y))
    # = 2*eps*pi^2*cos(pi*x)*sin(pi*y) - 0.8*pi*sin(pi*x)*sin(pi*y) + 0.3*pi*cos(pi*x)*cos(pi*y)
    
    eps_c = fem.Constant(domain, ScalarType(epsilon))
    beta_vec = fem.Constant(domain, np.array(beta, dtype=ScalarType))
    
    f_expr = (2.0 * epsilon * ufl.pi**2 * ufl.cos(ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])
              - 0.8 * ufl.pi * ufl.sin(ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])
              + 0.3 * ufl.pi * ufl.cos(ufl.pi * x[0]) * ufl.cos(ufl.pi * x[1]))

    # Trial and test functions
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    # Standard Galerkin
    a = eps_c * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx + ufl.inner(ufl.dot(beta_vec, ufl.grad(u)), v) * ufl.dx
    L = f_expr * v * ufl.dx

    # SUPG stabilization
    h = ufl.CellDiameter(domain)
    beta_norm = ufl.sqrt(ufl.dot(beta_vec, beta_vec))
    Pe_cell = beta_norm * h / (2.0 * eps_c)
    tau = h / (2.0 * beta_norm) * (1.0 / ufl.tanh(Pe_cell) - 1.0 / Pe_cell)
    
    # SUPG residual: strong form applied to trial function approximation
    # R(u) = -eps*laplacian(u) + beta.grad(u) - f
    # For linear elements, laplacian(u) = 0 within elements, but for degree 2 it's not zero
    # We use the test function modification: v_supg = tau * beta . grad(v)
    r_supg = ufl.dot(beta_vec, ufl.grad(v))
    
    a_supg = a + tau * ufl.inner(ufl.dot(beta_vec, ufl.grad(u)), r_supg) * ufl.dx
    # For degree >= 2, include diffusion part of residual in SUPG
    # -eps * div(grad(u)) term: for P2 elements this is nonzero
    a_supg += tau * (-eps_c) * ufl.inner(ufl.div(ufl.grad(u)), r_supg) * ufl.dx
    L_supg = L + tau * ufl.inner(f_expr, r_supg) * ufl.dx

    # Boundary conditions
    tdim = domain.topology.dim
    fdim = tdim - 1

    def boundary_all(x):
        return (np.isclose(x[0], 0.0) | np.isclose(x[0], 1.0) |
                np.isclose(x[1], 0.0) | np.isclose(x[1], 1.0))

    boundary_facets = mesh.locate_entities_boundary(domain, fdim, boundary_all)
    dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)

    u_bc = fem.Function(V)
    u_bc.interpolate(lambda x: np.cos(np.pi * x[0]) * np.sin(np.pi * x[1]))
    bc = fem.dirichletbc(u_bc, dofs)

    # Solve
    ksp_type = "gmres"
    pc_type = "ilu"
    rtol = 1e-10

    problem = petsc.LinearProblem(
        a_supg, L_supg, bcs=[bc],
        petsc_options={
            "ksp_type": ksp_type,
            "pc_type": pc_type,
            "ksp_rtol": str(rtol),
            "ksp_max_it": "1000",
            "ksp_monitor": None,
        },
        petsc_options_prefix="cdsolve_"
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
        "mesh_resolution": N,
        "element_degree": degree,
        "ksp_type": ksp_type,
        "pc_type": pc_type,
        "rtol": rtol,
        "iterations": iterations,
    }

    return {"u": u_grid, "solver_info": solver_info}