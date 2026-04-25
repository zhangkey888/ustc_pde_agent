import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc as fem_petsc
import ufl
from petsc4py import PETSc


def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD

    eps = 0.01
    beta_vec = np.array([14.0, 6.0])

    # Grid for output
    grid = case_spec["output"]["grid"]
    nx_out = grid["nx"]
    ny_out = grid["ny"]
    bbox = grid["bbox"]

    # Mesh resolution - quadrilateral
    N = 96
    degree = 2

    domain = mesh.create_rectangle(
        comm,
        [np.array([bbox[0], bbox[2]]), np.array([bbox[1], bbox[3]])],
        [N, N],
        cell_type=mesh.CellType.quadrilateral,
    )

    V = fem.functionspace(domain, ("Lagrange", degree))

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    x = ufl.SpatialCoordinate(domain)
    pi = ufl.pi

    # Exact solution
    u_exact = ufl.sin(pi * x[0]) * ufl.sin(pi * x[1])

    beta = fem.Constant(domain, PETSc.ScalarType((14.0, 6.0)))
    epsilon = fem.Constant(domain, PETSc.ScalarType(eps))

    # f = -eps * laplace(u) + beta . grad(u)
    # laplace(sin(pi x)sin(pi y)) = -2 pi^2 sin(pi x) sin(pi y)
    # grad u = (pi cos(pi x) sin(pi y), pi sin(pi x) cos(pi y))
    f_expr = (eps * 2 * pi**2 * ufl.sin(pi * x[0]) * ufl.sin(pi * x[1])
              + beta_vec[0] * pi * ufl.cos(pi * x[0]) * ufl.sin(pi * x[1])
              + beta_vec[1] * pi * ufl.sin(pi * x[0]) * ufl.cos(pi * x[1]))

    # SUPG stabilization
    h = ufl.CellDiameter(domain)
    beta_norm = ufl.sqrt(ufl.dot(beta, beta))
    Pe_h = beta_norm * h / (2.0 * epsilon)
    # tau (optimal for convection-dominated)
    tau = (h / (2.0 * beta_norm)) * (1.0 / ufl.tanh(Pe_h) - 1.0 / Pe_h)

    # Galerkin
    a_gal = (epsilon * ufl.dot(ufl.grad(u), ufl.grad(v))
             + ufl.dot(beta, ufl.grad(u)) * v) * ufl.dx
    L_gal = f_expr * v * ufl.dx

    # SUPG: add tau*(beta.grad v)*(residual of strong form)
    # Strong residual: -eps*laplace(u) + beta.grad(u) - f
    # For P2 on quads, laplace is non-zero
    residual_u = -epsilon * ufl.div(ufl.grad(u)) + ufl.dot(beta, ufl.grad(u))
    a_supg = tau * ufl.dot(beta, ufl.grad(v)) * residual_u * ufl.dx
    L_supg = tau * ufl.dot(beta, ufl.grad(v)) * f_expr * ufl.dx

    a = a_gal + a_supg
    L = L_gal + L_supg

    # Boundary condition: u = exact on boundary
    fdim = domain.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
    )
    u_bc = fem.Function(V)
    u_bc.interpolate(fem.Expression(u_exact, V.element.interpolation_points))
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc = fem.dirichletbc(u_bc, boundary_dofs)

    problem = fem_petsc.LinearProblem(
        a, L, bcs=[bc],
        petsc_options={"ksp_type": "gmres", "pc_type": "ilu", "ksp_rtol": 1e-8},
        petsc_options_prefix="cd_supg_"
    )
    u_sol = problem.solve()

    ksp_its = problem.solver.getIterationNumber()

    # Sample onto output grid
    xs = np.linspace(bbox[0], bbox[1], nx_out)
    ys = np.linspace(bbox[2], bbox[3], ny_out)
    XX, YY = np.meshgrid(xs, ys)
    pts = np.c_[XX.ravel(), YY.ravel(), np.zeros(nx_out * ny_out)]

    tree = geometry.bb_tree(domain, domain.topology.dim)
    cand = geometry.compute_collisions_points(tree, pts)
    colliding = geometry.compute_colliding_cells(domain, cand, pts)

    cells = []
    pts_on_proc = []
    idx_map = []
    for i in range(pts.shape[0]):
        links = colliding.links(i)
        if len(links) > 0:
            cells.append(links[0])
            pts_on_proc.append(pts[i])
            idx_map.append(i)

    u_values = np.zeros(pts.shape[0])
    if len(pts_on_proc) > 0:
        vals = u_sol.eval(np.array(pts_on_proc), np.array(cells, dtype=np.int32))
        u_values[idx_map] = vals.flatten()

    u_grid = u_values.reshape(ny_out, nx_out)

    return {
        "u": u_grid,
        "solver_info": {
            "mesh_resolution": N,
            "element_degree": degree,
            "ksp_type": "gmres",
            "pc_type": "ilu",
            "rtol": 1e-8,
            "iterations": int(ksp_its),
        }
    }
