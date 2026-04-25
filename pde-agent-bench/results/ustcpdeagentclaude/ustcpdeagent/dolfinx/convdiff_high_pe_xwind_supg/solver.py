import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc


def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD

    # Parameters
    eps = 0.01
    beta_vec = np.array([20.0, 0.0])

    # Output grid
    grid = case_spec["output"]["grid"]
    nx_out = grid["nx"]
    ny_out = grid["ny"]
    bbox = grid["bbox"]

    # Mesh
    N = 96
    degree = 2
    domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)

    V = fem.functionspace(domain, ("Lagrange", degree))

    # Manufactured solution
    x = ufl.SpatialCoordinate(domain)
    u_exact = ufl.sin(ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])

    # f = -eps * laplacian(u) + beta . grad(u)
    # laplacian(sin(pi x)sin(pi y)) = -2 pi^2 sin(pi x) sin(pi y)
    # grad(u) = [pi cos(pi x) sin(pi y), pi sin(pi x) cos(pi y)]
    # beta.grad(u) = 20 * pi * cos(pi x) sin(pi y)
    f_expr = (eps * 2.0 * ufl.pi**2 * ufl.sin(ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])
              + beta_vec[0] * ufl.pi * ufl.cos(ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])
              + beta_vec[1] * ufl.pi * ufl.sin(ufl.pi * x[0]) * ufl.cos(ufl.pi * x[1]))

    beta = fem.Constant(domain, PETSc.ScalarType(beta_vec))
    eps_c = fem.Constant(domain, PETSc.ScalarType(eps))

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    # Galerkin
    a_gal = eps_c * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx \
        + ufl.inner(ufl.dot(beta, ufl.grad(u)), v) * ufl.dx
    L_gal = f_expr * v * ufl.dx

    # SUPG
    h = ufl.CellDiameter(domain)
    beta_norm = ufl.sqrt(ufl.dot(beta, beta) + 1e-14)
    # Standard SUPG tau
    Pe = beta_norm * h / (2.0 * eps_c)
    # xi = coth(Pe) - 1/Pe, approximate for high Pe as 1 - 1/Pe
    # Use the classic formula
    tau = (h / (2.0 * beta_norm)) * (1.0 / ufl.tanh(Pe) - 1.0 / Pe)

    # Residual (for trial u, adding -eps*lap(u) requires P2 or higher; for P1 it's zero from trial)
    # For P2 we can include the full residual
    R_u = -eps_c * ufl.div(ufl.grad(u)) + ufl.dot(beta, ufl.grad(u))
    R_f = f_expr

    a_supg = tau * ufl.inner(R_u, ufl.dot(beta, ufl.grad(v))) * ufl.dx
    L_supg = tau * ufl.inner(R_f, ufl.dot(beta, ufl.grad(v))) * ufl.dx

    a = a_gal + a_supg
    L = L_gal + L_supg

    # BCs from exact solution
    fdim = domain.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda xx: np.ones(xx.shape[1], dtype=bool)
    )
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    u_bc = fem.Function(V)
    u_bc.interpolate(fem.Expression(u_exact, V.element.interpolation_points))
    bc = fem.dirichletbc(u_bc, boundary_dofs)

    problem = petsc.LinearProblem(
        a, L, bcs=[bc],
        petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
        petsc_options_prefix="convdiff_"
    )
    u_sol = problem.solve()

    ksp = problem.solver
    iters = ksp.getIterationNumber()

    # Sample on grid
    xs = np.linspace(bbox[0], bbox[1], nx_out)
    ys = np.linspace(bbox[2], bbox[3], ny_out)
    XX, YY = np.meshgrid(xs, ys)
    pts = np.c_[XX.ravel(), YY.ravel(), np.zeros(nx_out * ny_out)]

    tree = geometry.bb_tree(domain, domain.topology.dim)
    cand = geometry.compute_collisions_points(tree, pts)
    coll = geometry.compute_colliding_cells(domain, cand, pts)

    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    for i in range(pts.shape[0]):
        links = coll.links(i)
        if len(links) > 0:
            points_on_proc.append(pts[i])
            cells_on_proc.append(links[0])
            eval_map.append(i)

    u_values = np.full(pts.shape[0], np.nan)
    if len(points_on_proc) > 0:
        vals = u_sol.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u_values[eval_map] = vals.flatten()

    u_grid = u_values.reshape(ny_out, nx_out)

    # Verification
    u_ex_grid = np.sin(np.pi * XX) * np.sin(np.pi * YY)
    err = np.sqrt(np.mean((u_grid - u_ex_grid) ** 2))
    print(f"RMS error vs exact: {err:.3e}")

    return {
        "u": u_grid,
        "solver_info": {
            "mesh_resolution": N,
            "element_degree": degree,
            "ksp_type": "preonly",
            "pc_type": "lu",
            "rtol": 1e-12,
            "iterations": int(iters),
        }
    }


if __name__ == "__main__":
    import time
    case_spec = {
        "output": {
            "grid": {"nx": 64, "ny": 64, "bbox": [0.0, 1.0, 0.0, 1.0]}
        }
    }
    t0 = time.time()
    res = solve(case_spec)
    t1 = time.time()
    print(f"Time: {t1-t0:.3f}s, shape: {res['u'].shape}")
