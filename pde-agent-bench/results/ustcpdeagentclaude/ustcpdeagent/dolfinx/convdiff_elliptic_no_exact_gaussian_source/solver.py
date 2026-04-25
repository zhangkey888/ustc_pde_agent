import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc


def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD

    grid = case_spec["output"]["grid"]
    nx_out = grid["nx"]
    ny_out = grid["ny"]
    bbox = grid["bbox"]

    # Parameters
    eps = 0.05
    beta = np.array([2.0, 1.0])

    N = 300
    degree = 3

    domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", degree))

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    x = ufl.SpatialCoordinate(domain)
    f_expr = ufl.exp(-250.0 * ((x[0] - 0.35)**2 + (x[1] - 0.65)**2))

    eps_c = fem.Constant(domain, PETSc.ScalarType(eps))
    beta_c = fem.Constant(domain, PETSc.ScalarType((beta[0], beta[1])))

    # Galerkin form
    a = eps_c * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx \
        + ufl.inner(ufl.dot(beta_c, ufl.grad(u)), v) * ufl.dx
    L = f_expr * v * ufl.dx

    # SUPG stabilization
    h = ufl.CellDiameter(domain)
    beta_norm = ufl.sqrt(ufl.dot(beta_c, beta_c))
    Pe_elem = beta_norm * h / (2.0 * eps_c)
    # standard tau
    tau = (h / (2.0 * beta_norm)) * (1.0 / ufl.tanh(Pe_elem) - 1.0 / Pe_elem)

    # Residual (strong form): -eps*laplace(u) + beta.grad(u) - f
    residual = -eps_c * ufl.div(ufl.grad(u)) + ufl.dot(beta_c, ufl.grad(u)) - f_expr
    # SUPG test function: tau * beta.grad(v)
    supg_test = tau * ufl.dot(beta_c, ufl.grad(v))

    a_supg = (-eps_c * ufl.div(ufl.grad(u)) + ufl.dot(beta_c, ufl.grad(u))) * supg_test * ufl.dx
    L_supg = f_expr * supg_test * ufl.dx

    a_total = a + a_supg
    L_total = L + L_supg

    # Dirichlet BC: u = 0 on all boundary
    fdim = domain.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
    )
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    u_bc = fem.Function(V)
    u_bc.x.array[:] = 0.0
    bc = fem.dirichletbc(u_bc, boundary_dofs)

    problem = petsc.LinearProblem(
        a_total, L_total, bcs=[bc],
        petsc_options={"ksp_type": "gmres", "pc_type": "hypre", "ksp_rtol": 1e-10},
        petsc_options_prefix="convdiff_"
    )
    u_sol = problem.solve()

    ksp = problem.solver
    iterations = ksp.getIterationNumber()

    # Sample on uniform grid
    xs = np.linspace(bbox[0], bbox[1], nx_out)
    ys = np.linspace(bbox[2], bbox[3], ny_out)
    XX, YY = np.meshgrid(xs, ys)
    pts = np.c_[XX.ravel(), YY.ravel(), np.zeros(nx_out * ny_out)]

    tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(tree, pts)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, pts)

    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    for i in range(pts.shape[0]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(pts[i])
            cells_on_proc.append(links[0])
            eval_map.append(i)

    u_values = np.zeros(pts.shape[0])
    if len(points_on_proc) > 0:
        vals = u_sol.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u_values[eval_map] = vals.flatten()

    u_grid = u_values.reshape(ny_out, nx_out)

    return {
        "u": u_grid,
        "solver_info": {
            "mesh_resolution": N,
            "element_degree": degree,
            "ksp_type": "gmres",
            "pc_type": "hypre",
            "rtol": 1e-10,
            "iterations": iterations,
        }
    }


if __name__ == "__main__":
    import time
    case_spec = {
        "output": {
            "grid": {"nx": 128, "ny": 128, "bbox": [0.0, 1.0, 0.0, 1.0]}
        }
    }
    t0 = time.time()
    result = solve(case_spec)
    t1 = time.time()
    print(f"Time: {t1-t0:.2f}s")
    print(f"u shape: {result['u'].shape}")
    print(f"u min/max: {result['u'].min():.6e}, {result['u'].max():.6e}")
    print(f"solver_info: {result['solver_info']}")
