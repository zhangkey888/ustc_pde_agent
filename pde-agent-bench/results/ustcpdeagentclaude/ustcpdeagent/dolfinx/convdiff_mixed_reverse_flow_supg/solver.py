import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc


def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD
    eps_val = 0.005
    beta_val = np.array([-20.0, 5.0])

    N = 128
    degree = 2
    domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)

    V = fem.functionspace(domain, ("Lagrange", degree))
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    x = ufl.SpatialCoordinate(domain)
    # Manufactured solution
    u_exact = ufl.exp(x[0]) * ufl.sin(ufl.pi * x[1])
    grad_u_exact = ufl.as_vector([ufl.exp(x[0]) * ufl.sin(ufl.pi * x[1]),
                                   ufl.pi * ufl.exp(x[0]) * ufl.cos(ufl.pi * x[1])])
    lap_u_exact = ufl.exp(x[0]) * ufl.sin(ufl.pi * x[1]) - (ufl.pi**2) * ufl.exp(x[0]) * ufl.sin(ufl.pi * x[1])

    eps_c = fem.Constant(domain, PETSc.ScalarType(eps_val))
    beta = fem.Constant(domain, PETSc.ScalarType(beta_val))

    f = -eps_c * lap_u_exact + ufl.dot(beta, grad_u_exact)

    # Standard Galerkin
    a = eps_c * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx \
        + ufl.inner(ufl.dot(beta, ufl.grad(u)), v) * ufl.dx
    L = f * v * ufl.dx

    # SUPG stabilization
    h = ufl.CellDiameter(domain)
    beta_norm = ufl.sqrt(ufl.dot(beta, beta))
    Pe_h = beta_norm * h / (2.0 * eps_c)
    # tau for SUPG
    tau = (h / (2.0 * beta_norm)) * (1.0 / ufl.tanh(Pe_h) - 1.0 / Pe_h)

    # Residual-based SUPG
    r_u = -eps_c * ufl.div(ufl.grad(u)) + ufl.dot(beta, ufl.grad(u))
    a += tau * ufl.inner(r_u, ufl.dot(beta, ufl.grad(v))) * ufl.dx
    L += tau * ufl.inner(f, ufl.dot(beta, ufl.grad(v))) * ufl.dx

    # BCs
    fdim = domain.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
    )
    u_bc = fem.Function(V)
    u_bc.interpolate(fem.Expression(u_exact, V.element.interpolation_points))
    dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc = fem.dirichletbc(u_bc, dofs)

    problem = petsc.LinearProblem(
        a, L, bcs=[bc],
        petsc_options={"ksp_type": "gmres", "pc_type": "ilu", "ksp_rtol": 1e-10, "ksp_atol": 1e-12},
        petsc_options_prefix="cd_"
    )
    u_sol = problem.solve()

    ksp = problem.solver
    iters = ksp.getIterationNumber()

    # Sample on grid
    grid = case_spec["output"]["grid"]
    nx = grid["nx"]; ny = grid["ny"]
    bbox = grid["bbox"]
    xs = np.linspace(bbox[0], bbox[1], nx)
    ys = np.linspace(bbox[2], bbox[3], ny)
    XX, YY = np.meshgrid(xs, ys)
    pts = np.c_[XX.ravel(), YY.ravel(), np.zeros(nx * ny)]

    tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(tree, pts)
    colliding = geometry.compute_colliding_cells(domain, cell_candidates, pts)

    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    for i in range(pts.shape[0]):
        links = colliding.links(i)
        if len(links) > 0:
            points_on_proc.append(pts[i])
            cells_on_proc.append(links[0])
            eval_map.append(i)

    u_values = np.zeros(pts.shape[0])
    if len(points_on_proc) > 0:
        vals = u_sol.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        for k, idx in enumerate(eval_map):
            u_values[idx] = vals[k, 0] if vals.ndim > 1 else vals[k]

    u_grid = u_values.reshape(ny, nx)

    return {
        "u": u_grid,
        "solver_info": {
            "mesh_resolution": N,
            "element_degree": degree,
            "ksp_type": "gmres",
            "pc_type": "ilu",
            "rtol": 1e-10,
            "iterations": int(iters),
        }
    }


if __name__ == "__main__":
    import time
    case_spec = {
        "output": {"grid": {"nx": 100, "ny": 100, "bbox": [0.0, 1.0, 0.0, 1.0]}}
    }
    t0 = time.time()
    res = solve(case_spec)
    t1 = time.time()
    u_grid = res["u"]
    # Compare with exact
    nx, ny = 100, 100
    xs = np.linspace(0, 1, nx)
    ys = np.linspace(0, 1, ny)
    XX, YY = np.meshgrid(xs, ys)
    u_ex = np.exp(XX) * np.sin(np.pi * YY)
    err = np.sqrt(np.mean((u_grid - u_ex) ** 2))
    max_err = np.max(np.abs(u_grid - u_ex))
    print(f"Time: {t1-t0:.3f}s, RMSE: {err:.3e}, Max err: {max_err:.3e}")
    print(f"Iters: {res['solver_info']['iterations']}")
