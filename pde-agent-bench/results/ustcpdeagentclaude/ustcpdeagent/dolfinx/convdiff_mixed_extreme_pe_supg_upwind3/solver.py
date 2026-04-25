import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc


def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD

    # Parameters
    eps = 0.002
    beta_vals = [25.0, 10.0]

    # Mesh
    N = 128
    degree = 2
    domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)

    V = fem.functionspace(domain, ("Lagrange", degree))

    x = ufl.SpatialCoordinate(domain)
    pi = ufl.pi

    # Exact solution and source
    u_exact = ufl.sin(pi * x[0]) * ufl.sin(pi * x[1])
    beta = ufl.as_vector([fem.Constant(domain, PETSc.ScalarType(beta_vals[0])),
                           fem.Constant(domain, PETSc.ScalarType(beta_vals[1]))])
    eps_c = fem.Constant(domain, PETSc.ScalarType(eps))

    # f = -eps*lap(u) + beta . grad(u)
    f = -eps_c * ufl.div(ufl.grad(u_exact)) + ufl.dot(beta, ufl.grad(u_exact))

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    # Galerkin form
    a = eps_c * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx \
        + ufl.inner(ufl.dot(beta, ufl.grad(u)), v) * ufl.dx
    L = ufl.inner(f, v) * ufl.dx

    # SUPG stabilization
    h = ufl.CellDiameter(domain)
    bnorm = ufl.sqrt(ufl.dot(beta, beta) + 1e-14)
    Pe = bnorm * h / (2.0 * eps_c)
    # Use optimal tau
    tau = (h / (2.0 * bnorm)) * (1.0 / ufl.tanh(Pe) - 1.0 / Pe)

    # Strong residual
    r_u = -eps_c * ufl.div(ufl.grad(u)) + ufl.dot(beta, ufl.grad(u))
    r_f = f

    a_supg = tau * ufl.inner(r_u, ufl.dot(beta, ufl.grad(v))) * ufl.dx
    L_supg = tau * ufl.inner(r_f, ufl.dot(beta, ufl.grad(v))) * ufl.dx

    a_total = a + a_supg
    L_total = L + L_supg

    # BC: u = sin(pi*x)*sin(pi*y) = 0 on boundary of unit square
    fdim = domain.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda xx: np.ones(xx.shape[1], dtype=bool)
    )
    dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    u_bc = fem.Function(V)
    u_bc.x.array[:] = 0.0
    bc = fem.dirichletbc(u_bc, dofs)

    problem = petsc.LinearProblem(
        a_total, L_total, bcs=[bc],
        petsc_options={"ksp_type": "gmres", "pc_type": "ilu", "ksp_rtol": 1e-10, "ksp_atol": 1e-14, "ksp_max_it": 2000},
        petsc_options_prefix="cd_"
    )
    u_sol = problem.solve()

    ksp_iters = problem.solver.getIterationNumber()

    # Sample on grid
    gspec = case_spec["output"]["grid"]
    nx = gspec["nx"]
    ny = gspec["ny"]
    bbox = gspec["bbox"]
    xs = np.linspace(bbox[0], bbox[1], nx)
    ys = np.linspace(bbox[2], bbox[3], ny)
    XX, YY = np.meshgrid(xs, ys)
    pts = np.c_[XX.ravel(), YY.ravel(), np.zeros(nx * ny)]

    tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(tree, pts)
    colliding = geometry.compute_colliding_cells(domain, cell_candidates, pts)

    points_on_proc = []
    cells = []
    idx_map = []
    for i in range(pts.shape[0]):
        links = colliding.links(i)
        if len(links) > 0:
            points_on_proc.append(pts[i])
            cells.append(links[0])
            idx_map.append(i)

    u_values = np.zeros(pts.shape[0])
    if points_on_proc:
        vals = u_sol.eval(np.array(points_on_proc), np.array(cells, dtype=np.int32))
        u_values[idx_map] = vals.flatten()

    u_grid = u_values.reshape(ny, nx)

    return {
        "u": u_grid,
        "solver_info": {
            "mesh_resolution": N,
            "element_degree": degree,
            "ksp_type": "gmres",
            "pc_type": "ilu",
            "rtol": 1e-10,
            "iterations": int(ksp_iters),
        }
    }


if __name__ == "__main__":
    import time
    case_spec = {
        "output": {"grid": {"nx": 64, "ny": 64, "bbox": [0.0, 1.0, 0.0, 1.0]}}
    }
    t0 = time.time()
    res = solve(case_spec)
    t1 = time.time()
    u_grid = res["u"]
    # exact
    xs = np.linspace(0, 1, 64)
    ys = np.linspace(0, 1, 64)
    XX, YY = np.meshgrid(xs, ys)
    u_ex = np.sin(np.pi * XX) * np.sin(np.pi * YY)
    err = np.sqrt(np.mean((u_grid - u_ex)**2))
    print(f"Time: {t1-t0:.3f}s, RMS error: {err:.3e}, max err: {np.max(np.abs(u_grid-u_ex)):.3e}")
    print(f"Iterations: {res['solver_info']['iterations']}")
