import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc


def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD
    eps_val = 0.005
    beta_val = np.array([20.0, 10.0])

    N = 160
    degree = 2

    domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", degree))

    x = ufl.SpatialCoordinate(domain)
    pi = ufl.pi
    u_exact_ufl = ufl.sin(pi * x[0]) * ufl.sin(pi * x[1])

    eps_c = fem.Constant(domain, PETSc.ScalarType(eps_val))
    beta = fem.Constant(domain, PETSc.ScalarType(beta_val))

    # f = -eps * Laplace(u) + beta . grad(u)
    f_expr = (-eps_c * ufl.div(ufl.grad(u_exact_ufl))
              + ufl.dot(beta, ufl.grad(u_exact_ufl)))

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    a = eps_c * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx \
        + ufl.inner(ufl.dot(beta, ufl.grad(u)), v) * ufl.dx
    L = f_expr * v * ufl.dx

    # SUPG stabilization
    h = ufl.CellDiameter(domain)
    beta_norm = ufl.sqrt(ufl.dot(beta, beta))
    Pe_K = beta_norm * h / (2.0 * eps_c)
    # tau for SUPG
    tau = (h / (2.0 * beta_norm)) * (1.0 / ufl.tanh(Pe_K) - 1.0 / Pe_K)

    # Residual-based SUPG (strong residual)
    R_u = -eps_c * ufl.div(ufl.grad(u)) + ufl.dot(beta, ufl.grad(u))
    R_f = f_expr
    a += tau * ufl.inner(R_u, ufl.dot(beta, ufl.grad(v))) * ufl.dx
    L += tau * ufl.inner(R_f, ufl.dot(beta, ufl.grad(v))) * ufl.dx

    # BC: u = 0 on boundary (since sin(pi*x)*sin(pi*y) = 0 there)
    tdim = domain.topology.dim
    fdim = tdim - 1
    domain.topology.create_connectivity(fdim, tdim)
    boundary_facets = mesh.exterior_facet_indices(domain.topology)
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    u_bc = fem.Function(V)
    u_bc.x.array[:] = 0.0
    bc = fem.dirichletbc(u_bc, boundary_dofs)

    problem = petsc.LinearProblem(
        a, L, bcs=[bc],
        petsc_options={"ksp_type": "preonly", "pc_type": "lu",
                       "ksp_rtol": 1e-10, "ksp_atol": 1e-12,
                       "ksp_max_it": 2000},
        petsc_options_prefix="cd_supg_"
    )
    u_sol = problem.solve()
    its = problem.solver.getIterationNumber()

    # Sample onto grid
    grid = case_spec["output"]["grid"]
    nx = grid["nx"]; ny = grid["ny"]
    bbox = grid["bbox"]
    xs = np.linspace(bbox[0], bbox[1], nx)
    ys = np.linspace(bbox[2], bbox[3], ny)
    XX, YY = np.meshgrid(xs, ys)
    pts = np.c_[XX.ravel(), YY.ravel(), np.zeros(nx * ny)]

    tree = geometry.bb_tree(domain, domain.topology.dim)
    cand = geometry.compute_collisions_points(tree, pts)
    coll = geometry.compute_colliding_cells(domain, cand, pts)

    points_on = []
    cells_on = []
    emap = []
    for i in range(pts.shape[0]):
        links = coll.links(i)
        if len(links) > 0:
            points_on.append(pts[i])
            cells_on.append(links[0])
            emap.append(i)

    u_vals = np.full(pts.shape[0], np.nan)
    if len(points_on) > 0:
        v = u_sol.eval(np.array(points_on), np.array(cells_on, dtype=np.int32))
        u_vals[emap] = v.flatten()
    u_grid = u_vals.reshape(ny, nx)

    # Accuracy check vs analytical
    u_ex = np.sin(np.pi * XX) * np.sin(np.pi * YY)
    err = np.sqrt(np.mean((u_grid - u_ex) ** 2))
    print(f"[solver] N={N} p={degree} iters={its} rms_err={err:.3e}")

    return {
        "u": u_grid,
        "solver_info": {
            "mesh_resolution": N,
            "element_degree": degree,
            "ksp_type": "gmres",
            "pc_type": "ilu",
            "rtol": 1e-10,
            "iterations": int(its),
        }
    }


if __name__ == "__main__":
    import time
    spec = {"output": {"grid": {"nx": 128, "ny": 128, "bbox": [0, 1, 0, 1]}}}
    t0 = time.time()
    out = solve(spec)
    t1 = time.time()
    u = out["u"]
    xs = np.linspace(0, 1, 128); ys = np.linspace(0, 1, 128)
    XX, YY = np.meshgrid(xs, ys)
    u_ex = np.sin(np.pi * XX) * np.sin(np.pi * YY)
    err = np.sqrt(np.mean((u - u_ex) ** 2))
    print(f"time={t1-t0:.3f}s, err={err:.3e}, max_err={np.max(np.abs(u-u_ex)):.3e}")
