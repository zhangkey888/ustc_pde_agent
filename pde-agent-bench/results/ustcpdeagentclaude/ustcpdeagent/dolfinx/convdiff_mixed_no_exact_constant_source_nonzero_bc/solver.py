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

    eps_val = 0.005
    beta_val = np.array([12.0, 0.0])
    f_val = 1.0

    # High Peclet => need fine mesh + SUPG
    N = 480
    degree = 2

    domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", degree))

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    beta = fem.Constant(domain, PETSc.ScalarType(beta_val))
    eps_c = fem.Constant(domain, PETSc.ScalarType(eps_val))
    f = fem.Constant(domain, PETSc.ScalarType(f_val))

    # Standard Galerkin
    a = eps_c * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx \
        + ufl.inner(ufl.dot(beta, ufl.grad(u)), v) * ufl.dx
    L = f * v * ufl.dx

    # SUPG stabilization
    h = ufl.CellDiameter(domain)
    beta_norm = ufl.sqrt(ufl.dot(beta, beta) + 1e-14)
    Pe_h = beta_norm * h / (2.0 * eps_c)
    # tau = (h/(2|b|)) * (coth(Pe) - 1/Pe), approximated
    tau = (h / (2.0 * beta_norm)) * (1.0 / ufl.tanh(Pe_h) - 1.0 / Pe_h)

    # Residual for linear problem: -eps*lap(u) + b.grad(u) - f
    if degree == 1:
        # Laplacian of P1 is zero
        r_u = ufl.dot(beta, ufl.grad(u))
        r_rhs = f
    else:
        r_u = -eps_c * ufl.div(ufl.grad(u)) + ufl.dot(beta, ufl.grad(u))
        r_rhs = f

    a_supg = tau * ufl.inner(r_u, ufl.dot(beta, ufl.grad(v))) * ufl.dx
    L_supg = tau * ufl.inner(r_rhs, ufl.dot(beta, ufl.grad(v))) * ufl.dx

    a += a_supg
    L += L_supg

    # Dirichlet BC: u = sin(pi*x) on all boundary
    fdim = domain.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
    )
    u_bc = fem.Function(V)
    u_bc.interpolate(lambda x: np.sin(np.pi * x[0]))
    dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc = fem.dirichletbc(u_bc, dofs)

    petsc_options = {
        "ksp_type": "gmres",
        "pc_type": "hypre",
        "pc_hypre_type": "boomeramg",
        "ksp_rtol": 1e-10,
        "ksp_atol": 1e-14,
        "ksp_max_it": 2000,
    }
    try:
        problem = petsc.LinearProblem(
            a, L, bcs=[bc],
            petsc_options=petsc_options,
            petsc_options_prefix="cd_"
        )
        u_sol = problem.solve()
        iters = problem.solver.getIterationNumber()
        ksp_type = "gmres"
        pc_type = "hypre"
    except Exception:
        problem = petsc.LinearProblem(
            a, L, bcs=[bc],
            petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
            petsc_options_prefix="cd_lu_"
        )
        u_sol = problem.solve()
        iters = 1
        ksp_type = "preonly"
        pc_type = "lu"

    # Sample on grid
    xs = np.linspace(bbox[0], bbox[1], nx_out)
    ys = np.linspace(bbox[2], bbox[3], ny_out)
    XX, YY = np.meshgrid(xs, ys)
    pts = np.c_[XX.ravel(), YY.ravel(), np.zeros(nx_out * ny_out)]

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

    u_vals = np.full(nx_out * ny_out, np.nan)
    if len(points_on_proc) > 0:
        vals = u_sol.eval(np.array(points_on_proc),
                           np.array(cells_on_proc, dtype=np.int32))
        u_vals[eval_map] = vals.flatten()

    u_grid = u_vals.reshape(ny_out, nx_out)

    return {
        "u": u_grid,
        "solver_info": {
            "mesh_resolution": N,
            "element_degree": degree,
            "ksp_type": ksp_type,
            "pc_type": pc_type,
            "rtol": 1e-10,
            "iterations": int(iters),
        },
    }


if __name__ == "__main__":
    import time
    case_spec = {
        "output": {
            "grid": {
                "nx": 64, "ny": 64,
                "bbox": [0.0, 1.0, 0.0, 1.0],
            }
        }
    }
    t0 = time.time()
    result = solve(case_spec)
    t1 = time.time()
    print(f"Time: {t1-t0:.2f}s")
    print(f"u shape: {result['u'].shape}")
    print(f"u min/max: {np.nanmin(result['u']):.4f} / {np.nanmax(result['u']):.4f}")
    print(f"solver_info: {result['solver_info']}")
