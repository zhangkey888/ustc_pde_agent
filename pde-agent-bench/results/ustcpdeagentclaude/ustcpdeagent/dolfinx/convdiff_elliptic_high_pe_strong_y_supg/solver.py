import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc


def solve(case_spec: dict) -> dict:
    # Parameters
    eps_val = 0.01
    beta_vec = np.array([0.0, 15.0])

    # Output grid
    grid = case_spec["output"]["grid"]
    nx_out = grid["nx"]
    ny_out = grid["ny"]
    bbox = grid["bbox"]

    # Mesh
    N = 96
    degree = 2
    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)

    V = fem.functionspace(domain, ("Lagrange", degree))

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    x = ufl.SpatialCoordinate(domain)
    # Exact solution
    u_exact = ufl.sin(ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])
    # f = -eps*Laplacian(u) + beta . grad(u)
    lap_u = -2 * ufl.pi**2 * ufl.sin(ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])
    grad_u = ufl.as_vector([
        ufl.pi * ufl.cos(ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1]),
        ufl.pi * ufl.sin(ufl.pi * x[0]) * ufl.cos(ufl.pi * x[1]),
    ])
    beta = fem.Constant(domain, PETSc.ScalarType((beta_vec[0], beta_vec[1])))
    eps_c = fem.Constant(domain, PETSc.ScalarType(eps_val))

    f_expr = -eps_c * lap_u + ufl.dot(beta, grad_u)

    # Standard Galerkin
    a = (eps_c * ufl.dot(ufl.grad(u), ufl.grad(v))
         + ufl.dot(beta, ufl.grad(u)) * v) * ufl.dx
    L = f_expr * v * ufl.dx

    # SUPG stabilization
    h = ufl.CellDiameter(domain)
    beta_norm = ufl.sqrt(ufl.dot(beta, beta) + 1e-14)
    Pe_h = beta_norm * h / (2.0 * eps_c)
    # tau formula for SUPG
    # tau = (h / (2|b|)) * (coth(Pe_h) - 1/Pe_h), approximated
    # For high Pe, tau ~ h/(2|b|)
    tau = h / (2.0 * beta_norm) * (1.0 / ufl.tanh(Pe_h) - 1.0 / Pe_h)

    # Residual
    # Strong residual: -eps*div(grad(u)) + beta . grad(u) - f
    res_u = -eps_c * ufl.div(ufl.grad(u)) + ufl.dot(beta, ufl.grad(u))
    res_f = f_expr

    a_supg = tau * ufl.dot(beta, ufl.grad(v)) * res_u * ufl.dx
    L_supg = tau * ufl.dot(beta, ufl.grad(v)) * res_f * ufl.dx

    a_total = a + a_supg
    L_total = L + L_supg

    # Dirichlet BC from exact solution
    u_bc = fem.Function(V)
    u_bc.interpolate(fem.Expression(u_exact, V.element.interpolation_points))

    fdim = domain.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
    )
    bdofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc = fem.dirichletbc(u_bc, bdofs)

    problem = petsc.LinearProblem(
        a_total, L_total, bcs=[bc],
        petsc_options={"ksp_type": "gmres", "pc_type": "ilu", "ksp_rtol": 1e-10},
        petsc_options_prefix="cd_supg_"
    )
    u_sol = problem.solve()
    its = problem.solver.getIterationNumber()

    # Sample on grid
    xs = np.linspace(bbox[0], bbox[1], nx_out)
    ys = np.linspace(bbox[2], bbox[3], ny_out)
    XX, YY = np.meshgrid(xs, ys)
    pts = np.c_[XX.ravel(), YY.ravel(), np.zeros(nx_out * ny_out)]

    tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(tree, pts)
    colliding = geometry.compute_colliding_cells(domain, cell_candidates, pts)

    cells = []
    points_on_proc = []
    eval_map = []
    for i in range(pts.shape[0]):
        links = colliding.links(i)
        if len(links) > 0:
            points_on_proc.append(pts[i])
            cells.append(links[0])
            eval_map.append(i)

    u_vals = np.zeros(pts.shape[0])
    if points_on_proc:
        vals = u_sol.eval(np.array(points_on_proc), np.array(cells, dtype=np.int32))
        u_vals[eval_map] = vals.flatten()
    u_grid = u_vals.reshape(ny_out, nx_out)

    return {
        "u": u_grid,
        "solver_info": {
            "mesh_resolution": N,
            "element_degree": degree,
            "ksp_type": "gmres",
            "pc_type": "ilu",
            "rtol": 1e-10,
            "iterations": int(its),
        },
    }


if __name__ == "__main__":
    import time
    case_spec = {
        "output": {"grid": {"nx": 128, "ny": 128, "bbox": [0, 1, 0, 1]}}
    }
    t0 = time.time()
    result = solve(case_spec)
    t1 = time.time()
    print(f"Time: {t1-t0:.3f}s")
    u = result["u"]
    xs = np.linspace(0, 1, 128)
    ys = np.linspace(0, 1, 128)
    XX, YY = np.meshgrid(xs, ys)
    u_exact = np.sin(np.pi * XX) * np.sin(np.pi * YY)
    err = np.sqrt(np.mean((u - u_exact)**2))
    max_err = np.max(np.abs(u - u_exact))
    print(f"RMS error: {err:.3e}")
    print(f"Max error: {max_err:.3e}")
    print(f"Iterations: {result['solver_info']['iterations']}")
