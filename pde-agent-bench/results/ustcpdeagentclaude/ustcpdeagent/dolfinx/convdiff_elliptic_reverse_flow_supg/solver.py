import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc


def solve(case_spec: dict) -> dict:
    eps_val = 0.02
    beta_val = np.array([-8.0, 4.0])

    grid = case_spec["output"]["grid"]
    nx_out = grid["nx"]
    ny_out = grid["ny"]
    bbox = grid["bbox"]

    # Mesh
    N = 64
    degree = 2
    comm = MPI.COMM_WORLD
    msh = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)

    V = fem.functionspace(msh, ("Lagrange", degree))

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    x = ufl.SpatialCoordinate(msh)
    # Manufactured: u = exp(x)*sin(pi*y)
    u_exact = ufl.exp(x[0]) * ufl.sin(ufl.pi * x[1])
    # -eps*lap(u) + beta . grad(u) = f
    # lap(u) = exp(x)*sin(pi*y) + exp(x)*(-pi^2)*sin(pi*y) = exp(x)*sin(pi*y)*(1 - pi^2)
    # grad(u) = (exp(x)*sin(pi*y), exp(x)*pi*cos(pi*y))
    eps_c = fem.Constant(msh, PETSc.ScalarType(eps_val))
    beta = fem.Constant(msh, PETSc.ScalarType((beta_val[0], beta_val[1])))

    lap_u = ufl.div(ufl.grad(u_exact))
    f_expr = -eps_c * lap_u + ufl.dot(beta, ufl.grad(u_exact))

    # SUPG
    h = ufl.CellDiameter(msh)
    beta_norm = ufl.sqrt(ufl.dot(beta, beta))
    Pe = beta_norm * h / (2.0 * eps_c)
    # tau standard
    tau = (h / (2.0 * beta_norm)) * (1.0 / ufl.tanh(Pe) - 1.0 / Pe)

    # Galerkin
    a_gal = (eps_c * ufl.inner(ufl.grad(u), ufl.grad(v))
             + ufl.inner(ufl.dot(beta, ufl.grad(u)), v)) * ufl.dx
    L_gal = ufl.inner(f_expr, v) * ufl.dx

    # SUPG terms: tau * (beta.grad(v)) * (residual)
    # Residual strong: -eps*lap(u) + beta.grad(u) - f
    r_u = -eps_c * ufl.div(ufl.grad(u)) + ufl.dot(beta, ufl.grad(u))
    a_supg = tau * ufl.inner(r_u, ufl.dot(beta, ufl.grad(v))) * ufl.dx
    L_supg = tau * ufl.inner(f_expr, ufl.dot(beta, ufl.grad(v))) * ufl.dx

    a = a_gal + a_supg
    L = L_gal + L_supg

    # Dirichlet BC
    fdim = msh.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(
        msh, fdim, lambda xx: np.ones(xx.shape[1], dtype=bool)
    )
    dofs_bc = fem.locate_dofs_topological(V, fdim, boundary_facets)

    u_bc = fem.Function(V)
    u_bc.interpolate(fem.Expression(u_exact, V.element.interpolation_points))
    bc = fem.dirichletbc(u_bc, dofs_bc)

    problem = petsc.LinearProblem(
        a, L, bcs=[bc],
        petsc_options={"ksp_type": "gmres", "pc_type": "ilu",
                       "ksp_rtol": 1e-10, "ksp_atol": 1e-12,
                       "ksp_max_it": 2000},
        petsc_options_prefix="cd_"
    )
    u_sol = problem.solve()
    ksp = problem.solver
    iters = ksp.getIterationNumber()

    # Sample on grid
    xs = np.linspace(bbox[0], bbox[1], nx_out)
    ys = np.linspace(bbox[2], bbox[3], ny_out)
    XX, YY = np.meshgrid(xs, ys)
    pts = np.c_[XX.ravel(), YY.ravel(), np.zeros(nx_out * ny_out)]

    tree = geometry.bb_tree(msh, msh.topology.dim)
    cell_candidates = geometry.compute_collisions_points(tree, pts)
    colliding = geometry.compute_colliding_cells(msh, cell_candidates, pts)

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
        u_values[eval_map] = vals.flatten()

    u_grid = u_values.reshape(ny_out, nx_out)

    # Error check
    u_ex_grid = np.exp(XX) * np.sin(np.pi * YY)
    err = np.sqrt(np.mean((u_grid - u_ex_grid) ** 2))
    print(f"RMS error: {err:.3e}, iters: {iters}")

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
    spec = {"output": {"grid": {"nx": 64, "ny": 64, "bbox": [0, 1, 0, 1]}}}
    import time
    t0 = time.time()
    out = solve(spec)
    print(f"Time: {time.time() - t0:.3f}s, shape: {out['u'].shape}")
