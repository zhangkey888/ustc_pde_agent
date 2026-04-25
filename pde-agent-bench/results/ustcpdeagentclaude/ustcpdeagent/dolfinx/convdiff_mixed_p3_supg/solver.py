import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc


def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD

    # Parameters
    eps_val = 0.01
    beta_val = np.array([12.0, 4.0])

    # Grid for output
    grid = case_spec["output"]["grid"]
    nx_out = grid["nx"]
    ny_out = grid["ny"]
    bbox = grid["bbox"]

    # Mesh and element choice
    N = 160
    degree = 4

    msh = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(msh, ("Lagrange", degree))

    x = ufl.SpatialCoordinate(msh)
    pi = ufl.pi
    u_exact = ufl.sin(pi * x[0]) * ufl.sin(2 * pi * x[1])

    # f = -eps * laplacian(u) + beta . grad(u)
    grad_u = ufl.grad(u_exact)
    lap_u = ufl.div(grad_u)
    eps_c = fem.Constant(msh, PETSc.ScalarType(eps_val))
    beta_c = fem.Constant(msh, PETSc.ScalarType(beta_val))
    f_expr = -eps_c * lap_u + ufl.dot(beta_c, grad_u)

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    # SUPG stabilization
    h = ufl.CellDiameter(msh)
    beta_norm = ufl.sqrt(ufl.dot(beta_c, beta_c) + 1e-14)
    Pe_loc = beta_norm * h / (2.0 * eps_c)
    # tau from Codina/standard formula
    tau = h / (2.0 * beta_norm) * (1.0 / ufl.tanh(Pe_loc) - 1.0 / Pe_loc)

    # Galerkin
    a_gal = eps_c * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx + ufl.inner(ufl.dot(beta_c, ufl.grad(u)), v) * ufl.dx
    L_gal = ufl.inner(f_expr, v) * ufl.dx

    # SUPG: residual-based
    # Residual: -eps*lap(u) + beta.grad(u) - f
    res_u = -eps_c * ufl.div(ufl.grad(u)) + ufl.dot(beta_c, ufl.grad(u)) - f_expr
    test_supg = ufl.dot(beta_c, ufl.grad(v))
    a_supg = tau * res_u * test_supg * ufl.dx
    # The above combines both trial-dependent and f-dependent pieces, need to split for bilinear/linear
    # Let's split manually
    a_supg_bilinear = tau * (-eps_c * ufl.div(ufl.grad(u)) + ufl.dot(beta_c, ufl.grad(u))) * test_supg * ufl.dx
    L_supg = tau * f_expr * test_supg * ufl.dx

    a_form = a_gal + a_supg_bilinear
    L_form = L_gal + L_supg

    # BC
    fdim = msh.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(msh, fdim, lambda X: np.ones(X.shape[1], dtype=bool))
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    u_bc = fem.Function(V)
    u_bc.interpolate(fem.Expression(u_exact, V.element.interpolation_points))
    bc = fem.dirichletbc(u_bc, boundary_dofs)

    problem = petsc.LinearProblem(
        a_form, L_form, bcs=[bc],
        petsc_options={"ksp_type": "preonly", "pc_type": "lu", "pc_factor_mat_solver_type": "mumps"},
        petsc_options_prefix="cd_",
    )
    u_sol = problem.solve()

    iters = problem.solver.getIterationNumber()

    # L2 error
    err_form = fem.form((u_sol - u_exact) ** 2 * ufl.dx)
    L2_err = np.sqrt(comm.allreduce(fem.assemble_scalar(err_form), op=MPI.SUM))

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

    u_grid = np.full(pts.shape[0], np.nan)
    if len(points_on_proc) > 0:
        vals = u_sol.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u_grid[eval_map] = vals.flatten()
    u_grid = u_grid.reshape(ny_out, nx_out)

    return {
        "u": u_grid,
        "solver_info": {
            "mesh_resolution": N,
            "element_degree": degree,
            "ksp_type": "preonly",
            "pc_type": "lu",
            "rtol": 1e-12,
            "iterations": int(iters),
            "L2_error": float(L2_err),
        },
    }


if __name__ == "__main__":
    import time
    case_spec = {
        "output": {"grid": {"nx": 64, "ny": 64, "bbox": [0.0, 1.0, 0.0, 1.0]}}
    }
    t0 = time.time()
    result = solve(case_spec)
    t1 = time.time()
    print(f"Time: {t1 - t0:.2f}s")
    print(f"L2 error: {result['solver_info']['L2_error']:.3e}")
    print(f"Shape: {result['u'].shape}")

    # Compare with exact on grid
    xs = np.linspace(0, 1, 64)
    ys = np.linspace(0, 1, 64)
    XX, YY = np.meshgrid(xs, ys)
    u_ex = np.sin(np.pi * XX) * np.sin(2 * np.pi * YY)
    err = np.max(np.abs(result["u"] - u_ex))
    print(f"Max error vs exact on grid: {err:.3e}")
