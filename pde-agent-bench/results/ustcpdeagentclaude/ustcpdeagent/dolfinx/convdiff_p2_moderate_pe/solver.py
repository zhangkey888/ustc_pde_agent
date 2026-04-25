import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc


def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD
    eps_val = 0.03
    beta_vec = np.array([5.0, 2.0])

    N = 128
    degree = 2

    msh = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(msh, ("Lagrange", degree))

    x = ufl.SpatialCoordinate(msh)
    u_exact = ufl.sin(ufl.pi * x[0]) * ufl.sin(2 * ufl.pi * x[1])

    eps_c = fem.Constant(msh, PETSc.ScalarType(eps_val))
    beta = fem.Constant(msh, PETSc.ScalarType((beta_vec[0], beta_vec[1])))

    # Source term derived from manufactured solution
    # -eps*laplacian(u) + beta·grad(u) = f
    f_expr = -eps_c * ufl.div(ufl.grad(u_exact)) + ufl.dot(beta, ufl.grad(u_exact))

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    # Galerkin
    a = eps_c * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx \
        + ufl.inner(ufl.dot(beta, ufl.grad(u)), v) * ufl.dx
    L = ufl.inner(f_expr, v) * ufl.dx

    # SUPG stabilization
    h = ufl.CellDiameter(msh)
    beta_norm = ufl.sqrt(ufl.dot(beta, beta))
    Pe = beta_norm * h / (2.0 * eps_c)
    # tau
    tau = (h / (2.0 * beta_norm)) * (1.0 / ufl.tanh(Pe) - 1.0 / Pe)

    # Residual-based SUPG: add tau*(beta·grad(v)) * R(u) dx
    # R(u) = -eps*div(grad(u)) + beta·grad(u) - f
    R_u = -eps_c * ufl.div(ufl.grad(u)) + ufl.dot(beta, ufl.grad(u))
    R_f = f_expr
    a += tau * ufl.inner(ufl.dot(beta, ufl.grad(v)), R_u) * ufl.dx
    L += tau * ufl.inner(ufl.dot(beta, ufl.grad(v)), R_f) * ufl.dx

    # Dirichlet BC from exact solution
    u_bc = fem.Function(V)
    u_bc.interpolate(fem.Expression(u_exact, V.element.interpolation_points))

    fdim = msh.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(
        msh, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
    )
    bdofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc = fem.dirichletbc(u_bc, bdofs)

    problem = petsc.LinearProblem(
        a, L, bcs=[bc],
        petsc_options={"ksp_type": "preonly", "pc_type": "lu",
                       "pc_factor_mat_solver_type": "mumps"},
        petsc_options_prefix="cd_"
    )
    u_sol = problem.solve()

    try:
        its = problem.solver.getIterationNumber()
    except Exception:
        its = 1

    # Sample on output grid
    grid = case_spec["output"]["grid"]
    nx = grid["nx"]
    ny = grid["ny"]
    bbox = grid["bbox"]
    xs = np.linspace(bbox[0], bbox[1], nx)
    ys = np.linspace(bbox[2], bbox[3], ny)
    XX, YY = np.meshgrid(xs, ys)
    pts = np.column_stack([XX.ravel(), YY.ravel(), np.zeros(nx * ny)])

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

    vals = np.full(pts.shape[0], np.nan)
    if len(points_on_proc) > 0:
        v_eval = u_sol.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        vals[eval_map] = v_eval.flatten()

    u_grid = vals.reshape(ny, nx)

    # Accuracy check
    u_true = np.sin(np.pi * XX) * np.sin(2 * np.pi * YY)
    err = np.sqrt(np.mean((u_grid - u_true) ** 2))
    print(f"RMS error: {err:.3e}")

    return {
        "u": u_grid,
        "solver_info": {
            "mesh_resolution": N,
            "element_degree": degree,
            "ksp_type": "preonly",
            "pc_type": "lu",
            "rtol": 1e-12,
            "iterations": int(its),
        }
    }


if __name__ == "__main__":
    import time
    case_spec = {
        "output": {"grid": {"nx": 65, "ny": 65, "bbox": [0, 1, 0, 1]}}
    }
    t0 = time.time()
    result = solve(case_spec)
    elapsed = time.time() - t0
    print(f"Time: {elapsed:.3f}s")
    XX, YY = np.meshgrid(np.linspace(0, 1, 65), np.linspace(0, 1, 65))
    u_true = np.sin(np.pi * XX) * np.sin(2 * np.pi * YY)
    err = np.sqrt(np.mean((result["u"] - u_true) ** 2))
    maxerr = np.max(np.abs(result["u"] - u_true))
    print(f"RMS: {err:.3e}, Max: {maxerr:.3e}")
