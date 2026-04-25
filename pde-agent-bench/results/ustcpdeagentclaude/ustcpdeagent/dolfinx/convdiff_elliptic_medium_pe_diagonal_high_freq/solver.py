import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc


def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD
    eps = 0.05
    bx, by = 3.0, 3.0

    N = 64
    degree = 2
    domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)

    V = fem.functionspace(domain, ("Lagrange", degree))

    x = ufl.SpatialCoordinate(domain)
    u_exact = ufl.sin(4 * ufl.pi * x[0]) * ufl.sin(3 * ufl.pi * x[1])
    # f = -eps*lap(u) + beta . grad(u)
    grad_u = ufl.grad(u_exact)
    lap_u = ufl.div(grad_u)
    beta = ufl.as_vector([bx, by])
    f_expr = -eps * lap_u + ufl.dot(beta, grad_u)

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    # Galerkin
    a = eps * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx \
        + ufl.inner(ufl.dot(beta, ufl.grad(u)), v) * ufl.dx
    L = ufl.inner(f_expr, v) * ufl.dx

    # SUPG stabilization
    h = ufl.CellDiameter(domain)
    bnorm = ufl.sqrt(ufl.dot(beta, beta))
    Pe_h = bnorm * h / (2 * eps)
    # tau (optimal for convection-diffusion)
    tau = (h / (2 * bnorm)) * (1.0 / ufl.tanh(Pe_h) - 1.0 / Pe_h)

    # residual
    res_trial = -eps * ufl.div(ufl.grad(u)) + ufl.dot(beta, ufl.grad(u))
    res_test = ufl.dot(beta, ufl.grad(v))
    a += tau * ufl.inner(res_trial, res_test) * ufl.dx
    L += tau * ufl.inner(f_expr, res_test) * ufl.dx

    # BC: exact solution on boundary
    u_bc = fem.Function(V)
    u_bc.interpolate(fem.Expression(u_exact, V.element.interpolation_points))
    fdim = domain.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda xx: np.ones(xx.shape[1], dtype=bool)
    )
    dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc = fem.dirichletbc(u_bc, dofs)

    problem = petsc.LinearProblem(
        a, L, bcs=[bc],
        petsc_options={"ksp_type": "gmres", "pc_type": "ilu", "ksp_rtol": 1e-10},
        petsc_options_prefix="cd_",
    )
    u_sol = problem.solve()

    ksp = problem.solver
    its = ksp.getIterationNumber()

    # Sample onto grid
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

    cells = []
    points_on_proc = []
    eval_map = []
    for i in range(pts.shape[0]):
        links = colliding.links(i)
        if len(links) > 0:
            points_on_proc.append(pts[i])
            cells.append(links[0])
            eval_map.append(i)

    u_grid = np.zeros(nx * ny)
    if points_on_proc:
        vals = u_sol.eval(np.array(points_on_proc), np.array(cells, dtype=np.int32))
        u_grid[eval_map] = vals.flatten()
    u_grid = u_grid.reshape(ny, nx)

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
    case = {"output": {"grid": {"nx": 64, "ny": 64, "bbox": [0, 1, 0, 1]}}}
    t0 = time.time()
    out = solve(case)
    t1 = time.time()
    print(f"Time: {t1-t0:.3f}s")
    u = out["u"]
    nx = 64; ny = 64
    xs = np.linspace(0, 1, nx)
    ys = np.linspace(0, 1, ny)
    XX, YY = np.meshgrid(xs, ys)
    u_ex = np.sin(4*np.pi*XX) * np.sin(3*np.pi*YY)
    err = np.sqrt(np.mean((u - u_ex)**2))
    print(f"RMS error: {err:.3e}")
    print(f"Max error: {np.max(np.abs(u-u_ex)):.3e}")
    print(f"Iterations: {out['solver_info']['iterations']}")
