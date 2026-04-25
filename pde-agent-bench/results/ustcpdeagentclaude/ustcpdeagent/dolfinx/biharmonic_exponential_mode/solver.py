import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc


def solve(case_spec: dict) -> dict:
    grid = case_spec["output"]["grid"]
    nx_out = grid["nx"]
    ny_out = grid["ny"]
    bbox = grid["bbox"]

    comm = MPI.COMM_WORLD
    N = 128
    degree = 3
    domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)

    V = fem.functionspace(domain, ("Lagrange", degree))

    x = ufl.SpatialCoordinate(domain)
    u_exact = ufl.exp(x[0]) * ufl.sin(ufl.pi * x[1])
    # Δu = (1 - pi^2)*exp(x)*sin(pi*y)
    lap_u = (1.0 - ufl.pi**2) * ufl.exp(x[0]) * ufl.sin(ufl.pi * x[1])
    # w = -Δu
    w_exact = -lap_u
    # f = Δ²u = (1-pi^2)^2 * exp(x)*sin(pi*y)
    f_expr = (1.0 - ufl.pi**2) ** 2 * ufl.exp(x[0]) * ufl.sin(ufl.pi * x[1])

    fdim = domain.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
    )

    # Solve 1: -Δw = f, with w = w_exact on boundary
    w_bc = fem.Function(V)
    w_bc.interpolate(fem.Expression(w_exact, V.element.interpolation_points))
    dofs_V = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc_w = fem.dirichletbc(w_bc, dofs_V)

    u_tr = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    a_w = ufl.inner(ufl.grad(u_tr), ufl.grad(v)) * ufl.dx
    L_w = ufl.inner(f_expr, v) * ufl.dx

    problem_w = petsc.LinearProblem(
        a_w, L_w, bcs=[bc_w],
        petsc_options={"ksp_type": "cg", "pc_type": "hypre", "ksp_rtol": 1e-12},
        petsc_options_prefix="bih_w_",
    )
    w_sol = problem_w.solve()
    it_w = problem_w.solver.getIterationNumber()

    # Solve 2: -Δu = w_sol, with u = u_exact on boundary
    u_bc = fem.Function(V)
    u_bc.interpolate(fem.Expression(u_exact, V.element.interpolation_points))
    bc_u = fem.dirichletbc(u_bc, dofs_V)

    a_u = ufl.inner(ufl.grad(u_tr), ufl.grad(v)) * ufl.dx
    L_u = ufl.inner(w_sol, v) * ufl.dx

    problem_u = petsc.LinearProblem(
        a_u, L_u, bcs=[bc_u],
        petsc_options={"ksp_type": "cg", "pc_type": "hypre", "ksp_rtol": 1e-12},
        petsc_options_prefix="bih_u_",
    )
    u_sol = problem_u.solve()
    it_u = problem_u.solver.getIterationNumber()

    # Sample on uniform grid
    xs = np.linspace(bbox[0], bbox[1], nx_out)
    ys = np.linspace(bbox[2], bbox[3], ny_out)
    XX, YY = np.meshgrid(xs, ys)
    pts = np.c_[XX.ravel(), YY.ravel(), np.zeros(nx_out * ny_out)]

    tree = geometry.bb_tree(domain, domain.topology.dim)
    cand = geometry.compute_collisions_points(tree, pts)
    coll = geometry.compute_colliding_cells(domain, cand, pts)

    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    for i in range(pts.shape[0]):
        links = coll.links(i)
        if len(links) > 0:
            points_on_proc.append(pts[i])
            cells_on_proc.append(links[0])
            eval_map.append(i)

    u_values = np.full((pts.shape[0],), np.nan)
    if len(points_on_proc) > 0:
        vals = u_sol.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u_values[eval_map] = vals.flatten()

    u_grid = u_values.reshape(ny_out, nx_out)

    # Accuracy check (print)
    U_exact = np.exp(XX) * np.sin(np.pi * YY)
    err = np.max(np.abs(u_grid - U_exact))
    print(f"[biharmonic] max err vs exact = {err:.3e}, iters w={it_w}, u={it_u}")

    return {
        "u": u_grid,
        "solver_info": {
            "mesh_resolution": N,
            "element_degree": degree,
            "ksp_type": "cg",
            "pc_type": "hypre",
            "rtol": 1e-12,
            "iterations": int(it_w + it_u),
        },
    }


if __name__ == "__main__":
    import time
    spec = {"output": {"grid": {"nx": 64, "ny": 64, "bbox": [0.0, 1.0, 0.0, 1.0]}}}
    t0 = time.time()
    res = solve(spec)
    t1 = time.time()
    print(f"wall time = {t1 - t0:.2f}s")
    U_exact = np.exp(np.linspace(0,1,64)[None,:]) * np.sin(np.pi * np.linspace(0,1,64)[:,None])
    print(f"L2 err on grid = {np.sqrt(np.mean((res['u']-U_exact)**2)):.3e}")
