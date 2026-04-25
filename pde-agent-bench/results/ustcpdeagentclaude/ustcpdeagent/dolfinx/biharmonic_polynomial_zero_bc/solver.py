import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc


def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD
    N = 48
    degree = 2
    msh = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)

    V = fem.functionspace(msh, ("Lagrange", degree))
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    # Step 1: Solve -Δw = f, with w = 0 on ∂Ω
    # f = 8 (constant for the manufactured solution)
    f = fem.Constant(msh, PETSc.ScalarType(8.0))
    a = ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = ufl.inner(f, v) * ufl.dx

    fdim = msh.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(
        msh, fdim, lambda x: np.ones(x.shape[1], dtype=bool))
    dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    zero_func = fem.Function(V)
    zero_func.x.array[:] = 0.0
    bc = fem.dirichletbc(zero_func, dofs)

    # BC for w: w = -Delta u_exact = 2x(1-x) + 2y(1-y)
    w_bc_func = fem.Function(V)
    w_bc_func.interpolate(lambda x: 2*x[0]*(1-x[0]) + 2*x[1]*(1-x[1]))
    bc_w = fem.dirichletbc(w_bc_func, dofs)
    w = fem.Function(V)
    problem1 = petsc.LinearProblem(
        a, L, bcs=[bc_w],
        petsc_options={"ksp_type": "preonly", "pc_type": "lu",
                       "pc_factor_mat_solver_type": "mumps"},
        petsc_options_prefix="biharm1_"
    )
    w_sol = problem1.solve()
    w_sol.x.scatter_forward()
    iters1 = problem1.solver.getIterationNumber()

    # Step 2: Solve -Δu = w, with u = 0 on ∂Ω
    a2 = ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
    L2 = ufl.inner(w_sol, v) * ufl.dx

    u_sol = fem.Function(V)
    problem2 = petsc.LinearProblem(
        a2, L2, bcs=[bc],
        petsc_options={"ksp_type": "preonly", "pc_type": "lu",
                       "pc_factor_mat_solver_type": "mumps"},
        petsc_options_prefix="biharm2_"
    )
    u_sol = problem2.solve()
    u_sol.x.scatter_forward()
    iters2 = problem2.solver.getIterationNumber()

    # Sample on grid
    grid = case_spec["output"]["grid"]
    nx, ny = grid["nx"], grid["ny"]
    bbox = grid["bbox"]
    xs = np.linspace(bbox[0], bbox[1], nx)
    ys = np.linspace(bbox[2], bbox[3], ny)
    XX, YY = np.meshgrid(xs, ys)
    pts = np.c_[XX.ravel(), YY.ravel(), np.zeros(nx*ny)]

    tree = geometry.bb_tree(msh, msh.topology.dim)
    cand = geometry.compute_collisions_points(tree, pts)
    coll = geometry.compute_colliding_cells(msh, cand, pts)

    points_on_proc = []
    cells = []
    idxs = []
    for i in range(pts.shape[0]):
        links = coll.links(i)
        if len(links) > 0:
            points_on_proc.append(pts[i])
            cells.append(links[0])
            idxs.append(i)
    vals = np.full(pts.shape[0], np.nan)
    if points_on_proc:
        v = u_sol.eval(np.array(points_on_proc), np.array(cells, dtype=np.int32))
        vals[idxs] = v.flatten()
    u_grid = vals.reshape(ny, nx)

    return {
        "u": u_grid,
        "solver_info": {
            "mesh_resolution": N,
            "element_degree": degree,
            "ksp_type": "preonly",
            "pc_type": "lu",
            "rtol": 1e-10,
            "iterations": int(iters1 + iters2),
        }
    }


if __name__ == "__main__":
    import time
    spec = {"output": {"grid": {"nx": 64, "ny": 64, "bbox": [0, 1, 0, 1]}}}
    t0 = time.time()
    r = solve(spec)
    t1 = time.time()
    print(f"Time: {t1-t0:.3f}s")
    xs = np.linspace(0, 1, 64)
    ys = np.linspace(0, 1, 64)
    XX, YY = np.meshgrid(xs, ys)
    exact = XX*(1-XX)*YY*(1-YY)
    err = np.sqrt(np.mean((r["u"] - exact)**2))
    print(f"RMSE: {err:.3e}")
    print(f"Max err: {np.max(np.abs(r['u']-exact)):.3e}")
