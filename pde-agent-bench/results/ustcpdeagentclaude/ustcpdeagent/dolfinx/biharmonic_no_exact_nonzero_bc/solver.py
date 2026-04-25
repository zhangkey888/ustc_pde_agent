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
    N = 144
    degree = 2
    msh = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)

    V = fem.functionspace(msh, ("Lagrange", degree))

    x = ufl.SpatialCoordinate(msh)
    pi = np.pi
    u_exact = ufl.sin(3*pi*x[0]) + ufl.cos(2*pi*x[1])
    # w = -Delta u
    w_exact = 9*pi*pi*ufl.sin(3*pi*x[0]) + 4*pi*pi*ufl.cos(2*pi*x[1])
    # f = Delta^2 u
    f_expr = 81*pi**4*ufl.sin(3*pi*x[0]) + 16*pi**4*ufl.cos(2*pi*x[1])

    fdim = msh.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(
        msh, fdim, lambda xx: np.ones(xx.shape[1], dtype=bool)
    )
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)

    # Mixed formulation: solve -Delta w = f with w = w_exact on boundary
    # then -Delta u = w with u = u_exact on boundary
    w_bc_func = fem.Function(V)
    w_bc_func.interpolate(fem.Expression(w_exact, V.element.interpolation_points))
    bc_w = fem.dirichletbc(w_bc_func, boundary_dofs)

    u_bc_func = fem.Function(V)
    u_bc_func.interpolate(fem.Expression(u_exact, V.element.interpolation_points))
    bc_u = fem.dirichletbc(u_bc_func, boundary_dofs)

    # First solve: -Delta w = f
    w_tr = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    a1 = ufl.inner(ufl.grad(w_tr), ufl.grad(v)) * ufl.dx
    L1 = ufl.inner(f_expr, v) * ufl.dx

    total_iters = 0
    petsc_opts = {"ksp_type": "cg", "pc_type": "hypre", "ksp_rtol": 1e-12, "ksp_atol": 1e-14}

    problem1 = petsc.LinearProblem(a1, L1, bcs=[bc_w],
                                    petsc_options=petsc_opts,
                                    petsc_options_prefix="bh1_")
    w_sol = problem1.solve()
    try:
        total_iters += problem1.solver.getIterationNumber()
    except Exception:
        pass

    # Second solve: -Delta u = w
    u_tr = ufl.TrialFunction(V)
    a2 = ufl.inner(ufl.grad(u_tr), ufl.grad(v)) * ufl.dx
    L2 = ufl.inner(w_sol, v) * ufl.dx
    problem2 = petsc.LinearProblem(a2, L2, bcs=[bc_u],
                                    petsc_options=petsc_opts,
                                    petsc_options_prefix="bh2_")
    u_sol = problem2.solve()
    try:
        total_iters += problem2.solver.getIterationNumber()
    except Exception:
        pass

    # Sample on uniform grid
    xs = np.linspace(bbox[0], bbox[1], nx_out)
    ys = np.linspace(bbox[2], bbox[3], ny_out)
    XX, YY = np.meshgrid(xs, ys)
    pts = np.vstack([XX.ravel(), YY.ravel(), np.zeros(nx_out*ny_out)]).T

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

    return {
        "u": u_grid,
        "solver_info": {
            "mesh_resolution": N,
            "element_degree": degree,
            "ksp_type": "cg",
            "pc_type": "hypre",
            "rtol": 1e-12,
            "iterations": int(total_iters),
        }
    }


if __name__ == "__main__":
    import time
    case_spec = {"output": {"grid": {"nx": 64, "ny": 64, "bbox": [0,1,0,1]}}}
    t0 = time.time()
    res = solve(case_spec)
    t1 = time.time()
    u = res["u"]
    xs = np.linspace(0, 1, 64)
    ys = np.linspace(0, 1, 64)
    XX, YY = np.meshgrid(xs, ys)
    u_ex = np.sin(3*np.pi*XX) + np.cos(2*np.pi*YY)
    err = np.sqrt(np.mean((u - u_ex)**2))
    maxerr = np.max(np.abs(u - u_ex))
    print(f"Time: {t1-t0:.3f}s")
    print(f"RMS error: {err:.3e}")
    print(f"Max error: {maxerr:.3e}")
    print(f"Iters: {res['solver_info']['iterations']}")
