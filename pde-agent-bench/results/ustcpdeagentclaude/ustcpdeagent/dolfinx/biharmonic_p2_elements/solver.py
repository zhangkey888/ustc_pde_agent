import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc


def solve(case_spec: dict) -> dict:
    grid = case_spec["output"]["grid"]
    nx_out, ny_out = grid["nx"], grid["ny"]
    bbox = grid["bbox"]

    comm = MPI.COMM_WORLD
    N = 128
    degree = 2
    msh = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)

    V = fem.functionspace(msh, ("Lagrange", degree))

    x = ufl.SpatialCoordinate(msh)
    pi = np.pi
    u_exact = ufl.sin(2*pi*x[0]) * ufl.sin(2*pi*x[1])
    f_expr = 64.0 * pi**4 * ufl.sin(2*pi*x[0]) * ufl.sin(2*pi*x[1])
    # w = -Δu = 8π² sin(2πx)sin(2πy), so w = 0 on boundary
    w_bc_expr = 8.0 * pi**2 * ufl.sin(2*pi*x[0]) * ufl.sin(2*pi*x[1])  # equals 0 on boundary

    fdim = msh.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(
        msh, fdim, lambda xx: np.ones(xx.shape[1], dtype=bool)
    )
    dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)

    # Solve 1: -Δw = f, w = 0 on boundary
    w_trial = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    a1 = ufl.inner(ufl.grad(w_trial), ufl.grad(v)) * ufl.dx
    L1 = f_expr * v * ufl.dx

    w_bc = fem.Function(V)
    w_bc.x.array[:] = 0.0
    bc1 = fem.dirichletbc(w_bc, dofs)

    total_iters = 0
    prob1 = petsc.LinearProblem(
        a1, L1, bcs=[bc1],
        petsc_options={"ksp_type": "cg", "pc_type": "hypre", "ksp_rtol": 1e-12},
        petsc_options_prefix="bih1_"
    )
    w_sol = prob1.solve()
    total_iters += prob1.solver.getIterationNumber()

    # Solve 2: -Δu = w, u = 0 on boundary
    u_trial = ufl.TrialFunction(V)
    a2 = ufl.inner(ufl.grad(u_trial), ufl.grad(v)) * ufl.dx
    L2 = w_sol * v * ufl.dx

    u_bc = fem.Function(V)
    u_bc.x.array[:] = 0.0
    bc2 = fem.dirichletbc(u_bc, dofs)

    prob2 = petsc.LinearProblem(
        a2, L2, bcs=[bc2],
        petsc_options={"ksp_type": "cg", "pc_type": "hypre", "ksp_rtol": 1e-12},
        petsc_options_prefix="bih2_"
    )
    u_sol = prob2.solve()
    total_iters += prob2.solver.getIterationNumber()

    # Verification
    err_form = fem.form((u_sol - u_exact)**2 * ufl.dx)
    err_l2 = np.sqrt(comm.allreduce(fem.assemble_scalar(err_form), op=MPI.SUM))
    print(f"L2 error: {err_l2:.3e}")

    # Sample on grid
    xs = np.linspace(bbox[0], bbox[1], nx_out)
    ys = np.linspace(bbox[2], bbox[3], ny_out)
    XX, YY = np.meshgrid(xs, ys)
    pts = np.column_stack([XX.ravel(), YY.ravel(), np.zeros(nx_out*ny_out)])

    tree = geometry.bb_tree(msh, msh.topology.dim)
    cell_candidates = geometry.compute_collisions_points(tree, pts)
    colliding = geometry.compute_colliding_cells(msh, cell_candidates, pts)

    points_on_proc = []
    cells = []
    eval_map = []
    for i in range(pts.shape[0]):
        links = colliding.links(i)
        if len(links) > 0:
            points_on_proc.append(pts[i])
            cells.append(links[0])
            eval_map.append(i)

    vals = np.zeros(nx_out*ny_out)
    if len(points_on_proc) > 0:
        arr = u_sol.eval(np.array(points_on_proc), np.array(cells, dtype=np.int32))
        vals[eval_map] = arr.flatten()

    u_grid = vals.reshape(ny_out, nx_out)

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
    case_spec = {
        "output": {
            "grid": {"nx": 64, "ny": 64, "bbox": [0, 1, 0, 1]}
        }
    }
    t0 = time.time()
    res = solve(case_spec)
    t1 = time.time()
    print(f"Time: {t1-t0:.3f}s")
    # Compare with exact
    xs = np.linspace(0, 1, 64)
    ys = np.linspace(0, 1, 64)
    XX, YY = np.meshgrid(xs, ys)
    u_ex = np.sin(2*np.pi*XX)*np.sin(2*np.pi*YY)
    err = np.max(np.abs(res["u"] - u_ex))
    print(f"Max error on grid: {err:.3e}")
    print(f"L2 rel error on grid: {np.linalg.norm(res['u']-u_ex)/np.linalg.norm(u_ex):.3e}")
