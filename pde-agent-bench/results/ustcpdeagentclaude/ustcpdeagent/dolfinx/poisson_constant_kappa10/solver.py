import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc


def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD
    kappa_val = 10.0

    N = 64
    degree = 2

    domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", degree))

    x = ufl.SpatialCoordinate(domain)
    u_exact = ufl.sin(ufl.pi * x[0]) * ufl.sin(2 * ufl.pi * x[1])
    f_expr = kappa_val * 5.0 * ufl.pi**2 * ufl.sin(ufl.pi * x[0]) * ufl.sin(2 * ufl.pi * x[1])

    # BC
    fdim = domain.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
    )
    u_bc = fem.Function(V)
    u_bc.interpolate(fem.Expression(u_exact, V.element.interpolation_points))
    bdofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc = fem.dirichletbc(u_bc, bdofs)

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    kappa = fem.Constant(domain, PETSc.ScalarType(kappa_val))
    a = kappa * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = f_expr * v * ufl.dx

    rtol = 1e-10
    problem = petsc.LinearProblem(
        a, L, bcs=[bc],
        petsc_options={"ksp_type": "cg", "pc_type": "hypre", "ksp_rtol": rtol},
        petsc_options_prefix="poisson_",
    )
    u_sol = problem.solve()
    iters = problem.solver.getIterationNumber()

    # Sample on grid
    grid = case_spec["output"]["grid"]
    nx = grid["nx"]; ny = grid["ny"]; bbox = grid["bbox"]
    xs = np.linspace(bbox[0], bbox[1], nx)
    ys = np.linspace(bbox[2], bbox[3], ny)
    XX, YY = np.meshgrid(xs, ys)
    pts = np.c_[XX.ravel(), YY.ravel(), np.zeros(nx*ny)]

    tree = geometry.bb_tree(domain, domain.topology.dim)
    cand = geometry.compute_collisions_points(tree, pts)
    col = geometry.compute_colliding_cells(domain, cand, pts)

    cells = []
    pts_on = []
    eval_idx = []
    for i in range(pts.shape[0]):
        links = col.links(i)
        if len(links) > 0:
            pts_on.append(pts[i])
            cells.append(links[0])
            eval_idx.append(i)
    vals = np.full(pts.shape[0], np.nan)
    if pts_on:
        v_eval = u_sol.eval(np.array(pts_on), np.array(cells, dtype=np.int32))
        vals[eval_idx] = v_eval.flatten()
    u_grid = vals.reshape(ny, nx)

    return {
        "u": u_grid,
        "solver_info": {
            "mesh_resolution": N,
            "element_degree": degree,
            "ksp_type": "cg",
            "pc_type": "hypre",
            "rtol": rtol,
            "iterations": int(iters),
        }
    }


if __name__ == "__main__":
    import time
    case_spec = {
        "output": {"grid": {"nx": 128, "ny": 128, "bbox": [0, 1, 0, 1]}}
    }
    t0 = time.time()
    res = solve(case_spec)
    dt = time.time() - t0
    u = res["u"]
    # exact
    xs = np.linspace(0, 1, 128)
    ys = np.linspace(0, 1, 128)
    XX, YY = np.meshgrid(xs, ys)
    u_ex = np.sin(np.pi * XX) * np.sin(2 * np.pi * YY)
    err = np.sqrt(np.mean((u - u_ex)**2))
    print(f"Time: {dt:.3f}s, RMS err: {err:.3e}, iters: {res['solver_info']['iterations']}")
