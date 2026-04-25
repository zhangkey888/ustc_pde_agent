import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc


def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD
    N = 96
    degree = 2
    domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)

    V = fem.functionspace(domain, ("Lagrange", degree))
    x = ufl.SpatialCoordinate(domain)

    # Exact solution and kappa
    u_exact = ufl.sin(2*ufl.pi*x[0]) * ufl.sin(2*ufl.pi*x[1])
    kappa = 1 + 0.3*ufl.sin(8*ufl.pi*x[0])*ufl.sin(8*ufl.pi*x[1])

    # f = -div(kappa grad(u_exact))
    f = -ufl.div(kappa * ufl.grad(u_exact))

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    a = ufl.inner(kappa * ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = ufl.inner(f, v) * ufl.dx

    # Dirichlet BC from exact solution
    fdim = domain.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda xx: np.ones(xx.shape[1], dtype=bool)
    )
    dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    u_bc = fem.Function(V)
    u_bc.interpolate(fem.Expression(u_exact, V.element.interpolation_points))
    bc = fem.dirichletbc(u_bc, dofs)

    rtol = 1e-10
    problem = petsc.LinearProblem(
        a, L, bcs=[bc],
        petsc_options={"ksp_type": "cg", "pc_type": "hypre",
                       "ksp_rtol": rtol, "ksp_atol": 1e-14},
        petsc_options_prefix="poisson_"
    )
    u_sol = problem.solve()
    its = problem.solver.getIterationNumber()

    # Sample on grid
    grid = case_spec["output"]["grid"]
    nx = grid["nx"]; ny = grid["ny"]
    bbox = grid["bbox"]
    xs = np.linspace(bbox[0], bbox[1], nx)
    ys = np.linspace(bbox[2], bbox[3], ny)
    XX, YY = np.meshgrid(xs, ys)
    pts = np.column_stack([XX.ravel(), YY.ravel(), np.zeros(nx*ny)])

    tree = geometry.bb_tree(domain, domain.topology.dim)
    cand = geometry.compute_collisions_points(tree, pts)
    coll = geometry.compute_colliding_cells(domain, cand, pts)

    cells = []
    points_on_proc = []
    idx_map = []
    for i in range(pts.shape[0]):
        links = coll.links(i)
        if len(links) > 0:
            points_on_proc.append(pts[i])
            cells.append(links[0])
            idx_map.append(i)

    u_values = np.zeros(nx*ny)
    if points_on_proc:
        vals = u_sol.eval(np.array(points_on_proc), np.array(cells, dtype=np.int32))
        u_values[idx_map] = vals.flatten()

    u_grid = u_values.reshape(ny, nx)

    return {
        "u": u_grid,
        "solver_info": {
            "mesh_resolution": N,
            "element_degree": degree,
            "ksp_type": "cg",
            "pc_type": "hypre",
            "rtol": rtol,
            "iterations": int(its),
        }
    }


if __name__ == "__main__":
    import time
    case_spec = {"output": {"grid": {"nx": 64, "ny": 64, "bbox": [0, 1, 0, 1]}}}
    t0 = time.time()
    out = solve(case_spec)
    t1 = time.time()
    print(f"Wall time: {t1-t0:.3f}s, iters: {out['solver_info']['iterations']}")

    # Error
    nx, ny = 64, 64
    xs = np.linspace(0, 1, nx); ys = np.linspace(0, 1, ny)
    XX, YY = np.meshgrid(xs, ys)
    u_ex = np.sin(2*np.pi*XX) * np.sin(2*np.pi*YY)
    err = np.sqrt(np.mean((out["u"] - u_ex)**2))
    maxerr = np.max(np.abs(out["u"] - u_ex))
    print(f"RMSE: {err:.3e}, MaxErr: {maxerr:.3e}")
