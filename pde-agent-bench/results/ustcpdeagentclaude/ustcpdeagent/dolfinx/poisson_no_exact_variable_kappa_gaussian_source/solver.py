import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc


def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD
    N = 128
    degree = 2

    domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", degree))

    x = ufl.SpatialCoordinate(domain)
    kappa = 1.0 + 50.0 * ufl.exp(-150.0 * ((x[0] - 0.5) ** 2 + (x[1] - 0.5) ** 2))
    f = ufl.exp(-250.0 * ((x[0] - 0.4) ** 2 + (x[1] - 0.6) ** 2))

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    a = kappa * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = f * v * ufl.dx

    fdim = domain.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda xx: np.ones(xx.shape[1], dtype=bool)
    )
    dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    u_bc = fem.Function(V)
    u_bc.x.array[:] = 0.0
    bc = fem.dirichletbc(u_bc, dofs)

    rtol = 1e-10
    problem = petsc.LinearProblem(
        a, L, bcs=[bc],
        petsc_options={"ksp_type": "cg", "pc_type": "hypre",
                       "ksp_rtol": rtol, "ksp_atol": 1e-14},
        petsc_options_prefix="poisson_",
    )
    u_sol = problem.solve()
    ksp = problem.solver
    iters = ksp.getIterationNumber()

    # Sample on grid
    gspec = case_spec["output"]["grid"]
    nx, ny = gspec["nx"], gspec["ny"]
    bbox = gspec["bbox"]
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

    u_vals = np.zeros(pts.shape[0])
    if len(points_on_proc) > 0:
        vals = u_sol.eval(np.array(points_on_proc), np.array(cells, dtype=np.int32))
        u_vals[eval_map] = vals.flatten()

    u_grid = u_vals.reshape(ny, nx)

    return {
        "u": u_grid,
        "solver_info": {
            "mesh_resolution": N,
            "element_degree": degree,
            "ksp_type": "cg",
            "pc_type": "hypre",
            "rtol": rtol,
            "iterations": int(iters),
        },
    }


if __name__ == "__main__":
    import time
    spec = {"output": {"grid": {"nx": 128, "ny": 128, "bbox": [0, 1, 0, 1]}}}
    t0 = time.time()
    out = solve(spec)
    print(f"Time: {time.time() - t0:.3f}s")
    print(f"u shape: {out['u'].shape}, max: {out['u'].max():.6e}, min: {out['u'].min():.6e}")
    print(f"Iterations: {out['solver_info']['iterations']}")
