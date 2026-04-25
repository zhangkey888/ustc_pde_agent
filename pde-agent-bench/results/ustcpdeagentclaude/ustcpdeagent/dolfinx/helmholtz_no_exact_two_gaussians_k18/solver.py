import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc


def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD
    k_val = 18.0

    N = 256
    degree = 2

    domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", degree))

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    x = ufl.SpatialCoordinate(domain)
    f = 12.0 * (ufl.exp(-90.0*((x[0]-0.3)**2 + (x[1]-0.7)**2))
                - ufl.exp(-90.0*((x[0]-0.7)**2 + (x[1]-0.3)**2)))

    k = fem.Constant(domain, PETSc.ScalarType(k_val))

    a = ufl.inner(ufl.grad(u), ufl.grad(v))*ufl.dx - k*k*ufl.inner(u, v)*ufl.dx
    L = ufl.inner(f, v)*ufl.dx

    # BC u=0 on all boundary
    fdim = domain.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
    )
    dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    u_bc = fem.Function(V)
    u_bc.x.array[:] = 0.0
    bc = fem.dirichletbc(u_bc, dofs)

    problem = petsc.LinearProblem(
        a, L, bcs=[bc],
        petsc_options={"ksp_type": "preonly", "pc_type": "lu",
                       "pc_factor_mat_solver_type": "mumps"},
        petsc_options_prefix="helm_"
    )
    u_sol = problem.solve()

    try:
        iters = problem.solver.getIterationNumber()
    except Exception:
        iters = 1

    # Sample on output grid
    grid = case_spec["output"]["grid"]
    nx = grid["nx"]; ny = grid["ny"]
    bbox = grid["bbox"]
    xs = np.linspace(bbox[0], bbox[1], nx)
    ys = np.linspace(bbox[2], bbox[3], ny)
    XX, YY = np.meshgrid(xs, ys)
    pts = np.column_stack([XX.ravel(), YY.ravel(), np.zeros(nx*ny)])

    tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(tree, pts)
    colliding = geometry.compute_colliding_cells(domain, cell_candidates, pts)

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
        vals = u_sol.eval(np.array(points_on_proc),
                          np.array(cells_on_proc, dtype=np.int32))
        u_values[eval_map] = vals.flatten()

    u_grid = u_values.reshape(ny, nx)

    return {
        "u": u_grid,
        "solver_info": {
            "mesh_resolution": N,
            "element_degree": degree,
            "ksp_type": "preonly",
            "pc_type": "lu",
            "rtol": 1e-12,
            "iterations": int(iters),
        }
    }


if __name__ == "__main__":
    import time
    spec = {"output": {"grid": {"nx": 128, "ny": 128, "bbox": [0.0, 1.0, 0.0, 1.0]}}}
    t0 = time.time()
    out = solve(spec)
    t1 = time.time()
    print(f"time={t1-t0:.2f}s shape={out['u'].shape} min={out['u'].min():.4f} max={out['u'].max():.4f}")
    print("info:", out["solver_info"])
