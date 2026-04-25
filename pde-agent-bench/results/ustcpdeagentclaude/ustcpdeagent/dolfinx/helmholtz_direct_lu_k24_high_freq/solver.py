import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc


def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD
    k = 24.0

    grid = case_spec["output"]["grid"]
    nx_out = grid["nx"]
    ny_out = grid["ny"]
    bbox = grid["bbox"]

    N = 160
    degree = 2
    domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", degree))

    x = ufl.SpatialCoordinate(domain)
    u_exact = ufl.sin(5*ufl.pi*x[0]) * ufl.sin(4*ufl.pi*x[1])
    # -laplace(u_exact) = (25+16)pi^2 * u_exact = 41 pi^2 u
    # f = -lap(u) - k^2 u = 41 pi^2 u - k^2 u
    f_expr = (41.0 * ufl.pi**2 - k**2) * u_exact

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    a = ufl.inner(ufl.grad(u), ufl.grad(v))*ufl.dx - k*k*ufl.inner(u, v)*ufl.dx
    L = ufl.inner(f_expr, v)*ufl.dx

    # BC from exact solution (zero on boundary for this manufactured solution)
    u_bc = fem.Function(V)
    u_bc.interpolate(fem.Expression(u_exact, V.element.interpolation_points))
    fdim = domain.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool))
    dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc = fem.dirichletbc(u_bc, dofs)

    problem = petsc.LinearProblem(
        a, L, bcs=[bc],
        petsc_options={"ksp_type": "preonly", "pc_type": "lu",
                       "pc_factor_mat_solver_type": "mumps"},
        petsc_options_prefix="helm_"
    )
    u_sol = problem.solve()

    # Sample on grid
    xs = np.linspace(bbox[0], bbox[1], nx_out)
    ys = np.linspace(bbox[2], bbox[3], ny_out)
    XX, YY = np.meshgrid(xs, ys)
    pts = np.c_[XX.ravel(), YY.ravel(), np.zeros(nx_out*ny_out)]

    tree = geometry.bb_tree(domain, domain.topology.dim)
    cc = geometry.compute_collisions_points(tree, pts)
    colliding = geometry.compute_colliding_cells(domain, cc, pts)

    vals = np.zeros(nx_out*ny_out)
    pts_on = []
    cells_on = []
    idxs = []
    for i in range(pts.shape[0]):
        links = colliding.links(i)
        if len(links) > 0:
            pts_on.append(pts[i])
            cells_on.append(links[0])
            idxs.append(i)
    if len(pts_on) > 0:
        res = u_sol.eval(np.array(pts_on), np.array(cells_on, dtype=np.int32))
        vals[idxs] = res.flatten()

    u_grid = vals.reshape(ny_out, nx_out)

    # Accuracy check
    u_exact_grid = np.sin(5*np.pi*XX)*np.sin(4*np.pi*YY)
    err = np.sqrt(np.mean((u_grid - u_exact_grid)**2))
    print(f"RMS error: {err:.4e}")

    return {
        "u": u_grid,
        "solver_info": {
            "mesh_resolution": N,
            "element_degree": degree,
            "ksp_type": "preonly",
            "pc_type": "lu",
            "rtol": 1e-12,
            "iterations": 1,
        }
    }


if __name__ == "__main__":
    import time
    spec = {"output": {"grid": {"nx": 64, "ny": 64, "bbox": [0, 1, 0, 1]}}}
    t0 = time.time()
    r = solve(spec)
    print(f"Time: {time.time()-t0:.3f}s, shape={r['u'].shape}")
