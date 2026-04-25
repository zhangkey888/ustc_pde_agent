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

    N = 64
    degree = 2

    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", degree))

    x = ufl.SpatialCoordinate(domain)
    u_exact = ufl.sin(ufl.pi * x[0] * x[1])
    # f = -div(grad(u)) for kappa=1
    # grad u = (pi*y*cos(pi*xy), pi*x*cos(pi*xy))
    # div grad u = -pi^2 y^2 sin(pi*xy) - pi^2 x^2 sin(pi*xy) = -pi^2 (x^2+y^2) sin(pi*xy)
    # so f = pi^2 (x^2+y^2) sin(pi*xy)
    f = ufl.pi**2 * (x[0]**2 + x[1]**2) * ufl.sin(ufl.pi * x[0] * x[1])

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    a = ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = ufl.inner(f, v) * ufl.dx

    # BC
    u_bc = fem.Function(V)
    u_bc.interpolate(fem.Expression(u_exact, V.element.interpolation_points))
    fdim = domain.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
    )
    dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc = fem.dirichletbc(u_bc, dofs)

    rtol = 1e-10
    problem = petsc.LinearProblem(
        a, L, bcs=[bc],
        petsc_options={"ksp_type": "cg", "pc_type": "hypre", "ksp_rtol": rtol},
        petsc_options_prefix="poisson_"
    )
    u_sol = problem.solve()
    its = problem.solver.getIterationNumber()

    # Sample on grid
    xs = np.linspace(bbox[0], bbox[1], nx_out)
    ys = np.linspace(bbox[2], bbox[3], ny_out)
    XX, YY = np.meshgrid(xs, ys)
    pts = np.c_[XX.ravel(), YY.ravel(), np.zeros(nx_out * ny_out)]

    tree = geometry.bb_tree(domain, domain.topology.dim)
    cand = geometry.compute_collisions_points(tree, pts)
    coll = geometry.compute_colliding_cells(domain, cand, pts)

    cells = []
    points_used = []
    idx = []
    for i in range(pts.shape[0]):
        links = coll.links(i)
        if len(links) > 0:
            cells.append(links[0])
            points_used.append(pts[i])
            idx.append(i)
    vals = u_sol.eval(np.array(points_used), np.array(cells, dtype=np.int32)).flatten()
    out = np.zeros(nx_out * ny_out)
    out[idx] = vals
    u_grid = out.reshape(ny_out, nx_out)

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
    spec = {"output": {"grid": {"nx": 64, "ny": 64, "bbox": [0, 1, 0, 1]}}}
    t0 = time.time()
    res = solve(spec)
    t1 = time.time()
    print(f"Time: {t1-t0:.3f}s, iters: {res['solver_info']['iterations']}")
    # error vs exact
    xs = np.linspace(0, 1, 64)
    ys = np.linspace(0, 1, 64)
    XX, YY = np.meshgrid(xs, ys)
    u_ex = np.sin(np.pi * XX * YY)
    err = np.sqrt(np.mean((res["u"] - u_ex)**2))
    print(f"RMS error: {err:.3e}")
    print(f"Max error: {np.max(np.abs(res['u']-u_ex)):.3e}")
