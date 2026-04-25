import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc


def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD
    grid = case_spec["output"]["grid"]
    nx_out = grid["nx"]
    ny_out = grid["ny"]
    bbox = grid["bbox"]

    N = 64
    degree = 2
    domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", degree))

    x = ufl.SpatialCoordinate(domain)
    pi = ufl.pi

    u_exact = ufl.sin(2*pi*x[0]) * ufl.sin(2*pi*x[1])
    kappa = 1 + 0.5 * ufl.sin(2*pi*x[0]) * ufl.sin(2*pi*x[1])
    f = -ufl.div(kappa * ufl.grad(u_exact))

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    a = kappa * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = f * v * ufl.dx

    u_bc = fem.Function(V)
    u_bc.interpolate(fem.Expression(u_exact, V.element.interpolation_points))

    fdim = domain.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda xx: np.ones(xx.shape[1], dtype=bool)
    )
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc = fem.dirichletbc(u_bc, boundary_dofs)

    problem = petsc.LinearProblem(
        a, L, bcs=[bc],
        petsc_options={"ksp_type": "cg", "pc_type": "hypre", "ksp_rtol": 1e-10},
        petsc_options_prefix="poisson_"
    )
    u_sol = problem.solve()
    its = problem.solver.getIterationNumber()

    # Sample on uniform grid
    xs = np.linspace(bbox[0], bbox[1], nx_out)
    ys = np.linspace(bbox[2], bbox[3], ny_out)
    XX, YY = np.meshgrid(xs, ys)
    pts = np.c_[XX.ravel(), YY.ravel(), np.zeros(nx_out*ny_out)]

    tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(tree, pts)
    colliding = geometry.compute_colliding_cells(domain, cell_candidates, pts)

    points_on_proc = []
    cells = []
    idx_map = []
    for i in range(pts.shape[0]):
        links = colliding.links(i)
        if len(links) > 0:
            points_on_proc.append(pts[i])
            cells.append(links[0])
            idx_map.append(i)

    u_grid = np.zeros(nx_out*ny_out)
    if points_on_proc:
        vals = u_sol.eval(np.array(points_on_proc), np.array(cells, dtype=np.int32))
        u_grid[idx_map] = vals.flatten()
    u_grid = u_grid.reshape(ny_out, nx_out)

    return {
        "u": u_grid,
        "solver_info": {
            "mesh_resolution": N,
            "element_degree": degree,
            "ksp_type": "cg",
            "pc_type": "hypre",
            "rtol": 1e-10,
            "iterations": int(its),
        }
    }


if __name__ == "__main__":
    import time
    case = {"output": {"grid": {"nx": 128, "ny": 128, "bbox": [0,1,0,1]}}}
    t0 = time.time()
    res = solve(case)
    t1 = time.time()
    print(f"Time: {t1-t0:.3f}s")
    u = res["u"]
    xs = np.linspace(0,1,128); ys = np.linspace(0,1,128)
    XX, YY = np.meshgrid(xs, ys)
    u_ex = np.sin(2*np.pi*XX)*np.sin(2*np.pi*YY)
    err = np.sqrt(np.mean((u-u_ex)**2))
    print(f"RMSE: {err:.3e}")
    print(f"Max err: {np.max(np.abs(u-u_ex)):.3e}")
    print(f"Iter: {res['solver_info']['iterations']}")
