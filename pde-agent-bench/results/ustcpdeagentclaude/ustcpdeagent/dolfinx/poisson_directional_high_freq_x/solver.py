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

    u_exact = ufl.sin(8*ufl.pi*x[0]) * ufl.sin(ufl.pi*x[1])
    f_expr = -ufl.div(ufl.grad(u_exact))  # = (64+1)*pi^2 * u_exact

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    a = ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = ufl.inner(f_expr, v) * ufl.dx

    # BC from exact solution
    u_bc = fem.Function(V)
    u_bc.interpolate(fem.Expression(u_exact, V.element.interpolation_points))

    fdim = domain.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool))
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc = fem.dirichletbc(u_bc, boundary_dofs)

    rtol = 1e-10
    problem = petsc.LinearProblem(
        a, L, bcs=[bc],
        petsc_options={"ksp_type": "cg", "pc_type": "hypre", "ksp_rtol": rtol},
        petsc_options_prefix="poisson_"
    )
    u_sol = problem.solve()
    iters = problem.solver.getIterationNumber()

    # Sample on grid
    g = case_spec["output"]["grid"]
    nx = g["nx"]; ny = g["ny"]; bbox = g["bbox"]
    xs = np.linspace(bbox[0], bbox[1], nx)
    ys = np.linspace(bbox[2], bbox[3], ny)
    XX, YY = np.meshgrid(xs, ys)
    pts = np.c_[XX.ravel(), YY.ravel(), np.zeros(nx*ny)]

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

    u_grid = np.zeros(nx*ny)
    if points_on_proc:
        vals = u_sol.eval(np.array(points_on_proc),
                          np.array(cells_on_proc, dtype=np.int32))
        u_grid[eval_map] = vals.flatten()
    u_grid = u_grid.reshape(ny, nx)

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
    spec = {"output": {"grid": {"nx": 64, "ny": 64, "bbox": [0,1,0,1]}}}
    t0 = time.time()
    out = solve(spec)
    t1 = time.time()
    print(f"Time: {t1-t0:.3f}s")
    # Check error
    nx, ny = 64, 64
    xs = np.linspace(0,1,nx); ys = np.linspace(0,1,ny)
    XX, YY = np.meshgrid(xs, ys)
    u_ex = np.sin(8*np.pi*XX)*np.sin(np.pi*YY)
    err = np.sqrt(np.mean((out["u"] - u_ex)**2))
    print(f"L2 error: {err:.3e}")
    print(f"Max error: {np.max(np.abs(out['u']-u_ex)):.3e}")
