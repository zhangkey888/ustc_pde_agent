import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc


def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD
    eps_val = 0.2
    beta_val = np.array([0.8, 0.3])

    N = 96
    degree = 2

    msh = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(msh, ("Lagrange", degree))

    x = ufl.SpatialCoordinate(msh)
    u_exact = ufl.cos(ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])
    # Compute f = -eps * lap(u) + beta . grad(u)
    grad_ue = ufl.as_vector([-ufl.pi * ufl.sin(ufl.pi*x[0]) * ufl.sin(ufl.pi*x[1]),
                              ufl.pi * ufl.cos(ufl.pi*x[0]) * ufl.cos(ufl.pi*x[1])])
    lap_ue = -2.0 * (ufl.pi**2) * ufl.cos(ufl.pi*x[0]) * ufl.sin(ufl.pi*x[1])
    beta = fem.Constant(msh, PETSc.ScalarType(beta_val))
    eps_c = fem.Constant(msh, PETSc.ScalarType(eps_val))
    f = -eps_c * lap_ue + ufl.dot(beta, grad_ue)

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    a = eps_c * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx \
        + ufl.inner(ufl.dot(beta, ufl.grad(u)), v) * ufl.dx
    L = ufl.inner(f, v) * ufl.dx

    # Dirichlet BC from exact solution
    fdim = msh.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(
        msh, fdim, lambda x: np.ones(x.shape[1], dtype=bool))
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    u_bc = fem.Function(V)
    u_bc.interpolate(fem.Expression(u_exact, V.element.interpolation_points))
    bc = fem.dirichletbc(u_bc, boundary_dofs)

    problem = petsc.LinearProblem(
        a, L, bcs=[bc],
        petsc_options={"ksp_type": "gmres", "pc_type": "hypre", "ksp_rtol": 1e-10},
        petsc_options_prefix="cd_"
    )
    u_sol = problem.solve()
    its = problem.solver.getIterationNumber()

    # Sample onto grid
    grid = case_spec["output"]["grid"]
    nx, ny = grid["nx"], grid["ny"]
    bbox = grid["bbox"]
    xs = np.linspace(bbox[0], bbox[1], nx)
    ys = np.linspace(bbox[2], bbox[3], ny)
    XX, YY = np.meshgrid(xs, ys)
    pts = np.c_[XX.ravel(), YY.ravel(), np.zeros(nx * ny)]

    tree = geometry.bb_tree(msh, msh.topology.dim)
    cand = geometry.compute_collisions_points(tree, pts)
    coll = geometry.compute_colliding_cells(msh, cand, pts)

    points_on_proc = []
    cells_on_proc = []
    idx = []
    for i in range(pts.shape[0]):
        links = coll.links(i)
        if len(links) > 0:
            points_on_proc.append(pts[i])
            cells_on_proc.append(links[0])
            idx.append(i)

    u_grid = np.full(nx * ny, np.nan)
    if points_on_proc:
        vals = u_sol.eval(np.array(points_on_proc),
                          np.array(cells_on_proc, dtype=np.int32))
        u_grid[idx] = vals.flatten()
    u_grid = u_grid.reshape(ny, nx)

    return {
        "u": u_grid,
        "solver_info": {
            "mesh_resolution": N,
            "element_degree": degree,
            "ksp_type": "gmres",
            "pc_type": "hypre",
            "rtol": 1e-10,
            "iterations": int(its),
        }
    }


if __name__ == "__main__":
    import time
    spec = {"output": {"grid": {"nx": 64, "ny": 64, "bbox": [0, 1, 0, 1]}}}
    t0 = time.time()
    res = solve(spec)
    dt = time.time() - t0
    u = res["u"]
    xs = np.linspace(0, 1, 64); ys = np.linspace(0, 1, 64)
    XX, YY = np.meshgrid(xs, ys)
    uex = np.cos(np.pi*XX) * np.sin(np.pi*YY)
    err = np.sqrt(np.mean((u - uex)**2))
    print(f"time={dt:.3f}s  rmse={err:.3e}  iters={res['solver_info']['iterations']}")
