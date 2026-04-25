import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc


def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD
    eps_val = 0.3
    beta_val = np.array([0.5, 0.3])

    N = 24
    degree = 2
    msh = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)

    V = fem.functionspace(msh, ("Lagrange", degree))

    x = ufl.SpatialCoordinate(msh)
    u_exact = x[0]*(1-x[0])*x[1]*(1-x[1])
    # f = -eps*laplacian(u) + beta . grad(u)
    lap = ufl.div(ufl.grad(u_exact))
    grad_u = ufl.grad(u_exact)
    beta = ufl.as_vector([beta_val[0], beta_val[1]])
    f = -eps_val * lap + ufl.dot(beta, grad_u)

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    a = eps_val * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx \
        + ufl.inner(ufl.dot(beta, ufl.grad(u)), v) * ufl.dx
    L = ufl.inner(f, v) * ufl.dx

    fdim = msh.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(
        msh, fdim, lambda xx: np.ones(xx.shape[1], dtype=bool)
    )
    u_bc = fem.Function(V)
    u_bc.interpolate(fem.Expression(u_exact, V.element.interpolation_points))
    bc = fem.dirichletbc(u_bc, fem.locate_dofs_topological(V, fdim, boundary_facets))

    problem = petsc.LinearProblem(
        a, L, bcs=[bc],
        petsc_options={"ksp_type": "gmres", "pc_type": "ilu", "ksp_rtol": 1e-10},
        petsc_options_prefix="cd_",
    )
    u_sol = problem.solve()
    iters = problem.solver.getIterationNumber()

    # Sample onto grid
    grid = case_spec["output"]["grid"]
    nx = grid["nx"]; ny = grid["ny"]
    bbox = grid["bbox"]
    xs = np.linspace(bbox[0], bbox[1], nx)
    ys = np.linspace(bbox[2], bbox[3], ny)
    XX, YY = np.meshgrid(xs, ys)
    pts = np.c_[XX.ravel(), YY.ravel(), np.zeros(nx*ny)]

    tree = geometry.bb_tree(msh, msh.topology.dim)
    cand = geometry.compute_collisions_points(tree, pts)
    coll = geometry.compute_colliding_cells(msh, cand, pts)

    cells = []
    valid = []
    for i in range(pts.shape[0]):
        links = coll.links(i)
        if len(links) > 0:
            cells.append(links[0])
            valid.append(i)

    u_vals = np.zeros(pts.shape[0])
    if len(valid) > 0:
        vals = u_sol.eval(pts[valid], np.array(cells, dtype=np.int32))
        u_vals[valid] = vals.flatten()
    u_grid = u_vals.reshape(ny, nx)

    return {
        "u": u_grid,
        "solver_info": {
            "mesh_resolution": N,
            "element_degree": degree,
            "ksp_type": "gmres",
            "pc_type": "ilu",
            "rtol": 1e-10,
            "iterations": int(iters),
        },
    }


if __name__ == "__main__":
    import time
    spec = {"output": {"grid": {"nx": 65, "ny": 65, "bbox": [0, 1, 0, 1]}}}
    t0 = time.time()
    res = solve(spec)
    t1 = time.time()
    print(f"Time: {t1-t0:.3f}s")
    u = res["u"]
    nx, ny = 65, 65
    xs = np.linspace(0, 1, nx); ys = np.linspace(0, 1, ny)
    XX, YY = np.meshgrid(xs, ys)
    u_ex = XX*(1-XX)*YY*(1-YY)
    err = np.sqrt(np.mean((u - u_ex)**2))
    print(f"RMS error: {err:.3e}")
    print(f"Max error: {np.max(np.abs(u - u_ex)):.3e}")
