import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc


def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD

    # Parameters
    E = 1.0
    nu = 0.28
    mu = E / (2.0 * (1.0 + nu))
    lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))

    # Output grid
    grid = case_spec["output"]["grid"]
    nx_out = grid["nx"]
    ny_out = grid["ny"]
    bbox = grid["bbox"]

    # Mesh
    N = 128
    degree = 2
    msh = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
    gdim = msh.geometry.dim

    V = fem.functionspace(msh, ("Lagrange", degree, (gdim,)))

    # Manufactured solution
    x = ufl.SpatialCoordinate(msh)
    pi = ufl.pi
    u_exact = ufl.as_vector([
        ufl.sin(4 * pi * x[0]) * ufl.sin(3 * pi * x[1]),
        ufl.cos(3 * pi * x[0]) * ufl.sin(4 * pi * x[1]),
    ])

    def eps(w):
        return ufl.sym(ufl.grad(w))

    def sigma(w):
        return 2.0 * mu * eps(w) + lam * ufl.tr(eps(w)) * ufl.Identity(gdim)

    # f = -div(sigma(u_exact))
    f = -ufl.div(sigma(u_exact))

    u_tr = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    a = ufl.inner(sigma(u_tr), eps(v)) * ufl.dx
    L = ufl.inner(f, v) * ufl.dx

    # BCs
    fdim = msh.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(
        msh, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
    )
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)

    u_bc = fem.Function(V)
    u_bc.interpolate(fem.Expression(u_exact, V.element.interpolation_points))
    bc = fem.dirichletbc(u_bc, boundary_dofs)

    # Solve
    rtol = 1e-10
    petsc_options = {
        "ksp_type": "cg",
        "pc_type": "hypre",
        "pc_hypre_type": "boomeramg",
        "ksp_rtol": rtol,
        "ksp_atol": 1e-14,
    }
    problem = petsc.LinearProblem(
        a, L, bcs=[bc],
        petsc_options=petsc_options,
        petsc_options_prefix="elast_",
    )
    u_sol = problem.solve()
    iters = problem.solver.getIterationNumber()

    # Sample on grid
    xs = np.linspace(bbox[0], bbox[1], nx_out)
    ys = np.linspace(bbox[2], bbox[3], ny_out)
    XX, YY = np.meshgrid(xs, ys)
    pts = np.c_[XX.ravel(), YY.ravel(), np.zeros(nx_out * ny_out)]

    tree = geometry.bb_tree(msh, msh.topology.dim)
    cell_candidates = geometry.compute_collisions_points(tree, pts)
    colliding = geometry.compute_colliding_cells(msh, cell_candidates, pts)

    points_on_proc = []
    cells = []
    idx_map = []
    for i in range(pts.shape[0]):
        links = colliding.links(i)
        if len(links) > 0:
            points_on_proc.append(pts[i])
            cells.append(links[0])
            idx_map.append(i)

    u_vals = u_sol.eval(np.array(points_on_proc), np.array(cells, dtype=np.int32))
    magnitude = np.zeros(nx_out * ny_out)
    mags_local = np.linalg.norm(u_vals, axis=1)
    for k, i in enumerate(idx_map):
        magnitude[i] = mags_local[k]
    u_grid = magnitude.reshape(ny_out, nx_out)

    # Accuracy check against exact magnitude
    u1_ex = np.sin(4 * np.pi * XX) * np.sin(3 * np.pi * YY)
    u2_ex = np.cos(3 * np.pi * XX) * np.sin(4 * np.pi * YY)
    mag_ex = np.sqrt(u1_ex**2 + u2_ex**2)
    err = np.sqrt(np.mean((u_grid - mag_ex)**2))
    print(f"[solver] mesh N={N}, degree={degree}, iters={iters}, RMS err={err:.3e}")

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
    case_spec = {
        "output": {"grid": {"nx": 64, "ny": 64, "bbox": [0.0, 1.0, 0.0, 1.0]}},
    }
    t0 = time.time()
    res = solve(case_spec)
    t1 = time.time()
    print(f"Wall time: {t1 - t0:.2f}s, shape={res['u'].shape}")
