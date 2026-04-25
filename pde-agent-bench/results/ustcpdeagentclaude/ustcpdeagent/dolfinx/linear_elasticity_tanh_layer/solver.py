import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc


def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD

    # Grid spec
    grid = case_spec["output"]["grid"]
    nx_out = grid["nx"]
    ny_out = grid["ny"]
    bbox = grid["bbox"]

    # Mesh
    N = 96
    degree = 2
    msh = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
    gdim = msh.geometry.dim

    V = fem.functionspace(msh, ("Lagrange", degree, (gdim,)))

    # Material
    E, nu = 1.0, 0.3
    mu = E / (2.0 * (1.0 + nu))
    lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))

    x = ufl.SpatialCoordinate(msh)

    # Exact solution
    u_ex = ufl.as_vector([
        ufl.tanh(6.0 * (x[1] - 0.5)) * ufl.sin(ufl.pi * x[0]),
        0.1 * ufl.sin(2.0 * ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1]),
    ])

    def eps(u):
        return ufl.sym(ufl.grad(u))

    def sigma(u):
        return 2.0 * mu * eps(u) + lam * ufl.tr(eps(u)) * ufl.Identity(gdim)

    # f = -div(sigma(u_ex))
    sig_ex = sigma(u_ex)
    f = -ufl.as_vector([
        ufl.div(sig_ex[0, :]),
        ufl.div(sig_ex[1, :]),
    ])

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    a = ufl.inner(sigma(u), eps(v)) * ufl.dx
    L = ufl.inner(f, v) * ufl.dx

    # BC
    fdim = msh.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(
        msh, fdim, lambda X: np.ones(X.shape[1], dtype=bool)
    )
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)

    u_bc = fem.Function(V)
    u_bc.interpolate(fem.Expression(u_ex, V.element.interpolation_points))
    bc = fem.dirichletbc(u_bc, boundary_dofs)

    rtol = 1e-10
    petsc_opts = {
        "ksp_type": "cg",
        "pc_type": "hypre",
        "pc_hypre_type": "boomeramg",
        "ksp_rtol": rtol,
    }
    problem = petsc.LinearProblem(
        a, L, bcs=[bc],
        petsc_options=petsc_opts,
        petsc_options_prefix="elast_",
    )
    u_sol = problem.solve()
    iterations = problem.solver.getIterationNumber()

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
    mag = np.linalg.norm(u_vals, axis=1)
    magnitude = np.zeros(nx_out * ny_out)
    magnitude[idx_map] = mag
    magnitude = magnitude.reshape(ny_out, nx_out)

    return {
        "u": magnitude,
        "solver_info": {
            "mesh_resolution": N,
            "element_degree": degree,
            "ksp_type": "cg",
            "pc_type": "hypre",
            "rtol": rtol,
            "iterations": int(iterations),
        },
    }


if __name__ == "__main__":
    import time
    case = {"output": {"grid": {"nx": 128, "ny": 128, "bbox": [0, 1, 0, 1]}}}
    t0 = time.time()
    res = solve(case)
    t1 = time.time()
    print(f"Time: {t1-t0:.2f}s")
    print(f"Output shape: {res['u'].shape}")
    print(f"Iterations: {res['solver_info']['iterations']}")

    # Accuracy check against analytical
    nx, ny = 128, 128
    xs = np.linspace(0, 1, nx)
    ys = np.linspace(0, 1, ny)
    XX, YY = np.meshgrid(xs, ys)
    u1 = np.tanh(6*(YY-0.5)) * np.sin(np.pi*XX)
    u2 = 0.1 * np.sin(2*np.pi*XX) * np.sin(np.pi*YY)
    mag_exact = np.sqrt(u1**2 + u2**2)
    err = np.sqrt(np.mean((res['u'] - mag_exact)**2))
    print(f"RMS error: {err:.3e}")
    max_err = np.max(np.abs(res['u'] - mag_exact))
    print(f"Max error: {max_err:.3e}")
