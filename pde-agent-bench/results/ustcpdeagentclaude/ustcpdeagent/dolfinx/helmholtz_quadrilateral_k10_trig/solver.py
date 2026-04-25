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

    k_val = 10.0
    # Try to read k from case_spec if present
    pde = case_spec.get("pde", {})
    if "k" in pde:
        k_val = float(pde["k"])
    elif "wavenumber" in pde:
        k_val = float(pde["wavenumber"])

    # Mesh: use quadrilaterals, P3 elements for accuracy
    N = 64
    degree = 3

    domain = mesh.create_rectangle(
        comm,
        [np.array([bbox[0], bbox[2]]), np.array([bbox[1], bbox[3]])],
        [N, N],
        cell_type=mesh.CellType.quadrilateral,
    )

    V = fem.functionspace(domain, ("Lagrange", degree))

    x = ufl.SpatialCoordinate(domain)
    u_exact = ufl.sin(2 * ufl.pi * x[0]) * ufl.cos(3 * ufl.pi * x[1])

    # -laplacian(u) = (4pi^2 + 9pi^2) * u = 13 pi^2 u
    # f = -laplace(u) - k^2 u = 13 pi^2 u - k^2 u = (13 pi^2 - k^2) u
    k = fem.Constant(domain, PETSc.ScalarType(k_val))
    f_expr = (13.0 * ufl.pi**2 - k_val**2) * u_exact

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    a = ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx - (k * k) * ufl.inner(u, v) * ufl.dx
    L = ufl.inner(f_expr, v) * ufl.dx

    # BC
    fdim = domain.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda xx: np.ones(xx.shape[1], dtype=bool)
    )
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)

    u_bc = fem.Function(V)
    u_bc.interpolate(
        fem.Expression(u_exact, V.element.interpolation_points)
    )
    bc = fem.dirichletbc(u_bc, boundary_dofs)

    problem = petsc.LinearProblem(
        a, L, bcs=[bc],
        petsc_options={"ksp_type": "preonly", "pc_type": "lu",
                       "pc_factor_mat_solver_type": "mumps"},
        petsc_options_prefix="helm_",
    )
    u_sol = problem.solve()

    ksp = problem.solver
    iters = ksp.getIterationNumber()

    # Sample onto grid
    xs = np.linspace(bbox[0], bbox[1], nx_out)
    ys = np.linspace(bbox[2], bbox[3], ny_out)
    XX, YY = np.meshgrid(xs, ys)
    pts = np.c_[XX.ravel(), YY.ravel(), np.zeros(nx_out * ny_out)]

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
        vals = u_sol.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u_values[eval_map] = vals.flatten()

    u_grid = u_values.reshape(ny_out, nx_out)

    # Verification
    u_ex_grid = np.sin(2 * np.pi * XX) * np.cos(3 * np.pi * YY)
    err = np.sqrt(np.mean((u_grid - u_ex_grid) ** 2))
    print(f"[solver] N={N}, degree={degree}, k={k_val}, RMSE={err:.3e}")

    return {
        "u": u_grid,
        "solver_info": {
            "mesh_resolution": N,
            "element_degree": degree,
            "ksp_type": "preonly",
            "pc_type": "lu",
            "rtol": 1e-12,
            "iterations": int(iters),
        },
    }


if __name__ == "__main__":
    import time
    case = {
        "pde": {"k": 10.0},
        "output": {
            "grid": {"nx": 128, "ny": 128, "bbox": [0.0, 1.0, 0.0, 1.0]}
        }
    }
    t0 = time.time()
    res = solve(case)
    t1 = time.time()
    print(f"Time: {t1-t0:.2f}s, shape={res['u'].shape}")
