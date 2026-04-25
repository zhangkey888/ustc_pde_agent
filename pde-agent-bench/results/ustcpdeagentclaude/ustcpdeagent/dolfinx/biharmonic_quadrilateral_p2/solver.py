import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc


def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD

    # Output grid
    grid = case_spec["output"]["grid"]
    nx_out = grid["nx"]
    ny_out = grid["ny"]
    bbox = grid["bbox"]

    # Mesh
    N = 64
    degree = 3
    p0 = np.array([bbox[0], bbox[2]])
    p1 = np.array([bbox[1], bbox[3]])
    domain = mesh.create_rectangle(comm, [p0, p1], [N, N],
                                    cell_type=mesh.CellType.quadrilateral)

    V = fem.functionspace(domain, ("Lagrange", degree))

    x = ufl.SpatialCoordinate(domain)
    pi = np.pi
    u_exact_ufl = ufl.sin(2 * pi * x[0]) * ufl.sin(pi * x[1])
    sigma_exact_ufl = 5 * pi**2 * ufl.sin(2 * pi * x[0]) * ufl.sin(pi * x[1])
    f_ufl = 25 * pi**4 * ufl.sin(2 * pi * x[0]) * ufl.sin(pi * x[1])

    # Boundary
    fdim = domain.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda xx: np.ones(xx.shape[1], dtype=bool)
    )
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)

    # Solve 1: -Δσ = f, σ = σ_exact on boundary
    sigma_bc = fem.Function(V)
    sigma_bc.interpolate(
        fem.Expression(sigma_exact_ufl, V.element.interpolation_points)
    )
    bc_sigma = fem.dirichletbc(sigma_bc, boundary_dofs)

    sig = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    a1 = ufl.inner(ufl.grad(sig), ufl.grad(v)) * ufl.dx
    L1 = f_ufl * v * ufl.dx

    problem1 = petsc.LinearProblem(
        a1, L1, bcs=[bc_sigma],
        petsc_options={"ksp_type": "cg", "pc_type": "hypre", "ksp_rtol": 1e-12, "ksp_atol": 1e-14},
        petsc_options_prefix="biharm1_"
    )
    sigma_h = problem1.solve()
    iter1 = problem1.solver.getIterationNumber()

    # Solve 2: -Δu = σ, u = u_exact on boundary
    u_bc = fem.Function(V)
    u_bc.interpolate(
        fem.Expression(u_exact_ufl, V.element.interpolation_points)
    )
    bc_u = fem.dirichletbc(u_bc, boundary_dofs)

    u = ufl.TrialFunction(V)
    a2 = ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
    L2 = sigma_h * v * ufl.dx

    problem2 = petsc.LinearProblem(
        a2, L2, bcs=[bc_u],
        petsc_options={"ksp_type": "cg", "pc_type": "hypre", "ksp_rtol": 1e-12, "ksp_atol": 1e-14},
        petsc_options_prefix="biharm2_"
    )
    u_h = problem2.solve()
    iter2 = problem2.solver.getIterationNumber()

    # Sample on grid
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

    u_values = np.full(pts.shape[0], np.nan)
    if len(points_on_proc) > 0:
        vals = u_h.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u_values[eval_map] = vals.flatten()

    u_grid = u_values.reshape(ny_out, nx_out)

    return {
        "u": u_grid,
        "solver_info": {
            "mesh_resolution": N,
            "element_degree": degree,
            "ksp_type": "cg",
            "pc_type": "hypre",
            "rtol": 1e-12,
            "iterations": int(iter1 + iter2),
        }
    }


if __name__ == "__main__":
    import time
    case = {
        "output": {
            "grid": {"nx": 65, "ny": 65, "bbox": [0.0, 1.0, 0.0, 1.0]}
        }
    }
    t0 = time.time()
    res = solve(case)
    t1 = time.time()
    u = res["u"]
    ny, nx = u.shape
    xs = np.linspace(0, 1, nx)
    ys = np.linspace(0, 1, ny)
    XX, YY = np.meshgrid(xs, ys)
    u_ex = np.sin(2*np.pi*XX)*np.sin(np.pi*YY)
    err = np.sqrt(np.mean((u - u_ex)**2))
    print(f"Time: {t1-t0:.3f}s, RMSE: {err:.3e}, max err: {np.max(np.abs(u-u_ex)):.3e}")
    print(f"iters: {res['solver_info']['iterations']}")
