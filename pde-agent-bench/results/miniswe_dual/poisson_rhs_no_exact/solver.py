import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc

ScalarType = PETSc.ScalarType


def solve(case_spec: dict = None) -> dict:
    if case_spec is None:
        case_spec = {}

    pde_spec = case_spec.get("pde", {})
    coeffs = pde_spec.get("coefficients", {})
    kappa_val = coeffs.get("kappa", 0.5)

    output = case_spec.get("output", {})
    nx_out = output.get("nx", 50)
    ny_out = output.get("ny", 50)

    # Solve at a good resolution directly - P2 on 64x64 is very accurate for this problem
    N = 64
    element_degree = 2
    ksp_type = "cg"
    pc_type = "hypre"
    rtol = 1e-10

    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", element_degree))

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    x = ufl.SpatialCoordinate(domain)

    kappa = fem.Constant(domain, ScalarType(kappa_val))
    f_expr = ufl.sin(3 * ufl.pi * x[0]) * ufl.sin(2 * ufl.pi * x[1])

    a = kappa * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = f_expr * v * ufl.dx

    tdim = domain.topology.dim
    fdim = tdim - 1

    def boundary_all(x):
        return (np.isclose(x[0], 0.0) | np.isclose(x[0], 1.0) |
                np.isclose(x[1], 0.0) | np.isclose(x[1], 1.0))

    boundary_facets = mesh.locate_entities_boundary(domain, fdim, boundary_all)
    dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc = fem.dirichletbc(ScalarType(0.0), dofs, V)

    problem = petsc.LinearProblem(
        a, L, bcs=[bc],
        petsc_options={
            "ksp_type": ksp_type,
            "pc_type": pc_type,
            "ksp_rtol": str(rtol),
            "ksp_max_it": "1000",
        },
        petsc_options_prefix="poisson_"
    )
    u_sol = problem.solve()

    # Evaluate on output grid
    xs = np.linspace(0, 1, nx_out)
    ys = np.linspace(0, 1, ny_out)
    XX, YY = np.meshgrid(xs, ys, indexing='ij')

    points = np.zeros((nx_out * ny_out, 3))
    points[:, 0] = XX.flatten()
    points[:, 1] = YY.flatten()

    bb_tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, points)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points)

    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    for i in range(len(points)):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(points[i])
            cells_on_proc.append(links[0])
            eval_map.append(i)

    u_values = np.full(nx_out * ny_out, np.nan)
    if len(points_on_proc) > 0:
        pts_arr = np.array(points_on_proc)
        cells_arr = np.array(cells_on_proc, dtype=np.int32)
        vals = u_sol.eval(pts_arr, cells_arr)
        u_values[eval_map] = vals.flatten()

    u_grid = u_values.reshape((nx_out, ny_out))

    info = {
        "mesh_resolution": N,
        "element_degree": element_degree,
        "ksp_type": ksp_type,
        "pc_type": pc_type,
        "rtol": rtol,
        "iterations": 0,
    }

    return {"u": u_grid, "solver_info": info}


if __name__ == "__main__":
    import time
    t0 = time.time()
    result = solve()
    elapsed = time.time() - t0
    u = result["u"]
    print(f"Shape: {u.shape}, Max: {np.nanmax(u):.8f}, Min: {np.nanmin(u):.8f}, NaNs: {np.isnan(u).sum()}")
    print(f"Solver info: {result['solver_info']}")
    print(f"Wall time: {elapsed:.3f}s")
    # Compare with analytical
    A_exact = 2.0 / (13.0 * np.pi**2)
    print(f"Analytical max: {A_exact:.8f}")
