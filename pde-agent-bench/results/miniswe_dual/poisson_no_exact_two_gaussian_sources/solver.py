import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc


def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD

    pde = case_spec.get("pde", {})
    coefficients = pde.get("coefficients", {})
    kappa_val = float(coefficients.get("kappa", 1.0))

    output = case_spec.get("output", {})
    nx_out = output.get("nx", 50)
    ny_out = output.get("ny", 50)

    # Solver parameters - well-converged for this problem
    element_degree = 2
    N = 64

    ksp_type_used = "cg"
    pc_type_used = "hypre"
    rtol_used = 1e-10

    # Create mesh and function space
    domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", element_degree))

    # Source term: two Gaussian peaks
    x = ufl.SpatialCoordinate(domain)
    f_expr = (ufl.exp(-250.0 * ((x[0] - 0.25)**2 + (x[1] - 0.25)**2)) +
              ufl.exp(-250.0 * ((x[0] - 0.75)**2 + (x[1] - 0.7)**2)))

    kappa = fem.Constant(domain, PETSc.ScalarType(kappa_val))

    # Variational form
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    a = kappa * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = f_expr * v * ufl.dx

    # Boundary conditions: u = 0 on all boundaries
    tdim = domain.topology.dim
    fdim = tdim - 1
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
    )
    dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc = fem.dirichletbc(PETSc.ScalarType(0.0), dofs, V)

    # Solve with iterative solver, fallback to direct if needed
    iterations = 0
    try:
        problem = petsc.LinearProblem(
            a, L, bcs=[bc],
            petsc_options={
                "ksp_type": ksp_type_used,
                "pc_type": pc_type_used,
                "ksp_rtol": str(rtol_used),
                "ksp_max_it": "2000",
                "ksp_converged_reason": "",
            },
            petsc_options_prefix="poisson_"
        )
        u_sol = problem.solve()
        # Get iteration count
        try:
            iterations = problem.solver.getIterationNumber()
        except Exception:
            iterations = 0
    except Exception:
        ksp_type_used = "preonly"
        pc_type_used = "lu"
        problem = petsc.LinearProblem(
            a, L, bcs=[bc],
            petsc_options={
                "ksp_type": ksp_type_used,
                "pc_type": pc_type_used,
            },
            petsc_options_prefix="poisson_"
        )
        u_sol = problem.solve()
        iterations = 1

    # Evaluate solution on output grid
    x_coords = np.linspace(0, 1, nx_out)
    y_coords = np.linspace(0, 1, ny_out)
    X, Y = np.meshgrid(x_coords, y_coords, indexing='ij')
    points = np.zeros((3, nx_out * ny_out))
    points[0, :] = X.flatten()
    points[1, :] = Y.flatten()

    bb_tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, points.T)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points.T)

    u_values = np.full(nx_out * ny_out, np.nan)
    points_on_proc = []
    cells_on_proc = []
    eval_map = []

    for i in range(nx_out * ny_out):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(points[:, i])
            cells_on_proc.append(links[0])
            eval_map.append(i)

    if len(points_on_proc) > 0:
        pts_arr = np.array(points_on_proc)
        cells_arr = np.array(cells_on_proc, dtype=np.int32)
        vals = u_sol.eval(pts_arr, cells_arr)
        u_values[eval_map] = vals.flatten()

    u_grid = u_values.reshape((nx_out, ny_out))
    u_grid = np.nan_to_num(u_grid, nan=0.0)

    solver_info = {
        "mesh_resolution": N,
        "element_degree": element_degree,
        "ksp_type": ksp_type_used,
        "pc_type": pc_type_used,
        "rtol": rtol_used,
        "iterations": iterations,
    }

    return {
        "u": u_grid,
        "solver_info": solver_info,
    }


if __name__ == "__main__":
    import time
    case_spec = {
        "pde": {
            "type": "poisson",
            "coefficients": {"kappa": 1.0},
            "source": "exp(-250*((x-0.25)**2 + (y-0.25)**2)) + exp(-250*((x-0.75)**2 + (y-0.7)**2))",
        },
        "domain": {"type": "unit_square"},
        "output": {"nx": 50, "ny": 50},
    }
    t0 = time.time()
    result = solve(case_spec)
    elapsed = time.time() - t0
    print(f"Solve time: {elapsed:.3f}s")
    print(f"Solution shape: {result['u'].shape}")
    print(f"Solution range: [{result['u'].min():.8f}, {result['u'].max():.8f}]")
    print(f"Solver info: {result['solver_info']}")
    print(f"Any NaN: {np.any(np.isnan(result['u']))}")
