import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc


def solve(case_spec: dict = None) -> dict:
    """
    Solve Poisson equation: -div(kappa * grad(u)) = f on unit square
    with u = 0 on boundary (homogeneous Dirichlet).
    """
    comm = MPI.COMM_WORLD

    # Parse case_spec if provided
    if case_spec is None:
        case_spec = {}

    # Default parameters
    kappa_val = 1.0
    f_val = 1.0

    # Try to extract from case_spec
    oracle_cfg = case_spec.get('oracle_config', {})
    pde_spec = oracle_cfg.get('pde', case_spec.get('pde', {}))
    coeffs = pde_spec.get('coefficients', {})
    if 'kappa' in coeffs:
        k = coeffs['kappa']
        if isinstance(k, dict):
            kappa_val = float(k.get('value', 1.0))
        else:
            kappa_val = float(k)

    source = pde_spec.get('source_term', pde_spec.get('source', '1.0'))
    if isinstance(source, (int, float)):
        f_val = float(source)
    elif isinstance(source, dict):
        f_val = float(source.get('value', 1.0))
    elif isinstance(source, str):
        f_val = float(source)

    # BC value
    bc_val = 0.0
    bc_spec = oracle_cfg.get('bc', {}).get('dirichlet', {})
    if bc_spec:
        bc_val = float(bc_spec.get('value', 0.0))

    # Output grid size
    output_cfg = oracle_cfg.get('output', {}).get('grid', {})
    nx_out = output_cfg.get('nx', 50)
    ny_out = output_cfg.get('ny', 50)

    # Adaptive mesh refinement with convergence check
    resolutions = [32, 64, 128]
    element_degree = 2  # Degree 2 for good accuracy on quads
    prev_norm = None
    u_grid = None
    final_info = {}
    total_iterations = 0

    for N in resolutions:
        # Create quadrilateral mesh on unit square
        domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.quadrilateral)

        # Function space - Lagrange on quads
        V = fem.functionspace(domain, ("Lagrange", element_degree))

        # Define variational problem
        u = ufl.TrialFunction(V)
        v = ufl.TestFunction(V)

        kappa = fem.Constant(domain, PETSc.ScalarType(kappa_val))
        f = fem.Constant(domain, PETSc.ScalarType(f_val))

        a = kappa * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
        L = f * v * ufl.dx

        # Boundary conditions: u = bc_val on all boundaries
        tdim = domain.topology.dim
        fdim = tdim - 1

        boundary_facets = mesh.locate_entities_boundary(
            domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
        )
        boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
        bc = fem.dirichletbc(PETSc.ScalarType(bc_val), boundary_dofs, V)

        # Solve with iterative solver
        ksp_type = "cg"
        pc_type = "hypre"

        try:
            problem = petsc.LinearProblem(
                a, L, bcs=[bc],
                petsc_options={
                    "ksp_type": ksp_type,
                    "pc_type": pc_type,
                    "ksp_rtol": "1e-10",
                    "ksp_atol": "1e-12",
                    "ksp_max_it": "2000",
                    "ksp_converged_reason": "",
                },
                petsc_options_prefix="poisson_"
            )
            u_sol = problem.solve()
            # Get iteration count
            ksp = problem.solver
            total_iterations += ksp.getIterationNumber()
        except Exception:
            # Fallback to direct solver
            ksp_type = "preonly"
            pc_type = "lu"
            problem = petsc.LinearProblem(
                a, L, bcs=[bc],
                petsc_options={
                    "ksp_type": ksp_type,
                    "pc_type": pc_type,
                },
                petsc_options_prefix="poisson_"
            )
            u_sol = problem.solve()
            total_iterations += 1

        # Evaluate on output grid
        bbox = output_cfg.get('bbox', [0, 1, 0, 1])
        x_coords = np.linspace(bbox[0], bbox[1], nx_out)
        y_coords = np.linspace(bbox[2], bbox[3], ny_out)
        xx, yy = np.meshgrid(x_coords, y_coords, indexing='ij')
        points_2d = np.column_stack([xx.ravel(), yy.ravel()])

        # dolfinx needs 3D points
        points_3d = np.zeros((points_2d.shape[0], 3))
        points_3d[:, :2] = points_2d

        # Build bounding box tree and evaluate
        bb_tree = geometry.bb_tree(domain, domain.topology.dim)
        cell_candidates = geometry.compute_collisions_points(bb_tree, points_3d)
        colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points_3d)

        u_values = np.full(points_3d.shape[0], np.nan)
        points_on_proc = []
        cells_on_proc = []
        eval_map = []

        for i in range(points_3d.shape[0]):
            links = colliding_cells.links(i)
            if len(links) > 0:
                points_on_proc.append(points_3d[i])
                cells_on_proc.append(links[0])
                eval_map.append(i)

        if len(points_on_proc) > 0:
            vals = u_sol.eval(
                np.array(points_on_proc),
                np.array(cells_on_proc, dtype=np.int32)
            )
            u_values[eval_map] = vals.flatten()

        u_grid = u_values.reshape((nx_out, ny_out))

        # Compute L2 norm for convergence check
        valid = ~np.isnan(u_grid)
        current_norm = np.sqrt(np.sum(u_grid[valid]**2) / np.count_nonzero(valid))

        final_info = {
            "mesh_resolution": N,
            "element_degree": element_degree,
            "ksp_type": ksp_type,
            "pc_type": pc_type,
            "rtol": 1e-10,
            "iterations": total_iterations,
        }

        if prev_norm is not None:
            rel_change = abs(current_norm - prev_norm) / (current_norm + 1e-15)
            if rel_change < 1e-4:
                # Converged
                break

        prev_norm = current_norm

    return {
        "u": u_grid,
        "solver_info": final_info,
    }


if __name__ == "__main__":
    import time
    t0 = time.time()
    result = solve()
    elapsed = time.time() - t0
    u_grid = result["u"]
    print(f"Solution shape: {u_grid.shape}")
    print(f"Solution min: {np.nanmin(u_grid):.8f}, max: {np.nanmax(u_grid):.8f}")
    print(f"NaN count: {np.isnan(u_grid).sum()}")
    print(f"Wall time: {elapsed:.3f}s")
    print(f"Solver info: {result['solver_info']}")
    print(f"Center value (25,25): {u_grid[25, 25]:.8f}")
