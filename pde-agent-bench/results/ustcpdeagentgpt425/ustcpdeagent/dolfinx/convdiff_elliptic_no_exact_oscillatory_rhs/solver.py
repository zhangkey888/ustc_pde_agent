import time
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl


def solve(case_spec: dict) -> dict:
    """
    ```DIAGNOSIS
    equation_type: convection_diffusion
    spatial_dim: 2
    domain_geometry: rectangle
    unknowns: scalar
    coupling: none
    linearity: linear
    time_dependence: steady
    stiffness: stiff
    dominant_physics: mixed
    peclet_or_reynolds: high
    solution_regularity: smooth
    bc_type: all_dirichlet
    special_notes: variable_coeff
    ```

    ```METHOD
    spatial_method: fem
    element_or_basis: Lagrange_P1
    stabilization: supg
    time_method: none
    nonlinear_solver: none
    linear_solver: gmres
    preconditioner: ilu
    special_treatment: none
    pde_skill: convection_diffusion
    ```
    """
    comm = MPI.COMM_WORLD
    rank = comm.rank
    scalar_type = PETSc.ScalarType

    pde = case_spec.get("pde", {})
    coeffs = pde.get("coefficients", {})
    eps = float(coeffs.get("epsilon", coeffs.get("diffusion", 0.05)))
    beta_in = coeffs.get("beta", coeffs.get("velocity", [3.0, 3.0]))
    beta_vec = np.array(beta_in, dtype=np.float64)

    output = case_spec["output"]["grid"]
    nx = int(output["nx"])
    ny = int(output["ny"])
    bbox = output["bbox"]
    xmin, xmax, ymin, ymax = map(float, bbox)

    max_wall = 46.190
    t0 = time.perf_counter()

    def rhs_expr(x):
        return np.sin(6.0 * np.pi * x[0]) * np.sin(5.0 * np.pi * x[1])

    def boundary_marker(x):
        return (
            np.isclose(x[0], 0.0)
            | np.isclose(x[0], 1.0)
            | np.isclose(x[1], 0.0)
            | np.isclose(x[1], 1.0)
        )

    def sample_on_grid(domain, uh):
        xs = np.linspace(xmin, xmax, nx)
        ys = np.linspace(ymin, ymax, ny)
        xx, yy = np.meshgrid(xs, ys, indexing="xy")
        pts2 = np.column_stack([xx.ravel(), yy.ravel(), np.zeros(nx * ny, dtype=np.float64)])

        tree = geometry.bb_tree(domain, domain.topology.dim)
        candidates = geometry.compute_collisions_points(tree, pts2)
        colliding = geometry.compute_colliding_cells(domain, candidates, pts2)

        vals_local = np.full(nx * ny, np.nan, dtype=np.float64)
        points_on_proc = []
        cells_on_proc = []
        eval_map = []
        for i in range(nx * ny):
            links = colliding.links(i)
            if len(links) > 0:
                points_on_proc.append(pts2[i])
                cells_on_proc.append(links[0])
                eval_map.append(i)

        if points_on_proc:
            values = uh.eval(np.array(points_on_proc, dtype=np.float64), np.array(cells_on_proc, dtype=np.int32))
            vals_local[np.array(eval_map, dtype=np.int32)] = np.asarray(values, dtype=np.float64).reshape(-1)

        vals_global = comm.allreduce(vals_local, op=MPI.SUM)
        return vals_global.reshape((ny, nx))

    def solve_level(ncells, degree=1):
        domain = mesh.create_unit_square(comm, ncells, ncells, cell_type=mesh.CellType.triangle)
        V = fem.functionspace(domain, ("Lagrange", degree))

        u = ufl.TrialFunction(V)
        v = ufl.TestFunction(V)
        x = ufl.SpatialCoordinate(domain)

        f_expr = ufl.sin(6.0 * ufl.pi * x[0]) * ufl.sin(5.0 * ufl.pi * x[1])
        beta = fem.Constant(domain, scalar_type(beta_vec))
        eps_c = fem.Constant(domain, scalar_type(eps))

        h = ufl.CellDiameter(domain)
        beta_norm = float(np.linalg.norm(beta_vec))
        if beta_norm > 0.0:
            Pe = beta_norm * (1.0 / ncells) / (2.0 * eps)
            zeta = 1.0 / np.tanh(Pe) - 1.0 / Pe if Pe > 1.0e-10 else 0.0
            tau_val = (1.0 / beta_norm) * (1.0 / ncells) * 0.5 * zeta
        else:
            tau_val = 0.0
        tau = fem.Constant(domain, scalar_type(tau_val))

        a_std = eps_c * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx + ufl.inner(beta, ufl.grad(u)) * v * ufl.dx
        L_std = f_expr * v * ufl.dx

        residual_u = -eps_c * ufl.div(ufl.grad(u)) + ufl.inner(beta, ufl.grad(u))
        residual_rhs = f_expr
        a_supg = tau * ufl.inner(beta, ufl.grad(v)) * residual_u * ufl.dx
        L_supg = tau * ufl.inner(beta, ufl.grad(v)) * residual_rhs * ufl.dx

        a = a_std + a_supg
        L = L_std + L_supg

        fdim = domain.topology.dim - 1
        facets = mesh.locate_entities_boundary(domain, fdim, boundary_marker)
        dofs = fem.locate_dofs_topological(V, fdim, facets)
        bc = fem.dirichletbc(scalar_type(0.0), dofs, V)

        problem = petsc.LinearProblem(
            a,
            L,
            bcs=[bc],
            petsc_options_prefix=f"cd_{ncells}_",
            petsc_options={
                "ksp_type": "gmres",
                "pc_type": "ilu",
                "ksp_rtol": 1.0e-9,
                "ksp_atol": 1.0e-12,
                "ksp_max_it": 5000,
            },
        )
        uh = problem.solve()
        uh.x.scatter_forward()

        ksp = problem.solver
        its = int(ksp.getIterationNumber())
        return domain, uh, its, tau_val

    candidate_levels = [128, 192, 256, 320, 384, 448]
    selected = candidate_levels[0]
    all_results = []
    last_grid = None
    last_domain = None
    last_uh = None
    total_iterations = 0
    estimated_error = None
    previous_grid = None

    for ncells in candidate_levels:
        domain, uh, its, tau_val = solve_level(ncells, degree=1)
        total_iterations += its
        grid = sample_on_grid(domain, uh)
        all_results.append((ncells, domain, uh, its, tau_val, grid))

        if previous_grid is not None:
            estimated_error = float(np.linalg.norm(grid - previous_grid) / np.sqrt(grid.size))
        previous_grid = grid
        last_grid = grid
        last_domain = domain
        last_uh = uh
        selected = ncells

        elapsed = time.perf_counter() - t0
        remaining = max_wall - elapsed
        if remaining < 10.0:
            break

    if len(all_results) >= 2:
        estimated_error = float(np.linalg.norm(all_results[-1][5] - all_results[-2][5]) / np.sqrt(all_results[-1][5].size))
    else:
        estimated_error = np.nan

    solver_info = {
        "mesh_resolution": int(selected),
        "element_degree": 1,
        "ksp_type": "gmres",
        "pc_type": "ilu",
        "rtol": 1.0e-9,
        "iterations": int(total_iterations),
        "supg_tau": float(all_results[-1][4]) if all_results else 0.0,
        "mesh_convergence_estimate": None if np.isnan(estimated_error) else float(estimated_error),
        "wall_time_sec": float(time.perf_counter() - t0),
    }

    return {"u": last_grid, "solver_info": solver_info}


if __name__ == "__main__":
    case_spec = {
        "pde": {
            "coefficients": {
                "epsilon": 0.05,
                "beta": [3.0, 3.0],
            }
        },
        "output": {
            "grid": {
                "nx": 101,
                "ny": 101,
                "bbox": [0.0, 1.0, 0.0, 1.0],
            }
        },
    }
    out = solve(case_spec)
    if MPI.COMM_WORLD.rank == 0:
        print(out["u"].shape)
        print(out["solver_info"])
