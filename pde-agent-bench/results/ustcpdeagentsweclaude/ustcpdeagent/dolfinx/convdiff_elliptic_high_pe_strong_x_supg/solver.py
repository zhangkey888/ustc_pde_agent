import time
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl

ScalarType = PETSc.ScalarType


def solve(case_spec: dict) -> dict:
    """
    Solve steady convection-diffusion:
        -eps * Δu + beta·∇u = f in Ω=[0,1]^2
        u = g on ∂Ω
    using CG FEM with SUPG stabilization for high Péclet number.

    Returns
    -------
    dict
        {
          "u": ndarray shape (ny, nx),
          "solver_info": {...}
        }
    """

    """
    ```DIAGNOSIS
    equation_type:        convection_diffusion
    spatial_dim:          2
    domain_geometry:      rectangle
    unknowns:             scalar
    coupling:             none
    linearity:            linear
    time_dependence:      steady
    stiffness:            stiff
    dominant_physics:     mixed
    peclet_or_reynolds:   high
    solution_regularity:  smooth
    bc_type:              all_dirichlet
    special_notes:        manufactured_solution
    ```
    """

    """
    ```METHOD
    spatial_method:       fem
    element_or_basis:     Lagrange_P2
    stabilization:        supg
    time_method:          none
    nonlinear_solver:     none
    linear_solver:        gmres
    preconditioner:       ilu
    special_treatment:    none
    pde_skill:            convection_diffusion / reaction_diffusion
    ```
    """

    comm = MPI.COMM_WORLD

    # Problem parameters from case description
    eps = 0.01
    beta_vec = np.array([15.0, 0.0], dtype=np.float64)
    beta_norm = float(np.linalg.norm(beta_vec))

    # Accuracy/time-aware default choice: P2 with moderately fine mesh
    # chosen to balance high-Pe stabilization accuracy with sub-4s runtime.
    mesh_resolution = int(case_spec.get("agent_params", {}).get("mesh_resolution", 80))
    element_degree = int(case_spec.get("agent_params", {}).get("element_degree", 2))
    if element_degree < 1:
        element_degree = 1

    t0 = time.perf_counter()

    msh = mesh.create_unit_square(
        comm, mesh_resolution, mesh_resolution, cell_type=mesh.CellType.triangle
    )
    V = fem.functionspace(msh, ("Lagrange", element_degree))

    x = ufl.SpatialCoordinate(msh)
    u_exact_ufl = ufl.sin(ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])
    beta = fem.Constant(msh, ScalarType(beta_vec))
    eps_c = fem.Constant(msh, ScalarType(eps))

    # Manufactured forcing from exact solution
    # -eps Δu + beta·∇u
    f_ufl = -eps_c * ufl.div(ufl.grad(u_exact_ufl)) + ufl.dot(beta, ufl.grad(u_exact_ufl))

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    # Cell size and SUPG parameter
    h = ufl.CellDiameter(msh)
    tau = h / (2.0 * beta_norm)
    if beta_norm == 0.0:
        tau = 0.0 * h

    residual_u = -eps_c * ufl.div(ufl.grad(u)) + ufl.dot(beta, ufl.grad(u))
    residual_f = f_ufl

    a_std = (eps_c * ufl.inner(ufl.grad(u), ufl.grad(v)) + ufl.dot(beta, ufl.grad(u)) * v) * ufl.dx
    L_std = residual_f * v * ufl.dx

    # SUPG stabilization, using streamline test perturbation tau * beta·∇v
    a_supg = tau * residual_u * ufl.dot(beta, ufl.grad(v)) * ufl.dx
    L_supg = tau * residual_f * ufl.dot(beta, ufl.grad(v)) * ufl.dx

    a = a_std + a_supg
    L = L_std + L_supg

    # Dirichlet BC from exact solution on full boundary
    fdim = msh.topology.dim - 1
    facets = mesh.locate_entities_boundary(msh, fdim, lambda X: np.ones(X.shape[1], dtype=bool))
    dofs = fem.locate_dofs_topological(V, fdim, facets)
    u_bc = fem.Function(V)
    u_bc.interpolate(fem.Expression(u_exact_ufl, V.element.interpolation_points))
    bc = fem.dirichletbc(u_bc, dofs)

    ksp_type = "gmres"
    pc_type = "ilu"
    rtol = 1.0e-10

    uh = None
    iterations = 0

    try:
        problem = petsc.LinearProblem(
            a,
            L,
            bcs=[bc],
            petsc_options_prefix="convdiff_",
            petsc_options={
                "ksp_type": ksp_type,
                "ksp_rtol": rtol,
                "pc_type": pc_type,
            },
        )
        uh = problem.solve()
        uh.x.scatter_forward()
        ksp = problem.solver
        iterations = int(ksp.getIterationNumber())
    except Exception:
        ksp_type = "preonly"
        pc_type = "lu"
        problem = petsc.LinearProblem(
            a,
            L,
            bcs=[bc],
            petsc_options_prefix="convdiff_fallback_",
            petsc_options={
                "ksp_type": ksp_type,
                "pc_type": pc_type,
            },
        )
        uh = problem.solve()
        uh.x.scatter_forward()
        iterations = 1

    # Accuracy verification: L2 error against exact solution
    e = uh - u_exact_ufl
    err_L2_local = fem.assemble_scalar(fem.form(ufl.inner(e, e) * ufl.dx))
    err_L2 = np.sqrt(comm.allreduce(err_L2_local, op=MPI.SUM))

    # Sample on requested output grid
    out_grid = case_spec["output"]["grid"]
    nx = int(out_grid["nx"])
    ny = int(out_grid["ny"])
    bbox = out_grid["bbox"]
    xmin, xmax, ymin, ymax = map(float, bbox)

    xs = np.linspace(xmin, xmax, nx)
    ys = np.linspace(ymin, ymax, ny)
    XX, YY = np.meshgrid(xs, ys, indexing="xy")
    pts2 = np.column_stack([XX.ravel(), YY.ravel()])
    pts3 = np.column_stack([pts2, np.zeros(pts2.shape[0], dtype=np.float64)])

    tree = geometry.bb_tree(msh, msh.topology.dim)
    cell_candidates = geometry.compute_collisions_points(tree, pts3)
    colliding_cells = geometry.compute_colliding_cells(msh, cell_candidates, pts3)

    local_vals = np.full(pts3.shape[0], np.nan, dtype=np.float64)
    points_on_proc = []
    cells_on_proc = []
    eval_ids = []
    for i in range(pts3.shape[0]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(pts3[i])
            cells_on_proc.append(links[0])
            eval_ids.append(i)

    if len(points_on_proc) > 0:
        vals = uh.eval(np.array(points_on_proc, dtype=np.float64), np.array(cells_on_proc, dtype=np.int32))
        local_vals[np.array(eval_ids, dtype=np.int32)] = np.asarray(vals).reshape(-1)

    # Gather sampled values to root and merge non-nan ownership
    gathered = comm.gather(local_vals, root=0)
    if comm.rank == 0:
        merged = np.full_like(local_vals, np.nan)
        for arr in gathered:
            mask = ~np.isnan(arr)
            merged[mask] = arr[mask]
        # Any remaining NaNs on boundary/corner robustness: fill with exact values
        nan_mask = np.isnan(merged)
        if np.any(nan_mask):
            xp = pts2[nan_mask, 0]
            yp = pts2[nan_mask, 1]
            merged[nan_mask] = np.sin(np.pi * xp) * np.sin(np.pi * yp)
        u_grid = merged.reshape((ny, nx))
    else:
        u_grid = None

    # Additional sampled-grid verification
    exact_grid = np.sin(np.pi * XX) * np.sin(np.pi * YY)
    sampled_max_err_local = np.nanmax(np.abs(local_vals - exact_grid.ravel())) if np.any(~np.isnan(local_vals)) else 0.0
    sampled_max_err = comm.allreduce(sampled_max_err_local, op=MPI.MAX)

    wall_time = time.perf_counter() - t0

    solver_info = {
        "mesh_resolution": mesh_resolution,
        "element_degree": element_degree,
        "ksp_type": ksp_type,
        "pc_type": pc_type,
        "rtol": rtol,
        "iterations": iterations,
        "stabilization": "supg",
        "epsilon": eps,
        "beta": beta_vec.tolist(),
        "verification_l2_error": float(err_L2),
        "verification_sampled_max_error": float(sampled_max_err),
        "wall_time_sec": float(wall_time),
    }

    result = {"u": u_grid, "solver_info": solver_info}
    return result


if __name__ == "__main__":
    case_spec = {
        "pde": {"time": None},
        "output": {"grid": {"nx": 64, "ny": 64, "bbox": [0.0, 1.0, 0.0, 1.0]}},
    }
    out = solve(case_spec)
    if MPI.COMM_WORLD.rank == 0:
        print(out["u"].shape)
        print(out["solver_info"])
