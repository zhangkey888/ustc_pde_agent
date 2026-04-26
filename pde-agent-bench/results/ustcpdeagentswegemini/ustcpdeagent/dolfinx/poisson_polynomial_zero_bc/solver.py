import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
import ufl
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc


def _sample_on_grid(u_func, nx, ny, bbox):
    msh = u_func.function_space.mesh
    xmin, xmax, ymin, ymax = bbox
    xs = np.linspace(xmin, xmax, nx)
    ys = np.linspace(ymin, ymax, ny)
    XX, YY = np.meshgrid(xs, ys, indexing="xy")
    pts = np.column_stack(
        [XX.ravel(), YY.ravel(), np.zeros(nx * ny, dtype=np.float64)]
    )

    tree = geometry.bb_tree(msh, msh.topology.dim)
    cell_candidates = geometry.compute_collisions_points(tree, pts)
    colliding_cells = geometry.compute_colliding_cells(msh, cell_candidates, pts)

    values = np.full(nx * ny, np.nan, dtype=np.float64)
    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    for i in range(pts.shape[0]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(pts[i])
            cells_on_proc.append(links[0])
            eval_map.append(i)

    if points_on_proc:
        vals = u_func.eval(np.array(points_on_proc, dtype=np.float64),
                           np.array(cells_on_proc, dtype=np.int32))
        vals = np.asarray(vals).reshape(len(points_on_proc), -1)[:, 0]
        values[np.array(eval_map, dtype=np.int32)] = vals

    # For this benchmark on unit square, all points should exist on rank 0 in serial.
    # In parallel, gather robustly and take non-NaN entries.
    comm = msh.comm
    gathered = comm.gather(values, root=0)
    if comm.rank == 0:
        merged = np.full_like(values, np.nan)
        for arr in gathered:
            mask = ~np.isnan(arr)
            merged[mask] = arr[mask]
        # Fill any remaining NaNs conservatively with 0 on the boundary/corners
        merged = np.nan_to_num(merged, nan=0.0)
        grid = merged.reshape(ny, nx)
    else:
        grid = None
    return grid


def solve(case_spec: dict) -> dict:
    """
    Solve manufactured Poisson problem on unit square and sample to output grid.

    Returns:
      dict with keys:
        - "u": ndarray of shape (ny, nx)
        - "solver_info": metadata dict
    """
    # ```DIAGNOSIS
    # equation_type: poisson
    # spatial_dim: 2
    # domain_geometry: rectangle
    # unknowns: scalar
    # coupling: none
    # linearity: linear
    # time_dependence: steady
    # stiffness: N/A
    # dominant_physics: diffusion
    # peclet_or_reynolds: N/A
    # solution_regularity: smooth
    # bc_type: all_dirichlet
    # special_notes: manufactured_solution
    # ```
    #
    # ```METHOD
    # spatial_method: fem
    # element_or_basis: Lagrange_P2
    # stabilization: none
    # time_method: none
    # nonlinear_solver: none
    # linear_solver: direct_lu
    # preconditioner: none
    # special_treatment: none
    # pde_skill: poisson
    # ```

    comm = MPI.COMM_WORLD
    ScalarType = PETSc.ScalarType

    # Tight time budget: use a reliable high-accuracy choice that remains fast.
    mesh_resolution = int(case_spec.get("solver", {}).get("mesh_resolution", 24))
    element_degree = int(case_spec.get("solver", {}).get("element_degree", 2))

    msh = mesh.create_unit_square(
        comm, mesh_resolution, mesh_resolution, cell_type=mesh.CellType.triangle
    )
    V = fem.functionspace(msh, ("Lagrange", element_degree))

    x = ufl.SpatialCoordinate(msh)
    u_exact_ufl = x[0] * (1.0 - x[0]) * x[1] * (1.0 - x[1])
    f_ufl = 2.0 * (x[0] * (1.0 - x[0]) + x[1] * (1.0 - x[1]))

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    a = ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = ufl.inner(f_ufl, v) * ufl.dx

    fdim = msh.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(
        msh, fdim, lambda X: np.ones(X.shape[1], dtype=bool)
    )
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc = fem.dirichletbc(ScalarType(0.0), boundary_dofs, V)

    ksp_type = "preonly"
    pc_type = "lu"
    rtol = 1.0e-12

    problem = petsc.LinearProblem(
        a,
        L,
        bcs=[bc],
        petsc_options_prefix="poisson_",
        petsc_options={
            "ksp_type": ksp_type,
            "pc_type": pc_type,
        },
    )
    uh = problem.solve()
    uh.x.scatter_forward()

    # Accuracy verification
    u_exact = fem.Function(V)
    u_exact.interpolate(lambda X: X[0] * (1.0 - X[0]) * X[1] * (1.0 - X[1]))
    err_fun = fem.Function(V)
    err_fun.x.array[:] = uh.x.array - u_exact.x.array
    err_fun.x.scatter_forward()

    l2_error_local = fem.assemble_scalar(fem.form(ufl.inner(err_fun, err_fun) * ufl.dx))
    l2_error = np.sqrt(comm.allreduce(l2_error_local, op=MPI.SUM))
    max_error_local = np.max(np.abs(err_fun.x.array)) if err_fun.x.array.size > 0 else 0.0
    max_error = comm.allreduce(max_error_local, op=MPI.MAX)

    grid_spec = case_spec["output"]["grid"]
    nx = int(grid_spec["nx"])
    ny = int(grid_spec["ny"])
    bbox = grid_spec["bbox"]

    u_grid = _sample_on_grid(uh, nx, ny, bbox)

    # Iteration count extraction
    iterations = 1
    try:
        ksp = problem.solver
        iterations = int(ksp.getIterationNumber())
    except Exception:
        iterations = 1

    result = None
    if comm.rank == 0:
        solver_info = {
            "mesh_resolution": mesh_resolution,
            "element_degree": element_degree,
            "ksp_type": ksp_type,
            "pc_type": pc_type,
            "rtol": rtol,
            "iterations": iterations,
            "l2_error": float(l2_error),
            "max_error": float(max_error),
        }
        result = {"u": np.asarray(u_grid, dtype=np.float64).reshape(ny, nx),
                  "solver_info": solver_info}
    return result
