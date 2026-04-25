import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl

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
# special_notes: manufactured_solution, variable_coeff
# ```
#
# ```METHOD
# spatial_method: fem
# element_or_basis: Lagrange_P2
# stabilization: none
# time_method: none
# nonlinear_solver: none
# linear_solver: cg
# preconditioner: hypre
# special_treatment: none
# pde_skill: poisson
# ```

ScalarType = PETSc.ScalarType


def _make_exact_and_coeff_expressions(msh):
    x = ufl.SpatialCoordinate(msh)
    pi = np.pi

    u_exact = ufl.sin(3.0 * pi * x[0]) * ufl.sin(2.0 * pi * x[1])
    kappa = 1.0 + 0.9 * ufl.sin(2.0 * pi * x[0]) * ufl.sin(2.0 * pi * x[1])
    f = -ufl.div(kappa * ufl.grad(u_exact))
    return u_exact, kappa, f


def _sample_function_on_grid(msh, uh, grid_spec):
    nx = int(grid_spec["nx"])
    ny = int(grid_spec["ny"])
    bbox = grid_spec["bbox"]
    xmin, xmax, ymin, ymax = map(float, bbox)

    xs = np.linspace(xmin, xmax, nx)
    ys = np.linspace(ymin, ymax, ny)
    XX, YY = np.meshgrid(xs, ys, indexing="xy")
    pts = np.column_stack([XX.ravel(), YY.ravel(), np.zeros(nx * ny, dtype=np.float64)])

    tree = geometry.bb_tree(msh, msh.topology.dim)
    cell_candidates = geometry.compute_collisions_points(tree, pts)
    colliding_cells = geometry.compute_colliding_cells(msh, cell_candidates, pts)

    local_vals = np.full((nx * ny,), np.nan, dtype=np.float64)
    points_on_proc = []
    cells_on_proc = []
    eval_map = []

    for i in range(pts.shape[0]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(pts[i])
            cells_on_proc.append(links[0])
            eval_map.append(i)

    if len(points_on_proc) > 0:
        vals = uh.eval(np.array(points_on_proc, dtype=np.float64), np.array(cells_on_proc, dtype=np.int32))
        vals = np.asarray(vals).reshape(len(points_on_proc), -1)[:, 0]
        local_vals[np.array(eval_map, dtype=np.int32)] = vals

    comm = msh.comm
    gathered = comm.gather(local_vals, root=0)

    if comm.rank == 0:
        final = np.full((nx * ny,), np.nan, dtype=np.float64)
        for arr in gathered:
            mask = ~np.isnan(arr)
            final[mask] = arr[mask]
        if np.isnan(final).any():
            nan_idx = np.where(np.isnan(final))[0]
            raise RuntimeError(f"Failed to evaluate solution at {len(nan_idx)} output grid points.")
        return final.reshape((ny, nx))
    return None


def _compute_errors(msh, V, uh, u_exact_expr):
    degree_raise = 3
    u_ex = fem.Function(V)
    u_ex.interpolate(fem.Expression(u_exact_expr, V.element.interpolation_points))
    diff = fem.Function(V)
    diff.x.array[:] = uh.x.array - u_ex.x.array
    diff.x.scatter_forward()

    l2_form = fem.form(ufl.inner(diff, diff) * ufl.dx)
    l2_err_local = fem.assemble_scalar(l2_form)
    l2_err = np.sqrt(msh.comm.allreduce(l2_err_local, op=MPI.SUM))

    ex_l2_form = fem.form(ufl.inner(u_ex, u_ex) * ufl.dx)
    ex_l2_local = fem.assemble_scalar(ex_l2_form)
    ex_l2 = np.sqrt(msh.comm.allreduce(ex_l2_local, op=MPI.SUM))
    rel_l2 = l2_err / ex_l2 if ex_l2 > 0 else l2_err
    return l2_err, rel_l2


def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD

    # Accuracy/time trade-off tuned for smooth manufactured solution under tight time budget.
    mesh_resolution = 56
    element_degree = 2
    ksp_type = "cg"
    pc_type = "hypre"
    rtol = 1e-10

    msh = mesh.create_unit_square(
        comm, nx=mesh_resolution, ny=mesh_resolution, cell_type=mesh.CellType.triangle
    )
    V = fem.functionspace(msh, ("Lagrange", element_degree))

    u_exact_expr, kappa_expr, f_expr = _make_exact_and_coeff_expressions(msh)

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    kappa = fem.Function(V)
    kappa.interpolate(fem.Expression(kappa_expr, V.element.interpolation_points))

    f = fem.Function(V)
    f.interpolate(fem.Expression(f_expr, V.element.interpolation_points))

    a = ufl.inner(kappa * ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = ufl.inner(f, v) * ufl.dx

    tdim = msh.topology.dim
    fdim = tdim - 1
    boundary_facets = mesh.locate_entities_boundary(
        msh, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
    )
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)

    u_bc = fem.Function(V)
    u_bc.interpolate(fem.Expression(u_exact_expr, V.element.interpolation_points))
    bc = fem.dirichletbc(u_bc, boundary_dofs)

    problem = petsc.LinearProblem(
        a,
        L,
        bcs=[bc],
        petsc_options_prefix="poisson_var_kappa_",
        petsc_options={
            "ksp_type": ksp_type,
            "pc_type": pc_type,
            "ksp_rtol": rtol,
            "ksp_atol": 1e-14,
            "ksp_max_it": 5000,
        },
    )

    uh = problem.solve()
    uh.x.scatter_forward()

    # Record actual iterations if possible
    iterations = -1
    try:
        solver = problem.solver
        if solver is not None:
            iterations = int(solver.getIterationNumber())
    except Exception:
        iterations = -1

    l2_err, rel_l2 = _compute_errors(msh, V, uh, u_exact_expr)

    u_grid = _sample_function_on_grid(msh, uh, case_spec["output"]["grid"])
    if comm.rank != 0:
        return {"u": None, "solver_info": {}}

    solver_info = {
        "mesh_resolution": mesh_resolution,
        "element_degree": element_degree,
        "ksp_type": ksp_type,
        "pc_type": pc_type,
        "rtol": rtol,
        "iterations": iterations,
        "l2_error": float(l2_err),
        "relative_l2_error": float(rel_l2),
    }

    return {
        "u": u_grid,
        "solver_info": solver_info,
    }
