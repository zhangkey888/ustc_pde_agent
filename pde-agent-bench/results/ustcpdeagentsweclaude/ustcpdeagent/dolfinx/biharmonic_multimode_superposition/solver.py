import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
from basix.ufl import element, mixed_element
import ufl


ScalarType = PETSc.ScalarType


def _exact_u_numpy(x, y):
    return np.sin(np.pi * x) * np.sin(np.pi * y) + 0.5 * np.sin(2 * np.pi * x) * np.sin(3 * np.pi * y)


def _probe_points_scalar(domain, u_func, points_array):
    bb_tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, points_array)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points_array)

    local_vals = np.full(points_array.shape[0], np.nan, dtype=np.float64)
    points_on_proc = []
    cells_on_proc = []
    map_ids = []

    for i in range(points_array.shape[0]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(points_array[i])
            cells_on_proc.append(links[0])
            map_ids.append(i)

    if len(points_on_proc) > 0:
        vals = u_func.eval(np.array(points_on_proc, dtype=np.float64), np.array(cells_on_proc, dtype=np.int32))
        local_vals[np.array(map_ids, dtype=np.int32)] = np.asarray(vals, dtype=np.float64).reshape(-1)

    global_vals = np.empty_like(local_vals)
    domain.comm.Allreduce(local_vals, global_vals, op=MPI.MAX)
    return global_vals


def solve(case_spec: dict) -> dict:
    """
    Solve Δ²u = f on the unit square using a mixed formulation:
        w = -Δu
       -Δw = f
    with Dirichlet conditions u = g and w = -Δu_exact on the boundary.

    Returns
    -------
    dict with keys:
      - "u": sampled solution on requested grid, shape (ny, nx)
      - "solver_info": metadata including accuracy diagnostics
    """
    comm = MPI.COMM_WORLD

    # Problem-adapted defaults chosen for accuracy within time budget
    mesh_resolution = int(case_spec.get("solver", {}).get("mesh_resolution", 48))
    element_degree = int(case_spec.get("solver", {}).get("element_degree", 2))
    ksp_type = case_spec.get("solver", {}).get("ksp_type", "gmres")
    pc_type = case_spec.get("solver", {}).get("pc_type", "hypre")
    rtol = float(case_spec.get("solver", {}).get("rtol", 1.0e-10))

    # Domain
    domain = mesh.create_unit_square(comm, mesh_resolution, mesh_resolution, cell_type=mesh.CellType.triangle)
    cell_name = domain.topology.cell_name()

    # Mixed space for (u, w)
    P = element("Lagrange", cell_name, element_degree)
    W = fem.functionspace(domain, mixed_element([P, P]))
    V, _ = W.sub(0).collapse()

    x = ufl.SpatialCoordinate(domain)

    # Manufactured exact solution and derived quantities
    u_exact_ufl = ufl.sin(ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1]) + 0.5 * ufl.sin(2 * ufl.pi * x[0]) * ufl.sin(3 * ufl.pi * x[1])
    lap_u_exact = ufl.div(ufl.grad(u_exact_ufl))
    w_exact_ufl = -lap_u_exact
    f_ufl = -ufl.div(ufl.grad(w_exact_ufl))  # since -Δw = f

    # Trial/test
    (u, w) = ufl.TrialFunctions(W)
    (v, z) = ufl.TestFunctions(W)

    # Block-diagonal mixed formulation = two Poisson solves in one system
    a = (
        ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
        + ufl.inner(ufl.grad(w), ufl.grad(z)) * ufl.dx
    )
    L = (
        ufl.inner(w, v) * ufl.dx
        + ufl.inner(f_ufl, z) * ufl.dx
    )

    # Dirichlet BCs on entire boundary for u and w
    fdim = domain.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(domain, fdim, lambda X: np.ones(X.shape[1], dtype=bool))

    u_bc_fun = fem.Function(V)
    u_bc_fun.interpolate(fem.Expression(u_exact_ufl, V.element.interpolation_points))
    w_bc_fun = fem.Function(V)
    w_bc_fun.interpolate(fem.Expression(w_exact_ufl, V.element.interpolation_points))

    dofs_u = fem.locate_dofs_topological((W.sub(0), V), fdim, boundary_facets)
    dofs_w = fem.locate_dofs_topological((W.sub(1), V), fdim, boundary_facets)
    bc_u = fem.dirichletbc(u_bc_fun, dofs_u, W.sub(0))
    bc_w = fem.dirichletbc(w_bc_fun, dofs_w, W.sub(1))
    bcs = [bc_u, bc_w]

    # Solve
    problem = petsc.LinearProblem(
        a, L, bcs=bcs,
        petsc_options_prefix="biharmonic_",
        petsc_options={
            "ksp_type": ksp_type,
            "pc_type": pc_type,
            "ksp_rtol": rtol,
            "ksp_atol": 1.0e-14,
            "ksp_max_it": 1000,
        },
    )
    wh = problem.solve()
    wh.x.scatter_forward()

    uh = wh.sub(0).collapse()

    # Accuracy verification: L2 error against exact solution
    u_ex = fem.Function(V)
    u_ex.interpolate(fem.Expression(u_exact_ufl, V.element.interpolation_points))
    err_form = fem.form((uh - u_ex) ** 2 * ufl.dx)
    l2_err_sq_local = fem.assemble_scalar(err_form)
    l2_err_sq = comm.allreduce(l2_err_sq_local, op=MPI.SUM)
    l2_error = float(np.sqrt(max(l2_err_sq, 0.0)))

    # Sample on requested output grid
    out_grid = case_spec["output"]["grid"]
    nx = int(out_grid["nx"])
    ny = int(out_grid["ny"])
    xmin, xmax, ymin, ymax = out_grid["bbox"]

    xs = np.linspace(xmin, xmax, nx, dtype=np.float64)
    ys = np.linspace(ymin, ymax, ny, dtype=np.float64)
    XX, YY = np.meshgrid(xs, ys, indexing="xy")
    pts = np.column_stack([XX.ravel(), YY.ravel(), np.zeros(nx * ny, dtype=np.float64)])

    vals = _probe_points_scalar(domain, uh, pts)
    u_grid = vals.reshape(ny, nx)

    # PETSc iteration count if available
    iterations = 0
    try:
        ksp = problem.solver
        iterations = int(ksp.getIterationNumber())
        ksp_type_actual = ksp.getType()
        pc_type_actual = ksp.getPC().getType()
    except Exception:
        ksp_type_actual = ksp_type
        pc_type_actual = pc_type

    solver_info = {
        "mesh_resolution": mesh_resolution,
        "element_degree": element_degree,
        "ksp_type": str(ksp_type_actual),
        "pc_type": str(pc_type_actual),
        "rtol": rtol,
        "iterations": iterations,
        "l2_error": l2_error,
    }

    return {
        "u": u_grid,
        "solver_info": solver_info,
    }


if __name__ == "__main__":
    case_spec = {
        "output": {
            "grid": {
                "nx": 64,
                "ny": 64,
                "bbox": [0.0, 1.0, 0.0, 1.0],
            }
        }
    }
    result = solve(case_spec)
    print(result["u"].shape)
    print(result["solver_info"])
