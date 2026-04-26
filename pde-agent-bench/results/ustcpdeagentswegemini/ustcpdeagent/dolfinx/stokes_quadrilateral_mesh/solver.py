import time
import math
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc

from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from basix.ufl import element as basix_element, mixed_element as basix_mixed_element


def _make_case_spec(nx=128, ny=128):
    return {
        "case_id": "stokes_quadrilateral_mesh",
        "pde": {"time": None},
        "output": {
            "grid": {
                "nx": nx,
                "ny": ny,
                "bbox": [0.0, 1.0, 0.0, 1.0],
            }
        },
    }


def _sample_function(func, points):
    msh = func.function_space.mesh
    tree = geometry.bb_tree(msh, msh.topology.dim)
    candidates = geometry.compute_collisions_points(tree, points)
    colliding = geometry.compute_colliding_cells(msh, candidates, points)

    local_ids = []
    local_points = []
    local_cells = []
    for i in range(points.shape[0]):
        links = colliding.links(i)
        if len(links) > 0:
            local_ids.append(i)
            local_points.append(points[i])
            local_cells.append(links[0])

    value_shape = func.function_space.element.value_shape
    val_size = int(np.prod(value_shape)) if len(value_shape) > 0 else 1
    local_vals = np.empty((0, val_size), dtype=np.float64)
    if len(local_points) > 0:
        vals = func.eval(np.array(local_points, dtype=np.float64), np.array(local_cells, dtype=np.int32))
        vals = np.asarray(vals, dtype=np.float64)
        local_vals = vals.reshape(len(local_points), val_size)

    gathered_ids = msh.comm.allgather(np.array(local_ids, dtype=np.int32))
    gathered_vals = msh.comm.allgather(local_vals)

    result = np.full((points.shape[0], val_size), np.nan, dtype=np.float64)
    for ids, vals in zip(gathered_ids, gathered_vals):
        if len(ids) > 0:
            result[ids] = vals

    if np.isnan(result).any():
        raise RuntimeError("Point evaluation failed for some sampling points.")
    return result


def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD
    nu_value = 1.0
    mesh_resolution = 160
    degree_u = 2
    degree_p = 1
    ksp_type = "preonly"
    pc_type = "lu"
    rtol = 1.0e-12

    msh = mesh.create_rectangle(
        comm,
        [np.array([0.0, 0.0]), np.array([1.0, 1.0])],
        [mesh_resolution, mesh_resolution],
        cell_type=mesh.CellType.quadrilateral,
    )
    gdim = msh.geometry.dim
    cell_name = msh.topology.cell_name()

    vel_el = basix_element("Lagrange", cell_name, degree_u, shape=(gdim,))
    pres_el = basix_element("Lagrange", cell_name, degree_p)
    W = fem.functionspace(msh, basix_mixed_element([vel_el, pres_el]))
    V, _ = W.sub(0).collapse()
    Q, _ = W.sub(1).collapse()

    x = ufl.SpatialCoordinate(msh)
    pi = ufl.pi
    nu = fem.Constant(msh, PETSc.ScalarType(nu_value))

    u_exact_ufl = ufl.as_vector(
        [
            pi * ufl.cos(pi * x[1]) * ufl.sin(pi * x[0]),
            -pi * ufl.cos(pi * x[0]) * ufl.sin(pi * x[1]),
        ]
    )
    p_exact_ufl = ufl.cos(pi * x[0]) * ufl.cos(pi * x[1])

    f_ufl = -nu * ufl.div(ufl.grad(u_exact_ufl)) + ufl.grad(p_exact_ufl)

    (u, p) = ufl.TrialFunctions(W)
    (v, q) = ufl.TestFunctions(W)

    a = (
        nu * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
        - ufl.inner(p, ufl.div(v)) * ufl.dx
        + ufl.inner(ufl.div(u), q) * ufl.dx
    )
    L = ufl.inner(f_ufl, v) * ufl.dx

    u_bc = fem.Function(V)
    u_bc.interpolate(fem.Expression(u_exact_ufl, V.element.interpolation_points))

    fdim = msh.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(msh, fdim, lambda X: np.ones(X.shape[1], dtype=bool))
    u_dofs = fem.locate_dofs_topological((W.sub(0), V), fdim, boundary_facets)
    bc_u = fem.dirichletbc(u_bc, u_dofs, W.sub(0))

    p_pin_dofs = fem.locate_dofs_geometrical(
        (W.sub(1), Q),
        lambda X: np.isclose(X[0], 0.0) & np.isclose(X[1], 0.0),
    )
    p0 = fem.Function(Q)
    p0.x.array[:] = 0.0
    bc_p = fem.dirichletbc(p0, p_pin_dofs, W.sub(1))
    bcs = [bc_u, bc_p]

    problem = petsc.LinearProblem(
        a,
        L,
        bcs=bcs,
        petsc_options_prefix="stokes_",
        petsc_options={
            "ksp_type": ksp_type,
            "pc_type": pc_type,
            "pc_factor_mat_solver_type": "mumps",
        },
    )
    wh = problem.solve()
    wh.x.scatter_forward()

    uh = wh.sub(0).collapse()
    ph = wh.sub(1).collapse()

    u_exact = fem.Function(V)
    u_exact.interpolate(fem.Expression(u_exact_ufl, V.element.interpolation_points))

    err_fun = fem.Function(V)
    err_fun.x.array[:] = uh.x.array - u_exact.x.array
    local_l2_sq = fem.assemble_scalar(fem.form(ufl.inner(err_fun, err_fun) * ufl.dx))
    global_l2_sq = comm.allreduce(local_l2_sq, op=MPI.SUM)
    l2_error = math.sqrt(global_l2_sq)

    grid = case_spec["output"]["grid"]
    nx = int(grid["nx"])
    ny = int(grid["ny"])
    xmin, xmax, ymin, ymax = grid["bbox"]
    xs = np.linspace(xmin, xmax, nx)
    ys = np.linspace(ymin, ymax, ny)
    XX, YY = np.meshgrid(xs, ys, indexing="xy")
    points = np.column_stack(
        [XX.ravel(), YY.ravel(), np.zeros(nx * ny, dtype=np.float64)]
    )
    uvals = _sample_function(uh, points)
    umag = np.linalg.norm(uvals, axis=1).reshape(ny, nx)

    iterations = 1
    ksp = problem.solver
    try:
        iterations = int(ksp.getIterationNumber())
        if iterations <= 0 and ksp_type == "preonly":
            iterations = 1
    except Exception:
        iterations = 1

    solver_info = {
        "mesh_resolution": mesh_resolution,
        "element_degree": degree_u,
        "ksp_type": ksp_type,
        "pc_type": pc_type,
        "rtol": rtol,
        "iterations": iterations,
        "l2_error": l2_error,
    }
    return {"u": umag, "solver_info": solver_info}


if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    case_spec = _make_case_spec(128, 128)
    t0 = time.perf_counter()
    out = solve(case_spec)
    wall = time.perf_counter() - t0
    l2_error = out["solver_info"]["l2_error"]
    if comm.rank == 0:
        print(f"L2_ERROR: {l2_error:.16e}")
        print(f"WALL_TIME: {wall:.16f}")
        print(f"OUTPUT_SHAPE: {out['u'].shape}")
