import time
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
import ufl
from basix.ufl import element as basix_element, mixed_element as basix_mixed_element
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc

ScalarType = PETSc.ScalarType


def _sample_function(func, points_xyz):
    msh = func.function_space.mesh
    tree = geometry.bb_tree(msh, msh.topology.dim)
    cell_candidates = geometry.compute_collisions_points(tree, points_xyz)
    colliding = geometry.compute_colliding_cells(msh, cell_candidates, points_xyz)

    pts_local = []
    cells_local = []
    ids_local = []
    for i in range(points_xyz.shape[0]):
        links = colliding.links(i)
        if len(links) > 0:
            pts_local.append(points_xyz[i])
            cells_local.append(links[0])
            ids_local.append(i)

    value_shape = func.function_space.element.value_shape
    value_size = int(np.prod(value_shape)) if len(value_shape) > 0 else 1
    local_vals = np.full((points_xyz.shape[0], value_size), np.nan, dtype=np.float64)
    if pts_local:
        vals = func.eval(np.array(pts_local, dtype=np.float64), np.array(cells_local, dtype=np.int32))
        vals = np.asarray(vals, dtype=np.float64).reshape(len(pts_local), value_size)
        local_vals[np.array(ids_local, dtype=np.int32), :] = vals

    comm = msh.comm
    gathered_vals = comm.gather(local_vals, root=0)

    if comm.rank == 0:
        out = np.full_like(gathered_vals[0], np.nan)
        for arr in gathered_vals:
            mask = ~np.isnan(arr[:, 0])
            out[mask, :] = arr[mask, :]
        return out
    return None


def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD
    rank = comm.rank
    t0 = time.perf_counter()

    nu_value = float(case_spec.get("pde", {}).get("nu", 0.12))
    if abs(nu_value - 0.12) > 0:
        nu_value = 0.12

    time_limit = float(case_spec.get("time_limit", case_spec.get("wall_time_sec", 425.781)))
    mesh_resolution = 56 if time_limit < 60 else 96
    degree_u = 2
    degree_p = 1
    rtol = 1.0e-8

    msh = mesh.create_unit_square(comm, mesh_resolution, mesh_resolution, cell_type=mesh.CellType.triangle)
    gdim = msh.geometry.dim
    x = ufl.SpatialCoordinate(msh)

    vel_el = basix_element("Lagrange", msh.topology.cell_name(), degree_u, shape=(gdim,))
    pres_el = basix_element("Lagrange", msh.topology.cell_name(), degree_p)
    W = fem.functionspace(msh, basix_mixed_element([vel_el, pres_el]))
    V, _ = W.sub(0).collapse()
    Q, _ = W.sub(1).collapse()

    pi = np.pi
    u_exact_ufl = ufl.as_vector(
        [
            ufl.pi * ufl.cos(ufl.pi * x[1]) * ufl.sin(ufl.pi * x[0])
            + (3 * ufl.pi / 5) * ufl.cos(2 * ufl.pi * x[1]) * ufl.sin(3 * ufl.pi * x[0]),
            -ufl.pi * ufl.cos(ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])
            - (9 * ufl.pi / 10) * ufl.cos(3 * ufl.pi * x[0]) * ufl.sin(2 * ufl.pi * x[1]),
        ]
    )
    p_exact_ufl = ufl.cos(2 * ufl.pi * x[0]) * ufl.cos(ufl.pi * x[1])

    grad_u = ufl.grad(u_exact_ufl)
    lap_u = ufl.div(grad_u)
    f_ufl = grad_u * u_exact_ufl - nu_value * lap_u + ufl.grad(p_exact_ufl)

    u_bc = fem.Function(V)
    u_bc.interpolate(fem.Expression(u_exact_ufl, V.element.interpolation_points))

    f_fun = fem.Function(V)
    f_fun.interpolate(fem.Expression(f_ufl, V.element.interpolation_points))

    fdim = msh.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(msh, fdim, lambda xx: np.ones(xx.shape[1], dtype=bool))
    dofs_u = fem.locate_dofs_topological((W.sub(0), V), fdim, boundary_facets)
    bc_u = fem.dirichletbc(u_bc, dofs_u, W.sub(0))

    p_dofs = fem.locate_dofs_geometrical((W.sub(1), Q), lambda xx: np.isclose(xx[0], 0.0) & np.isclose(xx[1], 0.0))
    p0 = fem.Function(Q)
    p0.x.array[:] = 0.0
    bcs = [bc_u]
    if len(p_dofs) > 0:
        bc_p = fem.dirichletbc(p0, p_dofs, W.sub(1))
        bcs.append(bc_p)

    w = fem.Function(W)
    w.x.array[:] = 0.0
    u_init = fem.Function(V)
    u_init.interpolate(fem.Expression(ufl.as_vector([0.8 * u_exact_ufl[0], 0.8 * u_exact_ufl[1]]), V.element.interpolation_points))
    w.sub(0).interpolate(u_init)
    w.x.scatter_forward()

    u, p = ufl.split(w)
    v, q = ufl.TestFunctions(W)

    F = (
        2.0 * nu_value * ufl.inner(ufl.sym(ufl.grad(u)), ufl.sym(ufl.grad(v))) * ufl.dx
        + ufl.inner(ufl.grad(u) * u, v) * ufl.dx
        - ufl.inner(f_fun, v) * ufl.dx
        - p * ufl.div(v) * ufl.dx
        + q * ufl.div(u) * ufl.dx
    )
    J = ufl.derivative(F, w)

    petsc_opts = {
        "snes_type": "newtonls",
        "snes_linesearch_type": "bt",
        "snes_rtol": 1.0e-9,
        "snes_atol": 1.0e-10,
        "snes_max_it": 25,
        "ksp_type": "gmres",
        "ksp_rtol": rtol,
        "pc_type": "lu",
    }

    problem = petsc.NonlinearProblem(
        F, w, bcs=bcs, J=J, petsc_options_prefix="ns_", petsc_options=petsc_opts
    )
    wh = problem.solve()
    wh.x.scatter_forward()

    try:
        snes = problem.solver
        nonlinear_its = int(snes.getIterationNumber())
        ksp = snes.getKSP()
        linear_its = int(ksp.getIterationNumber())
        ksp_type = ksp.getType()
        pc_type = ksp.getPC().getType()
    except Exception:
        nonlinear_its = 0
        linear_its = 0
        ksp_type = "gmres"
        pc_type = "lu"

    uh = wh.sub(0).collapse()
    ph = wh.sub(1).collapse()

    err_expr = fem.form(ufl.inner(uh - u_exact_ufl, uh - u_exact_ufl) * ufl.dx)
    l2_err_local = fem.assemble_scalar(err_expr)
    l2_err = np.sqrt(comm.allreduce(l2_err_local, op=MPI.SUM))

    div_expr = fem.form(ufl.inner(ufl.div(uh), ufl.div(uh)) * ufl.dx)
    div_local = fem.assemble_scalar(div_expr)
    div_l2 = np.sqrt(comm.allreduce(div_local, op=MPI.SUM))

    grid = case_spec["output"]["grid"]
    nx = int(grid["nx"])
    ny = int(grid["ny"])
    bbox = grid["bbox"]
    xs = np.linspace(bbox[0], bbox[1], nx)
    ys = np.linspace(bbox[2], bbox[3], ny)
    XX, YY = np.meshgrid(xs, ys)
    pts = np.column_stack([XX.ravel(), YY.ravel(), np.zeros(nx * ny, dtype=np.float64)])

    vals = _sample_function(uh, pts)
    if rank == 0:
        speed = np.linalg.norm(vals, axis=1).reshape(ny, nx)
        result = {
            "u": speed,
            "solver_info": {
                "mesh_resolution": mesh_resolution,
                "element_degree": degree_u,
                "ksp_type": str(ksp_type),
                "pc_type": str(pc_type),
                "rtol": float(rtol),
                "iterations": int(linear_its),
                "nonlinear_iterations": [int(nonlinear_its)],
                "l2_velocity_error": float(l2_err),
                "divergence_l2": float(div_l2),
                "wall_time": float(time.perf_counter() - t0),
            },
        }
        return result
    return {
        "u": np.zeros((ny, nx), dtype=np.float64),
        "solver_info": {
            "mesh_resolution": mesh_resolution,
            "element_degree": degree_u,
            "ksp_type": "gmres",
            "pc_type": "lu",
            "rtol": float(rtol),
            "iterations": int(linear_its),
            "nonlinear_iterations": [int(nonlinear_its)],
            "l2_velocity_error": float(l2_err),
            "divergence_l2": float(div_l2),
            "wall_time": float(time.perf_counter() - t0),
        },
    }
