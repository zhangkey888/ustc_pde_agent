import numpy as np
from mpi4py import MPI
from dolfinx import fem, mesh, geometry
from dolfinx.fem import petsc
import basix.ufl
import ufl
from petsc4py import PETSc

ScalarType = PETSc.ScalarType


def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD

    # Parse case_spec - handle both flat and oracle_config formats
    oracle = case_spec.get("oracle_config", case_spec)

    # PDE parameters
    pde_sec = oracle.get("pde", {})
    pde_params = pde_sec.get("pde_params", {})
    nu_val = float(pde_params.get("nu", pde_sec.get("viscosity", 0.2)))

    # Domain
    output_sec = oracle.get("output", {})
    grid_sec = output_sec.get("grid", {})
    bbox = grid_sec.get("bbox", [0, 1, 0, 1])
    x_min, x_max = float(bbox[0]), float(bbox[1])
    y_min, y_max = float(bbox[2]), float(bbox[3])
    nx_out = int(grid_sec.get("nx", 100))
    ny_out = int(grid_sec.get("ny", 100))

    # Boundary conditions
    bc_sec = oracle.get("bc", {})
    dirichlet_bcs = bc_sec.get("dirichlet", [])

    # Adaptive mesh refinement
    resolutions = [32, 48, 64]
    prev_norm = None
    final_result = None
    final_info = None

    for N in resolutions:
        result, norm_val, info = _solve_stokes(
            comm, N, nu_val, x_min, x_max, y_min, y_max,
            dirichlet_bcs, nx_out, ny_out
        )
        if prev_norm is not None and norm_val > 0:
            rel_err = abs(norm_val - prev_norm) / max(norm_val, 1e-15)
            if rel_err < 0.01:
                return {"u": result, "solver_info": info}
        prev_norm = norm_val
        final_result = result
        final_info = info

    return {"u": final_result, "solver_info": final_info}


def _solve_stokes(comm, N, nu_val, x_min, x_max, y_min, y_max,
                  dirichlet_bcs, nx_out, ny_out):
    p0 = np.array([x_min, y_min])
    p1 = np.array([x_max, y_max])
    domain = mesh.create_rectangle(
        comm, [p0, p1], [N, N],
        cell_type=mesh.CellType.triangle
    )
    gdim = domain.geometry.dim
    tdim = domain.topology.dim
    fdim = tdim - 1

    degree_u = 2
    degree_p = 1
    V_el = basix.ufl.element("Lagrange", domain.topology.cell_name(), degree_u, shape=(gdim,))
    Q_el = basix.ufl.element("Lagrange", domain.topology.cell_name(), degree_p)
    mel = basix.ufl.mixed_element([V_el, Q_el])
    W = fem.functionspace(domain, mel)

    V, V_to_W = W.sub(0).collapse()
    Q, Q_to_W = W.sub(1).collapse()

    (u, p) = ufl.TrialFunctions(W)
    (v, q) = ufl.TestFunctions(W)

    # Use plain float for viscosity like the oracle
    nu = nu_val
    f = fem.Constant(domain, (0.0, 0.0))

    a = (
        nu * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
        - ufl.div(v) * p * ufl.dx
        - q * ufl.div(u) * ufl.dx
    )
    L_form = ufl.inner(f, v) * ufl.dx

    bcs = []

    if dirichlet_bcs:
        for bc_item in dirichlet_bcs:
            location = bc_item.get("on", bc_item.get("location", ""))
            value = bc_item.get("value", ["0.0", "0.0"])

            marker = _get_boundary_marker(location, x_min, x_max, y_min, y_max)
            boundary_dofs = fem.locate_dofs_geometrical((W.sub(0), V), marker)

            u_bc = fem.Function(V)
            if isinstance(value, (list, tuple)):
                vx = float(value[0])
                vy = float(value[1])
                u_bc.interpolate(lambda x, vx=vx, vy=vy: np.array(
                    [[vx] * x.shape[1], [vy] * x.shape[1]]
                ))
            else:
                vv = float(value)
                u_bc.interpolate(lambda x, vv=vv: np.array(
                    [[vv] * x.shape[1], [0.0] * x.shape[1]]
                ))
            bc = fem.dirichletbc(u_bc, boundary_dofs, W.sub(0))
            bcs.append(bc)
    else:
        # Default lid-driven cavity
        for on, val in [("y1", [1.0, 0.0]), ("y0", [0.0, 0.0]),
                        ("x0", [0.0, 0.0]), ("x1", [0.0, 0.0])]:
            marker = _get_boundary_marker(on, x_min, x_max, y_min, y_max)
            boundary_dofs = fem.locate_dofs_geometrical((W.sub(0), V), marker)
            u_bc = fem.Function(V)
            vx, vy = float(val[0]), float(val[1])
            u_bc.interpolate(lambda x, vx=vx, vy=vy: np.array(
                [[vx] * x.shape[1], [vy] * x.shape[1]]
            ))
            bcs.append(fem.dirichletbc(u_bc, boundary_dofs, W.sub(0)))

    # Pin pressure at corner to remove nullspace
    p_dofs = fem.locate_dofs_geometrical(
        (W.sub(1), Q),
        lambda x: np.isclose(x[0], x_min) & np.isclose(x[1], y_min)
    )
    p_zero = fem.Function(Q)
    p_zero.x.array[:] = 0.0
    bc_p = fem.dirichletbc(p_zero, p_dofs, W.sub(1))
    bcs.append(bc_p)

    # Use minres + hypre like the oracle (critical for saddle-point systems)
    ksp_type = "minres"
    pc_type = "hypre"
    rtol = 1e-10

    problem = petsc.LinearProblem(
        a, L_form, bcs=bcs,
        petsc_options={
            "ksp_type": ksp_type,
            "pc_type": pc_type,
            "ksp_rtol": str(rtol),
            "ksp_max_it": "5000",
        },
        petsc_options_prefix="stokes_"
    )
    wh = problem.solve()

    uh = wh.sub(0).collapse()

    norm_val = np.sqrt(comm.allreduce(
        fem.assemble_scalar(fem.form(ufl.inner(uh, uh) * ufl.dx)),
        op=MPI.SUM
    ))

    u_grid = _evaluate_on_grid(domain, uh, nx_out, ny_out, x_min, x_max, y_min, y_max)

    info = {
        "mesh_resolution": N,
        "element_degree": degree_u,
        "ksp_type": ksp_type,
        "pc_type": pc_type,
        "rtol": rtol,
        "iterations": 1,
    }

    return u_grid, norm_val, info


def _get_boundary_marker(location, x_min, x_max, y_min, y_max):
    loc = str(location).lower().strip()
    if loc in ("y1", "top", "y_max", "upper", "ymax"):
        return lambda x: np.isclose(x[1], y_max)
    elif loc in ("y0", "bottom", "y_min", "lower", "ymin"):
        return lambda x: np.isclose(x[1], y_min)
    elif loc in ("x0", "left", "x_min", "xmin"):
        return lambda x: np.isclose(x[0], x_min)
    elif loc in ("x1", "right", "x_max", "xmax"):
        return lambda x: np.isclose(x[0], x_max)
    elif loc in ("all", "boundary", "entire", "*"):
        return lambda x: np.ones(x.shape[1], dtype=bool)
    else:
        return lambda x: np.ones(x.shape[1], dtype=bool)


def _evaluate_on_grid(domain, uh, nx, ny, x_min, x_max, y_min, y_max):
    xs = np.linspace(x_min, x_max, nx)
    ys = np.linspace(y_min, y_max, ny)
    XX, YY = np.meshgrid(xs, ys, indexing='ij')
    points_2d = np.column_stack([XX.ravel(), YY.ravel()])
    points_3d = np.zeros((points_2d.shape[0], 3))
    points_3d[:, :2] = points_2d

    bb_tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, points_3d)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points_3d)

    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    for i in range(points_3d.shape[0]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(points_3d[i])
            cells_on_proc.append(links[0])
            eval_map.append(i)

    u_mag = np.zeros(nx * ny)
    if len(points_on_proc) > 0:
        pts = np.array(points_on_proc)
        cls = np.array(cells_on_proc, dtype=np.int32)
        vals = uh.eval(pts, cls)
        if vals.ndim == 1:
            u_mag[eval_map] = np.abs(vals)
        else:
            u_mag[eval_map] = np.linalg.norm(vals, axis=1)

    u_grid = u_mag.reshape(nx, ny)
    return u_grid
