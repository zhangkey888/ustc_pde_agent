import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
import ufl
from basix.ufl import element as basix_element, mixed_element as basix_mixed_element
from dolfinx import fem, mesh, geometry
from dolfinx.fem import petsc

ScalarType = PETSc.ScalarType


def _build_spaces(msh, degree_u=2, degree_p=1):
    cell = msh.topology.cell_name()
    gdim = msh.geometry.dim
    vel_el = basix_element("Lagrange", cell, degree_u, shape=(gdim,))
    pre_el = basix_element("Lagrange", cell, degree_p)
    W = fem.functionspace(msh, basix_mixed_element([vel_el, pre_el]))
    V, _ = W.sub(0).collapse()
    Q, _ = W.sub(1).collapse()
    return W, V, Q


def _velocity_bc_function(V, expr):
    g = fem.Function(V)
    g.interpolate(expr)
    return g


def _locate_boundary_dofs(W_sub, Vsub, msh, marker):
    fdim = msh.topology.dim - 1
    facets = mesh.locate_entities_boundary(msh, fdim, marker)
    return fem.locate_dofs_topological((W_sub, Vsub), fdim, facets)


def _eval_vector_function(func, pts3):
    msh = func.function_space.mesh
    gdim = msh.geometry.dim
    tree = geometry.bb_tree(msh, msh.topology.dim)
    cell_candidates = geometry.compute_collisions_points(tree, pts3)
    colliding_cells = geometry.compute_colliding_cells(msh, cell_candidates, pts3)

    local_vals = np.full((pts3.shape[0], gdim), np.nan, dtype=np.float64)
    points_on_proc = []
    cells_on_proc = []
    eval_ids = []

    for i in range(pts3.shape[0]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(pts3[i])
            cells_on_proc.append(links[0])
            eval_ids.append(i)

    if points_on_proc:
        vals = func.eval(np.array(points_on_proc, dtype=np.float64),
                         np.array(cells_on_proc, dtype=np.int32))
        local_vals[np.array(eval_ids, dtype=np.int32), :] = np.real(vals)

    gathered = msh.comm.gather(local_vals, root=0)
    if msh.comm.rank == 0:
        merged = np.full_like(local_vals, np.nan)
        for arr in gathered:
            mask = np.isnan(merged[:, 0]) & (~np.isnan(arr[:, 0]))
            merged[mask] = arr[mask]
    else:
        merged = None
    merged = msh.comm.bcast(merged, root=0)
    return merged


def _sample_velocity_magnitude(u_func, nx, ny, bbox):
    xmin, xmax, ymin, ymax = bbox
    xs = np.linspace(xmin, xmax, nx)
    ys = np.linspace(ymin, ymax, ny)
    XX, YY = np.meshgrid(xs, ys, indexing="xy")
    pts3 = np.zeros((nx * ny, 3), dtype=np.float64)
    pts3[:, 0] = XX.ravel()
    pts3[:, 1] = YY.ravel()
    vals = _eval_vector_function(u_func, pts3)
    mag = np.linalg.norm(vals, axis=1).reshape(ny, nx)
    return mag


def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD

    output_grid = case_spec["output"]["grid"]
    nx_out = int(output_grid["nx"])
    ny_out = int(output_grid["ny"])
    bbox = output_grid["bbox"]

    mesh_resolution = int(case_spec.get("agent_params", {}).get("mesh_resolution", 96))
    degree_u = int(case_spec.get("agent_params", {}).get("degree_u", 2))
    degree_p = int(case_spec.get("agent_params", {}).get("degree_p", 1))
    nu_value = float(case_spec.get("pde", {}).get("nu", case_spec.get("viscosity", 0.8)))
    if nu_value <= 0.0:
        nu_value = 0.8

    msh = mesh.create_unit_square(
        comm, mesh_resolution, mesh_resolution, cell_type=mesh.CellType.triangle
    )
    W, V, Q = _build_spaces(msh, degree_u=degree_u, degree_p=degree_p)
    gdim = msh.geometry.dim

    def inflow_expr(x):
        vals = np.zeros((gdim, x.shape[1]), dtype=np.float64)
        prof = 2.0 * x[1] * (1.0 - x[1])
        vals[0, :] = prof
        vals[1, :] = prof
        return vals

    def zero_expr(x):
        return np.zeros((gdim, x.shape[1]), dtype=np.float64)

    u_in = _velocity_bc_function(V, inflow_expr)
    u_zero = _velocity_bc_function(V, zero_expr)

    dofs_x0 = _locate_boundary_dofs(W.sub(0), V, msh, lambda x: np.isclose(x[0], 0.0))
    dofs_y0 = _locate_boundary_dofs(W.sub(0), V, msh, lambda x: np.isclose(x[1], 0.0))
    dofs_y1 = _locate_boundary_dofs(W.sub(0), V, msh, lambda x: np.isclose(x[1], 1.0))

    bcs = [
        fem.dirichletbc(u_in, dofs_x0, W.sub(0)),
        fem.dirichletbc(u_zero, dofs_y0, W.sub(0)),
        fem.dirichletbc(u_zero, dofs_y1, W.sub(0)),
    ]

    p_dofs = fem.locate_dofs_geometrical(
        (W.sub(1), Q), lambda x: np.isclose(x[0], 0.0) & np.isclose(x[1], 0.0)
    )
    if len(p_dofs) > 0:
        p0 = fem.Function(Q)
        p0.x.array[:] = 0.0
        bcs.append(fem.dirichletbc(p0, p_dofs, W.sub(1)))

    (u, p) = ufl.TrialFunctions(W)
    (v, q) = ufl.TestFunctions(W)

    nu = fem.Constant(msh, ScalarType(nu_value))
    f = fem.Constant(msh, np.array([0.0, 0.0], dtype=ScalarType))

    def eps(w):
        return ufl.sym(ufl.grad(w))

    a = (
        2.0 * nu * ufl.inner(eps(u), eps(v)) * ufl.dx
        - p * ufl.div(v) * ufl.dx
        + ufl.div(u) * q * ufl.dx
    )
    L = ufl.inner(f, v) * ufl.dx

    ksp_type = str(case_spec.get("agent_params", {}).get("ksp_type", "minres"))
    pc_type = str(case_spec.get("agent_params", {}).get("pc_type", "hypre"))
    rtol = float(case_spec.get("agent_params", {}).get("rtol", 1e-9))
    iterations = -1

    try:
        problem = petsc.LinearProblem(
            a,
            L,
            bcs=bcs,
            petsc_options_prefix="stokes_iter_",
            petsc_options={
                "ksp_type": ksp_type,
                "pc_type": pc_type,
                "ksp_rtol": rtol,
                "ksp_atol": 1e-12,
                "ksp_max_it": 5000,
            },
        )
        wh = problem.solve()
        try:
            iterations = int(problem.solver.getIterationNumber())
        except Exception:
            iterations = -1
    except Exception:
        ksp_type = "preonly"
        pc_type = "lu"
        problem = petsc.LinearProblem(
            a,
            L,
            bcs=bcs,
            petsc_options_prefix="stokes_lu_",
            petsc_options={
                "ksp_type": ksp_type,
                "pc_type": pc_type,
                "ksp_rtol": rtol,
            },
        )
        wh = problem.solve()
        try:
            iterations = int(problem.solver.getIterationNumber())
        except Exception:
            iterations = 1

    wh.x.scatter_forward()
    u_sol, _ = wh.sub(0).collapse(), wh.sub(1).collapse()

    div_form = fem.form(ufl.inner(ufl.div(u_sol), ufl.div(u_sol)) * ufl.dx)
    div_l2_sq_local = fem.assemble_scalar(div_form)
    div_l2_sq = comm.allreduce(div_l2_sq_local, op=MPI.SUM)
    div_l2 = float(np.sqrt(max(div_l2_sq, 0.0)))

    ys = np.linspace(0.0, 1.0, max(64, ny_out))
    pts_left = np.zeros((len(ys), 3), dtype=np.float64)
    pts_right = np.zeros((len(ys), 3), dtype=np.float64)
    pts_left[:, 1] = ys
    pts_right[:, 0] = 1.0
    pts_right[:, 1] = ys
    vals_left = _eval_vector_function(u_sol, pts_left)
    vals_right = _eval_vector_function(u_sol, pts_right)
    inflow_flux = float(np.trapz(vals_left[:, 0], ys))
    outflow_flux = float(np.trapz(vals_right[:, 0], ys))
    flux_imbalance = float(abs(inflow_flux - outflow_flux))

    u_grid = _sample_velocity_magnitude(u_sol, nx_out, ny_out, bbox)

    solver_info = {
        "mesh_resolution": mesh_resolution,
        "element_degree": degree_u,
        "ksp_type": ksp_type,
        "pc_type": pc_type,
        "rtol": rtol,
        "iterations": int(iterations),
        "verification": {
            "divergence_l2": div_l2,
            "inflow_flux": inflow_flux,
            "outflow_flux": outflow_flux,
            "flux_imbalance": flux_imbalance,
        },
    }

    return {"u": u_grid, "solver_info": solver_info}


if __name__ == "__main__":
    case_spec = {
        "pde": {"nu": 0.8, "time": None},
        "output": {"grid": {"nx": 64, "ny": 64, "bbox": [0.0, 1.0, 0.0, 1.0]}},
    }
    result = solve(case_spec)
    if MPI.COMM_WORLD.rank == 0:
        print(result["u"].shape)
        print(result["solver_info"])
