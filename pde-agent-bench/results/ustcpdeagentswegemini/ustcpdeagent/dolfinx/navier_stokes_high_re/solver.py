import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
import ufl

from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
from basix.ufl import element as basix_element, mixed_element as basix_mixed_element


ScalarType = PETSc.ScalarType


def _manufactured_u_expr(x):
    return ufl.as_vector(
        [
            ufl.pi * ufl.cos(ufl.pi * x[1]) * ufl.sin(ufl.pi * x[0]),
            -ufl.pi * ufl.cos(ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1]),
        ]
    )


def _forcing_expr(msh, nu):
    x = ufl.SpatialCoordinate(msh)
    uex = _manufactured_u_expr(x)
    pex = 0.0 * x[0]
    conv = ufl.grad(uex) * uex
    diff = -nu * ufl.div(ufl.grad(uex))
    gradp = ufl.grad(pex)
    return conv + diff + gradp


def _sample_function_on_grid(u_func, msh, nx, ny, bbox):
    xmin, xmax, ymin, ymax = bbox
    xs = np.linspace(xmin, xmax, nx)
    ys = np.linspace(ymin, ymax, ny)
    XX, YY = np.meshgrid(xs, ys, indexing="xy")
    pts2 = np.column_stack([XX.ravel(), YY.ravel()])
    pts3 = np.zeros((pts2.shape[0], 3), dtype=np.float64)
    pts3[:, :2] = pts2

    tree = geometry.bb_tree(msh, msh.topology.dim)
    cell_candidates = geometry.compute_collisions_points(tree, pts3)
    colliding_cells = geometry.compute_colliding_cells(msh, cell_candidates, pts3)

    local_vals = np.full((pts3.shape[0], msh.geometry.dim), np.nan, dtype=np.float64)
    points_on_proc = []
    cells_on_proc = []
    idx_map = []
    for i in range(pts3.shape[0]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(pts3[i])
            cells_on_proc.append(links[0])
            idx_map.append(i)

    if len(points_on_proc) > 0:
        vals = u_func.eval(np.array(points_on_proc, dtype=np.float64), np.array(cells_on_proc, dtype=np.int32))
        local_vals[np.array(idx_map, dtype=np.int32), :] = vals

    gathered = msh.comm.gather(local_vals, root=0)
    if msh.comm.rank == 0:
        merged = np.full_like(local_vals, np.nan)
        for arr in gathered:
            mask = np.isnan(merged[:, 0]) & ~np.isnan(arr[:, 0])
            merged[mask, :] = arr[mask, :]
        mag = np.linalg.norm(merged, axis=1).reshape(ny, nx)
    else:
        mag = None
    mag = msh.comm.bcast(mag, root=0)
    return mag


def _compute_l2_error(u_sol, V, msh):
    x = ufl.SpatialCoordinate(msh)
    uex = _manufactured_u_expr(x)
    err_form = fem.form(ufl.inner(u_sol - uex, u_sol - uex) * ufl.dx)
    norm_form = fem.form(ufl.inner(uex, uex) * ufl.dx)
    err_local = fem.assemble_scalar(err_form)
    norm_local = fem.assemble_scalar(norm_form)
    err = np.sqrt(msh.comm.allreduce(err_local, op=MPI.SUM))
    nrm = np.sqrt(msh.comm.allreduce(norm_local, op=MPI.SUM))
    return err, err / max(nrm, 1e-16)


def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD

    nu = float(case_spec.get("pde", {}).get("nu", case_spec.get("viscosity", 0.02)))
    out_grid = case_spec["output"]["grid"]
    nx_out = int(out_grid["nx"])
    ny_out = int(out_grid["ny"])
    bbox = out_grid["bbox"]

    mesh_resolution = 48
    degree_u = 2
    degree_p = 1
    newton_rtol = 1.0e-9
    newton_atol = 1.0e-10
    newton_max_it = 30

    msh = mesh.create_unit_square(comm, mesh_resolution, mesh_resolution, cell_type=mesh.CellType.triangle)
    gdim = msh.geometry.dim

    vel_el = basix_element("Lagrange", msh.topology.cell_name(), degree_u, shape=(gdim,))
    pres_el = basix_element("Lagrange", msh.topology.cell_name(), degree_p)
    W = fem.functionspace(msh, basix_mixed_element([vel_el, pres_el]))
    V, _ = W.sub(0).collapse()
    Q, _ = W.sub(1).collapse()

    x = ufl.SpatialCoordinate(msh)
    u_exact = _manufactured_u_expr(x)
    f_expr = _forcing_expr(msh, nu)

    w = fem.Function(W)
    w.name = "w"
    u, p = ufl.split(w)
    v, q = ufl.TestFunctions(W)

    def eps(u_):
        return ufl.sym(ufl.grad(u_))

    F = (
        2.0 * nu * ufl.inner(eps(u), eps(v)) * ufl.dx
        + ufl.inner(ufl.grad(u) * u, v) * ufl.dx
        - ufl.inner(f_expr, v) * ufl.dx
        - p * ufl.div(v) * ufl.dx
        + ufl.div(u) * q * ufl.dx
    )
    J = ufl.derivative(F, w)

    u_bc_fun = fem.Function(V)
    u_bc_expr = fem.Expression(u_exact, V.element.interpolation_points)
    u_bc_fun.interpolate(u_bc_expr)

    fdim = msh.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(msh, fdim, lambda xx: np.ones(xx.shape[1], dtype=bool))
    udofs = fem.locate_dofs_topological((W.sub(0), V), fdim, boundary_facets)
    bc_u = fem.dirichletbc(u_bc_fun, udofs, W.sub(0))

    pdofs = fem.locate_dofs_geometrical((W.sub(1), Q), lambda xx: np.isclose(xx[0], 0.0) & np.isclose(xx[1], 0.0))
    p0_fun = fem.Function(Q)
    p0_fun.x.array[:] = 0.0
    bc_p = fem.dirichletbc(p0_fun, pdofs, W.sub(1))
    bcs = [bc_u, bc_p]

    # Picard/Stokes-like initialization
    wk = fem.Function(W)
    uk, pk = ufl.split(wk)
    u_trial, p_trial = ufl.TrialFunctions(W)
    v_pic, q_pic = ufl.TestFunctions(W)

    a_pic = (
        2.0 * nu * ufl.inner(ufl.sym(ufl.grad(u_trial)), ufl.sym(ufl.grad(v_pic))) * ufl.dx
        + ufl.inner(ufl.grad(u_trial) * uk, v_pic) * ufl.dx
        - p_trial * ufl.div(v_pic) * ufl.dx
        + ufl.div(u_trial) * q_pic * ufl.dx
    )
    L_pic = ufl.inner(f_expr, v_pic) * ufl.dx

    picard_iterations = 0
    linear_iterations = 0
    for _ in range(6):
        problem_pic = petsc.LinearProblem(
            a_pic,
            L_pic,
            bcs=bcs,
            petsc_options_prefix="ns_picard_",
            petsc_options={
                "ksp_type": "gmres",
                "pc_type": "lu",
                "ksp_rtol": 1.0e-10,
            },
        )
        w_new = problem_pic.solve()
        linear_iterations += int(problem_pic.solver.getIterationNumber())
        diff = np.linalg.norm(w_new.x.array - wk.x.array)
        base = max(np.linalg.norm(w_new.x.array), 1.0e-14)
        wk.x.array[:] = w_new.x.array
        wk.x.scatter_forward()
        picard_iterations += 1
        if diff / base < 1.0e-8:
            break

    w.x.array[:] = wk.x.array
    w.x.scatter_forward()

    snes_opts = {
        "snes_type": "newtonls",
        "snes_linesearch_type": "bt",
        "snes_rtol": newton_rtol,
        "snes_atol": newton_atol,
        "snes_max_it": newton_max_it,
        "ksp_type": "gmres",
        "pc_type": "lu",
        "ksp_rtol": 1.0e-10,
    }

    problem = petsc.NonlinearProblem(
        F,
        w,
        bcs=bcs,
        J=J,
        petsc_options_prefix="ns_",
        petsc_options=snes_opts,
    )
    w = problem.solve()
    w.x.scatter_forward()

    try:
        snes = problem.solver
        nonlinear_its = int(snes.getIterationNumber())
        linear_iterations += int(snes.getLinearSolveIterations())
        ksp_type = snes.getKSP().getType()
        pc_type = snes.getKSP().getPC().getType()
    except Exception:
        nonlinear_its = picard_iterations
        ksp_type = "gmres"
        pc_type = "lu"

    u_sol = w.sub(0).collapse()
    u_grid = _sample_function_on_grid(u_sol, msh, nx_out, ny_out, bbox)

    l2_err, rel_l2_err = _compute_l2_error(u_sol, V, msh)

    solver_info = {
        "mesh_resolution": mesh_resolution,
        "element_degree": degree_u,
        "ksp_type": str(ksp_type),
        "pc_type": str(pc_type),
        "rtol": 1.0e-10,
        "iterations": int(linear_iterations),
        "nonlinear_iterations": [int(nonlinear_its)],
        "l2_error": float(l2_err),
        "rel_l2_error": float(rel_l2_err),
        "viscosity": float(nu),
    }

    return {"u": u_grid, "solver_info": solver_info}
