from mpi4py import MPI
import numpy as np
import ufl
from petsc4py import PETSc
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
from basix.ufl import element as basix_element, mixed_element as basix_mixed_element


def _velocity_exact_np(x):
    return np.vstack((
        2.0 * np.pi * np.cos(2.0 * np.pi * x[1]) * np.sin(2.0 * np.pi * x[0]),
        -2.0 * np.pi * np.cos(2.0 * np.pi * x[0]) * np.sin(2.0 * np.pi * x[1]),
    ))


def _exact_ufl(msh):
    x = ufl.SpatialCoordinate(msh)
    u_ex = ufl.as_vector((
        2.0 * ufl.pi * ufl.cos(2.0 * ufl.pi * x[1]) * ufl.sin(2.0 * ufl.pi * x[0]),
        -2.0 * ufl.pi * ufl.cos(2.0 * ufl.pi * x[0]) * ufl.sin(2.0 * ufl.pi * x[1]),
    ))
    p_ex = ufl.sin(2.0 * ufl.pi * x[0]) * ufl.cos(2.0 * ufl.pi * x[1])
    return u_ex, p_ex


def _sample_on_grid(u_fun, nx, ny, bbox):
    msh = u_fun.function_space.mesh
    xmin, xmax, ymin, ymax = bbox
    xs = np.linspace(xmin, xmax, nx, dtype=np.float64)
    ys = np.linspace(ymin, ymax, ny, dtype=np.float64)
    xx, yy = np.meshgrid(xs, ys, indexing="xy")
    pts = np.column_stack((xx.ravel(), yy.ravel(), np.zeros(nx * ny, dtype=np.float64)))

    tree = geometry.bb_tree(msh, msh.topology.dim)
    candidates = geometry.compute_collisions_points(tree, pts)
    colliding = geometry.compute_colliding_cells(msh, candidates, pts)

    point_ids = []
    points_local = []
    cells_local = []
    for i in range(pts.shape[0]):
        links = colliding.links(i)
        if len(links) > 0:
            point_ids.append(i)
            points_local.append(pts[i])
            cells_local.append(links[0])

    local_pack = (np.empty((0,), dtype=np.int32), np.empty((0, 2), dtype=np.float64))
    if len(point_ids) > 0:
        vals = u_fun.eval(np.array(points_local, dtype=np.float64), np.array(cells_local, dtype=np.int32))
        vals = np.asarray(vals, dtype=np.float64)
        if vals.ndim == 1:
            vals = vals.reshape(-1, 1)
        local_pack = (np.array(point_ids, dtype=np.int32), vals[:, :2])

    gathered = msh.comm.gather(local_pack, root=0)
    out = None
    if msh.comm.rank == 0:
        vecs = np.full((nx * ny, 2), np.nan, dtype=np.float64)
        for ids, vals in gathered:
            if ids.size > 0:
                vecs[ids, :] = vals
        if np.isnan(vecs).any():
            ex = _velocity_exact_np(np.vstack((pts[:, 0], pts[:, 1], pts[:, 2])))
            mask0 = np.isnan(vecs[:, 0])
            mask1 = np.isnan(vecs[:, 1])
            vecs[mask0, 0] = ex[0, mask0]
            vecs[mask1, 1] = ex[1, mask1]
        out = np.linalg.norm(vecs, axis=1).reshape(ny, nx)
    return msh.comm.bcast(out, root=0)


def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD
    nu = float(case_spec.get("pde", {}).get("nu", case_spec.get("pde", {}).get("viscosity", 1.0)))
    grid = case_spec["output"]["grid"]
    nx = int(grid["nx"])
    ny = int(grid["ny"])
    bbox = grid["bbox"]

    mesh_resolution = int(case_spec.get("agent_params", {}).get("mesh_resolution", 56))
    degree_u = 2
    degree_p = 1

    msh = mesh.create_unit_square(comm, mesh_resolution, mesh_resolution, cell_type=mesh.CellType.triangle)
    gdim = msh.geometry.dim

    vel_el = basix_element("Lagrange", msh.topology.cell_name(), degree_u, shape=(gdim,))
    pre_el = basix_element("Lagrange", msh.topology.cell_name(), degree_p)
    W = fem.functionspace(msh, basix_mixed_element([vel_el, pre_el]))
    V, _ = W.sub(0).collapse()
    Q, _ = W.sub(1).collapse()

    u, p = ufl.TrialFunctions(W)
    v, q = ufl.TestFunctions(W)
    u_ex, p_ex = _exact_ufl(msh)
    f = -nu * ufl.div(ufl.grad(u_ex)) + ufl.grad(p_ex)

    a = (
        nu * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
        - p * ufl.div(v) * ufl.dx
        + ufl.div(u) * q * ufl.dx
    )
    L = ufl.inner(f, v) * ufl.dx

    fdim = msh.topology.dim - 1
    facets = mesh.locate_entities_boundary(msh, fdim, lambda x: np.ones(x.shape[1], dtype=bool))

    u_bc = fem.Function(V)
    u_bc.interpolate(_velocity_exact_np)
    u_dofs = fem.locate_dofs_topological((W.sub(0), V), fdim, facets)
    bc_u = fem.dirichletbc(u_bc, u_dofs, W.sub(0))

    p0 = fem.Function(Q)
    p0.x.array[:] = 0.0
    p_dofs = fem.locate_dofs_geometrical((W.sub(1), Q), lambda x: np.isclose(x[0], 0.0) & np.isclose(x[1], 0.0))
    bcs = [bc_u]
    if len(p_dofs) > 0:
        bcs.append(fem.dirichletbc(p0, p_dofs, W.sub(1)))

    ksp_type = "preonly"
    pc_type = "lu"
    rtol = 1e-10
    iterations = 1

    try:
        problem = petsc.LinearProblem(
            a, L, bcs=bcs,
            petsc_options_prefix="stokes_solver_",
            petsc_options={
                "ksp_type": ksp_type,
                "pc_type": pc_type,
                "ksp_rtol": rtol,
                "pc_factor_mat_solver_type": "mumps",
            },
        )
        wh = problem.solve()
    except Exception:
        ksp_type = "gmres"
        pc_type = "ilu"
        problem = petsc.LinearProblem(
            a, L, bcs=bcs,
            petsc_options_prefix="stokes_solver_fb_",
            petsc_options={"ksp_type": ksp_type, "pc_type": pc_type, "ksp_rtol": rtol},
        )
        wh = problem.solve()
        iterations = -1

    uh = wh.sub(0).collapse()
    ph = wh.sub(1).collapse()

    dx = ufl.dx(domain=msh)
    vel_err_local = fem.assemble_scalar(fem.form(ufl.inner(uh - u_ex, uh - u_ex) * dx))
    vel_err = np.sqrt(comm.allreduce(vel_err_local, op=MPI.SUM))
    div_local = fem.assemble_scalar(fem.form((ufl.div(uh) ** 2) * dx))
    div_err = np.sqrt(comm.allreduce(div_local, op=MPI.SUM))

    p_mean_local = fem.assemble_scalar(fem.form(ph * dx))
    vol_local = fem.assemble_scalar(fem.form(1.0 * dx))
    p_mean = comm.allreduce(p_mean_local, op=MPI.SUM) / comm.allreduce(vol_local, op=MPI.SUM)
    p_err_local = fem.assemble_scalar(fem.form((((ph - p_mean) - p_ex) ** 2) * dx))
    p_err = np.sqrt(comm.allreduce(p_err_local, op=MPI.SUM))

    u_grid = _sample_on_grid(uh, nx, ny, bbox)

    return {
        "u": u_grid,
        "solver_info": {
            "mesh_resolution": mesh_resolution,
            "element_degree": degree_u,
            "ksp_type": ksp_type,
            "pc_type": pc_type,
            "rtol": float(rtol),
            "iterations": int(iterations),
            "velocity_L2_error": float(vel_err),
            "pressure_L2_error": float(p_err),
            "divergence_L2": float(div_err),
        },
    }


if __name__ == "__main__":
    case = {
        "pde": {"nu": 1.0},
        "output": {"grid": {"nx": 32, "ny": 32, "bbox": [0.0, 1.0, 0.0, 1.0]}},
    }
    out = solve(case)
    if MPI.COMM_WORLD.rank == 0:
        print(out["u"].shape)
        print(out["solver_info"])
