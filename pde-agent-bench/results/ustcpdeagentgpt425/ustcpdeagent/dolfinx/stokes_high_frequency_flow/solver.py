import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
import ufl
from dolfinx import mesh, fem, geometry
from dolfinx.fem.petsc import LinearProblem
from basix.ufl import element as basix_element, mixed_element as basix_mixed_element

ScalarType = PETSc.ScalarType


def _u_exact(x):
    return ufl.as_vector(
        (
            2 * ufl.pi * ufl.cos(2 * ufl.pi * x[1]) * ufl.sin(2 * ufl.pi * x[0]),
            -2 * ufl.pi * ufl.cos(2 * ufl.pi * x[0]) * ufl.sin(2 * ufl.pi * x[1]),
        )
    )


def _p_exact(x):
    return ufl.sin(2 * ufl.pi * x[0]) * ufl.cos(2 * ufl.pi * x[1])


def _f_exact(msh):
    x = ufl.SpatialCoordinate(msh)
    return ufl.as_vector(
        (
            8 * ufl.pi**3 * ufl.cos(2 * ufl.pi * x[1]) * ufl.sin(2 * ufl.pi * x[0])
            + 2 * ufl.pi * ufl.cos(2 * ufl.pi * x[0]) * ufl.cos(2 * ufl.pi * x[1]),
            -8 * ufl.pi**3 * ufl.cos(2 * ufl.pi * x[0]) * ufl.sin(2 * ufl.pi * x[1])
            - 2 * ufl.pi * ufl.sin(2 * ufl.pi * x[0]) * ufl.sin(2 * ufl.pi * x[1]),
        )
    )


def _sample_velocity_magnitude(u_fun, nx, ny, bbox):
    msh = u_fun.function_space.mesh
    xmin, xmax, ymin, ymax = bbox
    xs = np.linspace(xmin, xmax, nx)
    ys = np.linspace(ymin, ymax, ny)
    XX, YY = np.meshgrid(xs, ys, indexing="xy")
    pts = np.c_[XX.ravel(), YY.ravel(), np.zeros(nx * ny)]

    tree = geometry.bb_tree(msh, msh.topology.dim)
    candidates = geometry.compute_collisions_points(tree, pts)
    colliding = geometry.compute_colliding_cells(msh, candidates, pts)

    local_vals = np.full((pts.shape[0], msh.geometry.dim), np.nan, dtype=np.float64)
    points_on_proc = []
    cells = []
    ids = []
    for i in range(pts.shape[0]):
        links = colliding.links(i)
        if len(links) > 0:
            points_on_proc.append(pts[i])
            cells.append(links[0])
            ids.append(i)

    if points_on_proc:
        vals = u_fun.eval(np.array(points_on_proc, dtype=np.float64), np.array(cells, dtype=np.int32))
        local_vals[np.array(ids, dtype=np.int32)] = vals

    gathered = msh.comm.gather(local_vals, root=0)
    if msh.comm.rank == 0:
        merged = gathered[0]
        for arr in gathered[1:]:
            mask = np.isnan(merged[:, 0]) & ~np.isnan(arr[:, 0])
            merged[mask] = arr[mask]
        mag = np.linalg.norm(merged, axis=1).reshape(ny, nx)
    else:
        mag = None
    return msh.comm.bcast(mag, root=0)


def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD
    n = 48
    msh = mesh.create_unit_square(comm, n, n, cell_type=mesh.CellType.triangle)
    gdim = msh.geometry.dim

    vel_el = basix_element("Lagrange", msh.topology.cell_name(), 2, shape=(gdim,))
    pre_el = basix_element("Lagrange", msh.topology.cell_name(), 1)
    W = fem.functionspace(msh, basix_mixed_element([vel_el, pre_el]))
    V, _ = W.sub(0).collapse()
    Q, _ = W.sub(1).collapse()

    u, p = ufl.TrialFunctions(W)
    v, q = ufl.TestFunctions(W)
    x = ufl.SpatialCoordinate(msh)
    nu = 1.0
    f = _f_exact(msh)

    a = (
        2.0 * nu * ufl.inner(ufl.sym(ufl.grad(u)), ufl.sym(ufl.grad(v))) * ufl.dx
        - p * ufl.div(v) * ufl.dx
        - q * ufl.div(u) * ufl.dx
    )
    L = ufl.inner(f, v) * ufl.dx

    fdim = msh.topology.dim - 1
    facets = mesh.locate_entities_boundary(msh, fdim, lambda xx: np.ones(xx.shape[1], dtype=bool))
    u_bc = fem.Function(V)
    u_bc.interpolate(fem.Expression(_u_exact(x), V.element.interpolation_points))
    u_dofs = fem.locate_dofs_topological((W.sub(0), V), fdim, facets)
    bc_u = fem.dirichletbc(u_bc, u_dofs, W.sub(0))

    p_dofs = fem.locate_dofs_geometrical((W.sub(1), Q), lambda xx: np.isclose(xx[0], 0.0) & np.isclose(xx[1], 0.0))
    p_bc_fun = fem.Function(Q)
    p_bc_fun.x.array[:] = 0.0
    bc_p = fem.dirichletbc(p_bc_fun, p_dofs, W.sub(1))

    problem = LinearProblem(
        a,
        L,
        bcs=[bc_u, bc_p],
        petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
        petsc_options_prefix="stokes_",
    )
    wh = problem.solve()
    wh.x.scatter_forward()

    uh = fem.Function(V)
    uh.interpolate(wh.sub(0))
    u_ex = _u_exact(x)
    err2 = fem.assemble_scalar(fem.form(ufl.inner(uh - u_ex, uh - u_ex) * ufl.dx))
    norm2 = fem.assemble_scalar(fem.form(ufl.inner(u_ex, u_ex) * ufl.dx))
    err2 = comm.allreduce(err2, op=MPI.SUM)
    norm2 = comm.allreduce(norm2, op=MPI.SUM)
    rel_l2 = float(np.sqrt(err2 / norm2))

    grid = case_spec["output"]["grid"]
    u_grid = _sample_velocity_magnitude(uh, int(grid["nx"]), int(grid["ny"]), grid["bbox"])

    return {
        "u": np.asarray(u_grid, dtype=np.float64).reshape(int(grid["ny"]), int(grid["nx"])),
        "solver_info": {
            "mesh_resolution": n,
            "element_degree": 2,
            "ksp_type": "preonly",
            "pc_type": "lu",
            "rtol": 1e-5,
            "iterations": 0,
            "manufactured_velocity_rel_l2_error": rel_l2,
        },
    }


if __name__ == "__main__":
    case_spec = {"output": {"grid": {"nx": 32, "ny": 32, "bbox": [0.0, 1.0, 0.0, 1.0]}}}
    out = solve(case_spec)
    print(out["u"].shape)
    print(out["solver_info"])
