import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
from basix.ufl import element as basix_element, mixed_element as basix_mixed_element
import ufl
from petsc4py import PETSc


def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD
    nu_val = 0.08

    N = 160
    msh = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
    gdim = msh.geometry.dim

    vel_el = basix_element("Lagrange", msh.topology.cell_name(), 2, shape=(gdim,))
    pres_el = basix_element("Lagrange", msh.topology.cell_name(), 1)
    W = fem.functionspace(msh, basix_mixed_element([vel_el, pres_el]))

    V, _ = W.sub(0).collapse()
    Q, _ = W.sub(1).collapse()

    w = fem.Function(W)
    (u, p) = ufl.split(w)
    (v, q) = ufl.TestFunctions(W)

    nu = fem.Constant(msh, PETSc.ScalarType(nu_val))
    f = fem.Constant(msh, np.zeros(gdim, dtype=PETSc.ScalarType))

    F = (
        nu * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
        + ufl.inner(ufl.grad(u) * u, v) * ufl.dx
        - p * ufl.div(v) * ufl.dx
        + ufl.div(u) * q * ufl.dx
        - ufl.inner(f, v) * ufl.dx
    )

    fdim = msh.topology.dim - 1

    # Lid (y=1): u=(1,0)
    def on_lid(x):
        return np.isclose(x[1], 1.0)

    def on_walls(x):
        return np.isclose(x[0], 0.0) | np.isclose(x[0], 1.0) | np.isclose(x[1], 0.0)

    lid_facets = mesh.locate_entities_boundary(msh, fdim, on_lid)
    wall_facets = mesh.locate_entities_boundary(msh, fdim, on_walls)

    u_lid = fem.Function(V)
    u_lid.interpolate(lambda x: np.vstack([np.ones(x.shape[1]), np.zeros(x.shape[1])]))
    dofs_lid = fem.locate_dofs_topological((W.sub(0), V), fdim, lid_facets)
    bc_lid = fem.dirichletbc(u_lid, dofs_lid, W.sub(0))

    u_wall = fem.Function(V)
    u_wall.x.array[:] = 0.0
    dofs_wall = fem.locate_dofs_topological((W.sub(0), V), fdim, wall_facets)
    bc_wall = fem.dirichletbc(u_wall, dofs_wall, W.sub(0))

    # Pressure pin at (0,0)
    p_dofs = fem.locate_dofs_geometrical(
        (W.sub(1), Q),
        lambda x: np.isclose(x[0], 0.0) & np.isclose(x[1], 0.0),
    )
    p0f = fem.Function(Q)
    p0f.x.array[:] = 0.0
    bc_p = fem.dirichletbc(p0f, p_dofs, W.sub(1))

    bcs = [bc_wall, bc_lid, bc_p]

    J = ufl.derivative(F, w)

    petsc_options = {
        "snes_type": "newtonls",
        "snes_linesearch_type": "bt",
        "snes_rtol": 1e-9,
        "snes_atol": 1e-10,
        "snes_max_it": 50,
        "ksp_type": "preonly",
        "pc_type": "lu",
        "pc_factor_mat_solver_type": "mumps",
    }

    problem = petsc.NonlinearProblem(
        F, w, bcs=bcs, J=J,
        petsc_options_prefix="ns_lid_",
        petsc_options=petsc_options,
    )

    w_h = problem.solve()
    w.x.scatter_forward()

    try:
        snes = problem.solver
        newton_its = snes.getIterationNumber()
        ksp_its = snes.getKSP().getIterationNumber()
    except Exception:
        newton_its = -1
        ksp_its = -1

    # Extract velocity
    u_sol = w.sub(0).collapse()

    # Sample on grid
    grid = case_spec["output"]["grid"]
    nx, ny = grid["nx"], grid["ny"]
    bbox = grid["bbox"]
    xs = np.linspace(bbox[0], bbox[1], nx)
    ys = np.linspace(bbox[2], bbox[3], ny)
    XX, YY = np.meshgrid(xs, ys)
    pts = np.c_[XX.ravel(), YY.ravel(), np.zeros(nx * ny)]

    # Clamp slightly inside to avoid boundary misses
    eps = 1e-12
    pts[:, 0] = np.clip(pts[:, 0], bbox[0] + eps, bbox[1] - eps)
    pts[:, 1] = np.clip(pts[:, 1], bbox[2] + eps, bbox[3] - eps)

    tree = geometry.bb_tree(msh, msh.topology.dim)
    cell_candidates = geometry.compute_collisions_points(tree, pts)
    colliding = geometry.compute_colliding_cells(msh, cell_candidates, pts)

    points_on_proc = []
    cells_on_proc = []
    idx_map = []
    for i in range(pts.shape[0]):
        links = colliding.links(i)
        if len(links) > 0:
            points_on_proc.append(pts[i])
            cells_on_proc.append(links[0])
            idx_map.append(i)

    mag = np.zeros(pts.shape[0])
    if len(points_on_proc) > 0:
        vals = u_sol.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        m = np.linalg.norm(vals, axis=1)
        mag[idx_map] = m

    u_grid = mag.reshape(ny, nx)

    return {
        "u": u_grid,
        "solver_info": {
            "mesh_resolution": N,
            "element_degree": 2,
            "ksp_type": "preonly",
            "pc_type": "lu",
            "rtol": 1e-9,
            "iterations": int(ksp_its) if ksp_its >= 0 else 0,
            "nonlinear_iterations": [int(newton_its)] if newton_its >= 0 else [0],
        },
    }
