import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
import ufl
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc as fpetsc
from basix.ufl import element as basix_element, mixed_element as basix_mixed_element

# ```DIAGNOSIS
# equation_type: stokes
# spatial_dim: 2
# domain_geometry: rectangle
# unknowns: vector+scalar
# coupling: saddle_point
# linearity: linear
# time_dependence: steady
# stiffness: N/A
# dominant_physics: diffusion
# peclet_or_reynolds: low
# solution_regularity: smooth
# bc_type: mixed
# special_notes: pressure_pinning
# ```

# ```METHOD
# spatial_method: fem
# element_or_basis: Taylor-Hood_P2P1
# stabilization: none
# time_method: none
# nonlinear_solver: none
# linear_solver: direct_lu
# preconditioner: none
# special_treatment: pressure_pinning
# pde_skill: stokes
# ```

ScalarType = PETSc.ScalarType


def _sample_velocity_magnitude(u_func, bbox, nx, ny):
    msh = u_func.function_space.mesh
    xs = np.linspace(float(bbox[0]), float(bbox[1]), int(nx))
    ys = np.linspace(float(bbox[2]), float(bbox[3]), int(ny))
    xx, yy = np.meshgrid(xs, ys, indexing="xy")
    pts = np.zeros((nx * ny, 3), dtype=np.float64)
    pts[:, 0] = xx.ravel()
    pts[:, 1] = yy.ravel()

    tree = geometry.bb_tree(msh, msh.topology.dim)
    candidates = geometry.compute_collisions_points(tree, pts)
    colliding = geometry.compute_colliding_cells(msh, candidates, pts)

    local_vals = np.full((pts.shape[0], msh.geometry.dim), -1.0e300, dtype=np.float64)
    points_on_proc = []
    cells_on_proc = []
    ids = []
    for i in range(pts.shape[0]):
        links = colliding.links(i)
        if len(links) > 0:
            points_on_proc.append(pts[i])
            cells_on_proc.append(links[0])
            ids.append(i)

    if points_on_proc:
        vals = np.asarray(
            u_func.eval(
                np.asarray(points_on_proc, dtype=np.float64),
                np.asarray(cells_on_proc, dtype=np.int32),
            ),
            dtype=np.float64,
        )
        if vals.ndim == 1:
            vals = vals.reshape(-1, 1)
        local_vals[np.asarray(ids, dtype=np.int32), : vals.shape[1]] = vals

    global_vals = np.empty_like(local_vals)
    msh.comm.Allreduce(local_vals, global_vals, op=MPI.MAX)
    missing = global_vals[:, 0] < -1.0e299
    if np.any(missing):
        global_vals[missing, :] = 0.0

    return np.linalg.norm(global_vals[:, :2], axis=1).reshape((ny, nx))


def _build_case(mesh_resolution):
    comm = MPI.COMM_WORLD
    msh = mesh.create_unit_square(comm, mesh_resolution, mesh_resolution, cell_type=mesh.CellType.triangle)
    gdim = msh.geometry.dim
    cell_name = msh.topology.cell_name()

    vel_el = basix_element("Lagrange", cell_name, 2, shape=(gdim,))
    pre_el = basix_element("Lagrange", cell_name, 1)
    W = fem.functionspace(msh, basix_mixed_element([vel_el, pre_el]))
    V, _ = W.sub(0).collapse()
    Q, _ = W.sub(1).collapse()

    nu = ScalarType(0.4)
    f = fem.Constant(msh, np.array([1.0, 0.0], dtype=np.float64))

    (u, p) = ufl.TrialFunctions(W)
    (v, q) = ufl.TestFunctions(W)

    def eps(w):
        return ufl.sym(ufl.grad(w))

    a = (
        2.0 * nu * ufl.inner(eps(u), eps(v)) * ufl.dx
        - p * ufl.div(v) * ufl.dx
        + q * ufl.div(u) * ufl.dx
    )
    L = ufl.inner(f, v) * ufl.dx

    fdim = msh.topology.dim - 1
    left = mesh.locate_entities_boundary(msh, fdim, lambda x: np.isclose(x[0], 0.0))
    bottom = mesh.locate_entities_boundary(msh, fdim, lambda x: np.isclose(x[1], 0.0))
    top = mesh.locate_entities_boundary(msh, fdim, lambda x: np.isclose(x[1], 1.0))
    wall_facets = np.unique(np.concatenate([left, bottom, top]).astype(np.int32))

    u_zero = fem.Function(V)
    u_zero.x.array[:] = 0.0
    dofs_u = fem.locate_dofs_topological((W.sub(0), V), fdim, wall_facets)
    bc_u = fem.dirichletbc(u_zero, dofs_u, W.sub(0))

    p_pin_dofs = fem.locate_dofs_geometrical(
        (W.sub(1), Q), lambda x: np.isclose(x[0], 0.0) & np.isclose(x[1], 0.0)
    )
    p_zero = fem.Function(Q)
    p_zero.x.array[:] = 0.0
    bc_p = fem.dirichletbc(p_zero, p_pin_dofs, W.sub(1))

    prefix = f"stokes_{mesh_resolution}_"
    opts = {
        "ksp_type": "preonly",
        "pc_type": "lu",
        "pc_factor_mat_solver_type": "mumps",
    }
    problem = fpetsc.LinearProblem(a, L, bcs=[bc_u, bc_p], petsc_options_prefix=prefix, petsc_options=opts)
    wh = problem.solve()
    wh.x.scatter_forward()
    uh = wh.sub(0).collapse()

    div_l2_local = fem.assemble_scalar(fem.form(ufl.inner(ufl.div(uh), ufl.div(uh)) * ufl.dx))
    div_l2 = np.sqrt(max(comm.allreduce(div_l2_local, op=MPI.SUM), 0.0))

    flux_local = fem.assemble_scalar(fem.form(ufl.dot(uh, ufl.FacetNormal(msh)) * ufl.ds))
    boundary_flux = comm.allreduce(flux_local, op=MPI.SUM)

    ksp = problem.solver
    return uh, {
        "mesh_resolution": int(mesh_resolution),
        "element_degree": 2,
        "ksp_type": str(ksp.getType()),
        "pc_type": str(ksp.getPC().getType()),
        "rtol": float(ksp.getTolerances()[0]),
        "iterations": int(ksp.getIterationNumber()),
        "verification": {
            "divergence_l2": float(div_l2),
            "boundary_flux": float(boundary_flux),
        },
    }


def solve(case_spec: dict) -> dict:
    output = case_spec.get("output", {})
    grid = output.get("grid", {})
    nx = int(grid.get("nx", 64))
    ny = int(grid.get("ny", 64))
    bbox = grid.get("bbox", [0.0, 1.0, 0.0, 1.0])

    mesh_resolution = max(40, min(64, max(nx, ny)))
    uh, solver_info = _build_case(mesh_resolution)
    u_grid = _sample_velocity_magnitude(uh, bbox, nx, ny)
    return {"u": u_grid, "solver_info": solver_info}


if __name__ == "__main__":
    case_spec = {
        "pde": {"type": "stokes", "time": None},
        "output": {"grid": {"nx": 32, "ny": 32, "bbox": [0.0, 1.0, 0.0, 1.0]}},
    }
    out = solve(case_spec)
    if MPI.COMM_WORLD.rank == 0:
        print(out["u"].shape)
        print(out["solver_info"])
        print(float(np.nanmax(out["u"])))
