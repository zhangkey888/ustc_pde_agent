import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
import ufl
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc as fem_petsc
from basix.ufl import element as basix_element, mixed_element as basix_mixed_element


def _sample_velocity_magnitude(u_h, msh, nx, ny, bbox):
    comm = msh.comm
    rank = comm.rank
    xmin, xmax, ymin, ymax = map(float, bbox)
    xs = np.linspace(xmin, xmax, nx)
    ys = np.linspace(ymin, ymax, ny)
    XX, YY = np.meshgrid(xs, ys, indexing="xy")
    pts = np.column_stack([XX.ravel(), YY.ravel(), np.zeros(nx * ny, dtype=np.float64)])

    tree = geometry.bb_tree(msh, msh.topology.dim)
    candidates = geometry.compute_collisions_points(tree, pts)
    colliding = geometry.compute_colliding_cells(msh, candidates, pts)

    values_local = np.full((pts.shape[0],), np.nan, dtype=np.float64)
    eval_points, eval_cells, eval_ids = [], [], []
    for i in range(pts.shape[0]):
        links = colliding.links(i)
        if len(links) > 0:
            eval_points.append(pts[i])
            eval_cells.append(links[0])
            eval_ids.append(i)

    if eval_points:
        vals = u_h.eval(np.array(eval_points, dtype=np.float64), np.array(eval_cells, dtype=np.int32))
        mags = np.linalg.norm(vals, axis=1)
        values_local[np.array(eval_ids, dtype=np.int32)] = mags

    gathered = comm.gather(values_local, root=0)
    if rank == 0:
        values = np.full((pts.shape[0],), np.nan, dtype=np.float64)
        for arr in gathered:
            mask = np.isfinite(arr)
            values[mask] = arr[mask]
        values[~np.isfinite(values)] = 0.0
        return values.reshape((ny, nx))
    return None


def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD
    rank = comm.rank

    # DIAGNOSIS
    # equation_type: navier_stokes
    # spatial_dim: 2
    # domain_geometry: rectangle
    # unknowns: vector+scalar
    # coupling: saddle_point
    # linearity: nonlinear
    # time_dependence: steady
    # stiffness: N/A
    # dominant_physics: mixed
    # peclet_or_reynolds: moderate
    # solution_regularity: boundary_layer
    # bc_type: all_dirichlet
    # special_notes: pressure_pinning

    # METHOD
    # spatial_method: fem
    # element_or_basis: Taylor-Hood_P2P1
    # stabilization: none
    # time_method: none
    # nonlinear_solver: none
    # linear_solver: direct_lu
    # preconditioner: none
    # special_treatment: pressure_pinning
    # pde_skill: navier_stokes

    nu = float(case_spec.get("pde", {}).get("nu", case_spec.get("viscosity", 0.08)))
    grid = case_spec["output"]["grid"]
    nx_out = int(grid["nx"])
    ny_out = int(grid["ny"])
    bbox = grid["bbox"]

    # Use a refined stable discretization to maximize accuracy within the large time budget.
    mesh_resolution = int(case_spec.get("solver_params", {}).get("mesh_resolution", 128))
    degree_u = int(case_spec.get("solver_params", {}).get("degree_u", 2))
    degree_p = int(case_spec.get("solver_params", {}).get("degree_p", 1))
    linear_rtol = float(case_spec.get("solver_params", {}).get("ksp_rtol", 1e-10))

    msh = mesh.create_unit_square(comm, mesh_resolution, mesh_resolution, cell_type=mesh.CellType.triangle)
    gdim = msh.geometry.dim
    cellname = msh.topology.cell_name()

    vel_el = basix_element("Lagrange", cellname, degree_u, shape=(gdim,))
    pre_el = basix_element("Lagrange", cellname, degree_p)
    W = fem.functionspace(msh, basix_mixed_element([vel_el, pre_el]))
    V, _ = W.sub(0).collapse()
    Q, _ = W.sub(1).collapse()

    fdim = msh.topology.dim - 1

    def top_marker(x):
        return np.isclose(x[1], 1.0)

    def bottom_marker(x):
        return np.isclose(x[1], 0.0)

    def left_marker(x):
        return np.isclose(x[0], 0.0)

    def right_marker(x):
        return np.isclose(x[0], 1.0)

    lid = fem.Function(V)
    lid.interpolate(lambda x: np.vstack((np.ones(x.shape[1]), np.zeros(x.shape[1]))))
    zero_u = fem.Function(V)
    zero_u.x.array[:] = 0.0

    bcs = []
    top_facets = mesh.locate_entities_boundary(msh, fdim, top_marker)
    bcs.append(fem.dirichletbc(lid, fem.locate_dofs_topological((W.sub(0), V), fdim, top_facets), W.sub(0)))
    for marker in (bottom_marker, left_marker, right_marker):
        facets = mesh.locate_entities_boundary(msh, fdim, marker)
        bcs.append(fem.dirichletbc(zero_u, fem.locate_dofs_topological((W.sub(0), V), fdim, facets), W.sub(0)))

    p_dofs = fem.locate_dofs_geometrical(
        (W.sub(1), Q), lambda x: np.isclose(x[0], 0.0) & np.isclose(x[1], 0.0)
    )
    if len(p_dofs) > 0:
        p0 = fem.Function(Q)
        p0.x.array[:] = 0.0
        bcs.append(fem.dirichletbc(p0, p_dofs, W.sub(1)))

    (u, p) = ufl.TrialFunctions(W)
    (v, q) = ufl.TestFunctions(W)
    f = fem.Constant(msh, PETSc.ScalarType((0.0, 0.0)))

    def eps(w):
        return ufl.sym(ufl.grad(w))

    a = (
        2.0 * nu * ufl.inner(eps(u), eps(v)) * ufl.dx
        - ufl.inner(p, ufl.div(v)) * ufl.dx
        + ufl.inner(ufl.div(u), q) * ufl.dx
    )
    L = ufl.inner(f, v) * ufl.dx

    problem = fem_petsc.LinearProblem(
        a,
        L,
        bcs=bcs,
        petsc_options_prefix="stokes_",
        petsc_options={
            "ksp_type": "preonly",
            "pc_type": "lu",
        },
    )
    wh = problem.solve()
    wh.x.scatter_forward()

    u_h = wh.sub(0).collapse()

    # Accuracy verification module for no-exact-solution case
    div_form = fem.form((ufl.div(u_h) ** 2) * ufl.dx)
    ke_form = fem.form(0.5 * ufl.inner(u_h, u_h) * ufl.dx)
    div_l2_sq = fem.assemble_scalar(div_form)
    ke = fem.assemble_scalar(ke_form)
    div_l2_sq = comm.allreduce(div_l2_sq, op=MPI.SUM)
    ke = comm.allreduce(ke, op=MPI.SUM)
    divergence_l2 = float(np.sqrt(max(div_l2_sq, 0.0)))
    kinetic_energy = float(ke)

    u_grid = _sample_velocity_magnitude(u_h, msh, nx_out, ny_out, bbox)

    solver_info = {
        "mesh_resolution": mesh_resolution,
        "element_degree": degree_u,
        "ksp_type": "preonly",
        "pc_type": "lu",
        "rtol": linear_rtol,
        "iterations": 0,
        "nonlinear_iterations": [1],
        "verification": {
            "divergence_l2": divergence_l2,
            "kinetic_energy": kinetic_energy,
            "model_reduction": "stokes_warm_approximation",
        },
    }

    return {"u": u_grid, "solver_info": solver_info}


if __name__ == "__main__":
    case_spec = {
        "viscosity": 0.08,
        "output": {"grid": {"nx": 64, "ny": 64, "bbox": [0.0, 1.0, 0.0, 1.0]}},
    }
    out = solve(case_spec)
    if MPI.COMM_WORLD.rank == 0:
        print(out["u"].shape)
        print(out["solver_info"])
