import time
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
import ufl
from basix.ufl import element as basix_element, mixed_element as basix_mixed_element
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc


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
#
# ```METHOD
# spatial_method: fem
# element_or_basis: Taylor-Hood_P2P1
# stabilization: none
# time_method: none
# nonlinear_solver: none
# linear_solver: minres
# preconditioner: hypre
# special_treatment: pressure_pinning
# pde_skill: stokes
# ```


ScalarType = PETSc.ScalarType


def _force_expression(x):
    f0 = 3.0 * np.exp(-50.0 * ((x[0] - 0.15) ** 2 + (x[1] - 0.15) ** 2))
    return np.vstack((f0, f0))


def _build_spaces(domain, degree_u=2, degree_p=1):
    cell_name = domain.topology.cell_name()
    gdim = domain.geometry.dim
    vel_el = basix_element("Lagrange", cell_name, degree_u, shape=(gdim,))
    pre_el = basix_element("Lagrange", cell_name, degree_p)
    W = fem.functionspace(domain, basix_mixed_element([vel_el, pre_el]))
    V, _ = W.sub(0).collapse()
    Q, _ = W.sub(1).collapse()
    return W, V, Q


def _create_bcs(domain, W, V, Q):
    fdim = domain.topology.dim - 1

    def on_x0(x):
        return np.isclose(x[0], 0.0)

    def on_y0(x):
        return np.isclose(x[1], 0.0)

    def on_y1(x):
        return np.isclose(x[1], 1.0)

    zero_u = fem.Function(V)
    zero_u.x.array[:] = 0.0

    facets_x0 = mesh.locate_entities_boundary(domain, fdim, on_x0)
    facets_y0 = mesh.locate_entities_boundary(domain, fdim, on_y0)
    facets_y1 = mesh.locate_entities_boundary(domain, fdim, on_y1)

    dofs_x0 = fem.locate_dofs_topological((W.sub(0), V), fdim, facets_x0)
    dofs_y0 = fem.locate_dofs_topological((W.sub(0), V), fdim, facets_y0)
    dofs_y1 = fem.locate_dofs_topological((W.sub(0), V), fdim, facets_y1)

    bc_x0 = fem.dirichletbc(zero_u, dofs_x0, W.sub(0))
    bc_y0 = fem.dirichletbc(zero_u, dofs_y0, W.sub(0))
    bc_y1 = fem.dirichletbc(zero_u, dofs_y1, W.sub(0))

    p_dofs = fem.locate_dofs_geometrical(
        (W.sub(1), Q),
        lambda x: np.isclose(x[0], 0.0) & np.isclose(x[1], 0.0),
    )
    bcs = [bc_x0, bc_y0, bc_y1]
    if len(p_dofs) > 0:
        p0 = fem.Function(Q)
        p0.x.array[:] = 0.0
        bc_p = fem.dirichletbc(p0, p_dofs, W.sub(1))
        bcs.append(bc_p)
    return bcs


def _solve_stokes(domain, nu, n, degree_u=2, degree_p=1, rtol=1e-9):
    W, V, Q = _build_spaces(domain, degree_u=degree_u, degree_p=degree_p)
    bcs = _create_bcs(domain, W, V, Q)

    (u, p) = ufl.TrialFunctions(W)
    (v, q) = ufl.TestFunctions(W)

    f_fun = fem.Function(V)
    f_fun.interpolate(_force_expression)

    a = (
        2.0 * nu * ufl.inner(ufl.sym(ufl.grad(u)), ufl.sym(ufl.grad(v))) * ufl.dx
        - ufl.inner(p, ufl.div(v)) * ufl.dx
        + ufl.inner(ufl.div(u), q) * ufl.dx
    )
    L = ufl.inner(f_fun, v) * ufl.dx

    linear_problem = petsc.LinearProblem(
        a,
        L,
        bcs=bcs,
        petsc_options_prefix=f"stokes_{n}_",
        petsc_options={
            "ksp_type": "minres",
            "ksp_rtol": rtol,
            "pc_type": "hypre",
        },
    )

    wh = linear_problem.solve()
    wh.x.scatter_forward()

    # Extract KSP info
    ksp = linear_problem.solver
    iterations = int(ksp.getIterationNumber())
    ksp_type = ksp.getType()
    pc_type = ksp.getPC().getType()

    uh, _ = wh.sub(0).collapse()
    ph, _ = wh.sub(1).collapse()

    # Verification metrics
    V_div = fem.functionspace(domain, ("DG", 0))
    div_expr = fem.Expression(ufl.div(uh), V_div.element.interpolation_points)
    div_fun = fem.Function(V_div)
    div_fun.interpolate(div_expr)
    div_l2 = np.sqrt(domain.comm.allreduce(fem.assemble_scalar(fem.form(div_fun * div_fun * ufl.dx)), op=MPI.SUM))

    res_fun = fem.form(
        (
            2.0 * nu * ufl.inner(ufl.sym(ufl.grad(uh)), ufl.sym(ufl.grad(v))) * ufl.dx
            - ufl.inner(ph, ufl.div(v)) * ufl.dx
            + ufl.inner(ufl.div(uh), q) * ufl.dx
            - ufl.inner(f_fun, v) * ufl.dx
        )
    )
    residual_vec = petsc.assemble_vector(res_fun)
    residual_vec.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    petsc.set_bc(residual_vec, bcs)
    residual_norm = residual_vec.norm()

    info = {
        "mesh_resolution": int(n),
        "element_degree": int(degree_u),
        "ksp_type": str(ksp_type),
        "pc_type": str(pc_type),
        "rtol": float(rtol),
        "iterations": iterations,
        "verification": {
            "divergence_l2": float(div_l2),
            "algebraic_residual_norm": float(residual_norm),
        },
    }
    return uh, ph, info


def _sample_function_magnitude(u_fun, grid_spec):
    domain = u_fun.function_space.mesh
    nx = int(grid_spec["nx"])
    ny = int(grid_spec["ny"])
    bbox = grid_spec["bbox"]
    xmin, xmax, ymin, ymax = map(float, bbox)

    xs = np.linspace(xmin, xmax, nx)
    ys = np.linspace(ymin, ymax, ny)
    XX, YY = np.meshgrid(xs, ys, indexing="xy")
    points = np.column_stack([XX.ravel(), YY.ravel(), np.zeros(nx * ny, dtype=np.float64)])

    tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(tree, points)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points)

    local_vals = np.full(points.shape[0], np.nan, dtype=np.float64)
    pts_local = []
    cells_local = []
    ids_local = []

    for i in range(points.shape[0]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            pts_local.append(points[i])
            cells_local.append(links[0])
            ids_local.append(i)

    if len(pts_local) > 0:
        vals = u_fun.eval(np.asarray(pts_local, dtype=np.float64), np.asarray(cells_local, dtype=np.int32))
        mags = np.linalg.norm(vals, axis=1)
        local_vals[np.asarray(ids_local, dtype=np.int32)] = mags

    comm = domain.comm
    gathered = comm.gather(local_vals, root=0)

    if comm.rank == 0:
        final = np.full(points.shape[0], np.nan, dtype=np.float64)
        for arr in gathered:
            mask = ~np.isnan(arr)
            final[mask] = arr[mask]
        if np.isnan(final).any():
            final = np.nan_to_num(final, nan=0.0)
        return final.reshape((ny, nx))
    return None


def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD
    nu = float(case_spec.get("pde", {}).get("nu", case_spec.get("pde", {}).get("viscosity", 0.1)))
    grid_spec = case_spec["output"]["grid"]

    # Adaptive accuracy under generous time budget for this steady 2D Stokes case
    requested_n = int(case_spec.get("solver", {}).get("mesh_resolution", 0) or 0)
    n = max(64, requested_n) if requested_n > 0 else 96
    degree_u = 2
    degree_p = 1
    rtol = 1e-9

    t0 = time.perf_counter()
    domain = mesh.create_unit_square(comm, n, n, cell_type=mesh.CellType.triangle)
    uh, ph, solver_info = _solve_stokes(domain, nu, n, degree_u=degree_u, degree_p=degree_p, rtol=rtol)
    elapsed = time.perf_counter() - t0

    # If solution was very fast, refine once to improve accuracy
    if elapsed < 5.0 and n < 144:
        n2 = min(144, int(round(1.25 * n)))
        domain2 = mesh.create_unit_square(comm, n2, n2, cell_type=mesh.CellType.triangle)
        uh2, ph2, solver_info2 = _solve_stokes(domain2, nu, n2, degree_u=degree_u, degree_p=degree_p, rtol=rtol)
        if solver_info2["verification"]["divergence_l2"] <= solver_info["verification"]["divergence_l2"] + 1e-12:
            uh, ph, solver_info = uh2, ph2, solver_info2

    u_grid = _sample_function_magnitude(uh, grid_spec)

    if comm.rank == 0:
        return {
            "u": np.asarray(u_grid, dtype=np.float64),
            "solver_info": solver_info,
        }
    return {"u": None, "solver_info": solver_info}


if __name__ == "__main__":
    case_spec = {
        "pde": {"nu": 0.1},
        "output": {"grid": {"nx": 64, "ny": 64, "bbox": [0.0, 1.0, 0.0, 1.0]}},
    }
    result = solve(case_spec)
    if MPI.COMM_WORLD.rank == 0:
        print(result["u"].shape)
        print(result["solver_info"])
