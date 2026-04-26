import time
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
import ufl
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
from basix.ufl import element as basix_element, mixed_element as basix_mixed_element

ScalarType = PETSc.ScalarType


def _sample_function_on_points(func, points):
    msh = func.function_space.mesh
    tree = geometry.bb_tree(msh, msh.topology.dim)
    cell_candidates = geometry.compute_collisions_points(tree, points)
    colliding_cells = geometry.compute_colliding_cells(msh, cell_candidates, points)

    local_points = []
    local_cells = []
    local_ids = []
    for i in range(points.shape[0]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            local_points.append(points[i])
            local_cells.append(links[0])
            local_ids.append(i)

    value_size = func.function_space.element.value_size
    local_vals = np.full((points.shape[0], value_size), np.nan, dtype=np.float64)
    if len(local_points) > 0:
        vals = func.eval(np.array(local_points, dtype=np.float64), np.array(local_cells, dtype=np.int32))
        vals = np.asarray(vals, dtype=np.float64).reshape(len(local_points), value_size)
        local_vals[np.array(local_ids, dtype=np.int32), :] = vals

    comm = msh.comm
    gathered = comm.gather(local_vals, root=0)
    if comm.rank == 0:
        out = np.full_like(gathered[0], np.nan)
        for arr in gathered:
            mask = ~np.isnan(arr).any(axis=1)
            out[mask, :] = arr[mask, :]
        return out
    return None


def _sample_velocity_magnitude(u_func, grid_spec):
    nx = int(grid_spec["nx"])
    ny = int(grid_spec["ny"])
    bbox = grid_spec["bbox"]
    xmin, xmax, ymin, ymax = map(float, bbox)
    xs = np.linspace(xmin, xmax, nx)
    ys = np.linspace(ymin, ymax, ny)
    XX, YY = np.meshgrid(xs, ys, indexing="xy")
    points = np.column_stack(
        [XX.ravel(), YY.ravel(), np.zeros(nx * ny, dtype=np.float64)]
    )
    vals = _sample_function_on_points(u_func, points)
    if u_func.function_space.mesh.comm.rank == 0:
        mag = np.linalg.norm(vals, axis=1).reshape(ny, nx)
        return mag
    return None


def solve(case_spec: dict) -> dict:
    """
    Return a dict with:
    - "u": velocity magnitude sampled on requested grid as shape (ny, nx)
    - "solver_info": metadata and solver statistics
    """

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
    # bc_type: all_dirichlet
    # special_notes: pressure_pinning, manufactured_solution
    # ```

    # ```METHOD
    # spatial_method: fem
    # element_or_basis: Taylor-Hood_P3P2
    # stabilization: none
    # time_method: none
    # nonlinear_solver: none
    # linear_solver: minres
    # preconditioner: lu
    # special_treatment: pressure_pinning
    # pde_skill: stokes
    # ```

    comm = MPI.COMM_WORLD
    rank = comm.rank
    nu_value = float(case_spec.get("pde", {}).get("nu", 1.0))

    # Adaptive accuracy/time trade-off: use fairly fine mesh within large time budget
    # and keep stable direct factorization through MINRES+LU fallback to direct PREONLY+LU.
    mesh_resolution = int(case_spec.get("solver", {}).get("mesh_resolution", 48))
    if mesh_resolution < 40:
        mesh_resolution = 40

    t0 = time.perf_counter()
    msh = mesh.create_unit_square(comm, mesh_resolution, mesh_resolution, cell_type=mesh.CellType.triangle)
    gdim = msh.geometry.dim
    cell_name = msh.topology.cell_name()

    vel_el = basix_element("Lagrange", cell_name, 3, shape=(gdim,))
    pre_el = basix_element("Lagrange", cell_name, 2)
    W = fem.functionspace(msh, basix_mixed_element([vel_el, pre_el]))
    V, _ = W.sub(0).collapse()
    Q, _ = W.sub(1).collapse()

    x = ufl.SpatialCoordinate(msh)
    pi = np.pi

    u_exact_ufl = ufl.as_vector([
        ufl.pi * ufl.cos(ufl.pi * x[1]) * ufl.sin(ufl.pi * x[0]),
        -ufl.pi * ufl.cos(ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1]),
    ])
    p_exact_ufl = ufl.cos(ufl.pi * x[0]) * ufl.cos(ufl.pi * x[1])

    f_ufl = -nu_value * ufl.div(ufl.grad(u_exact_ufl)) + ufl.grad(p_exact_ufl)

    (u, p) = ufl.TrialFunctions(W)
    (v, q) = ufl.TestFunctions(W)

    a = (
        nu_value * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
        - ufl.inner(p, ufl.div(v)) * ufl.dx
        + ufl.inner(ufl.div(u), q) * ufl.dx
    )
    L = ufl.inner(f_ufl, v) * ufl.dx

    u_bc = fem.Function(V)
    u_bc.interpolate(fem.Expression(u_exact_ufl, V.element.interpolation_points))

    fdim = msh.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(msh, fdim, lambda X: np.ones(X.shape[1], dtype=bool))
    dofs_u = fem.locate_dofs_topological((W.sub(0), V), fdim, boundary_facets)
    bc_u = fem.dirichletbc(u_bc, dofs_u, W.sub(0))

    p_dofs = fem.locate_dofs_geometrical(
        (W.sub(1), Q),
        lambda X: np.isclose(X[0], 0.0) & np.isclose(X[1], 0.0),
    )
    p0 = fem.Function(Q)
    p0.x.array[:] = 0.0
    bc_p = fem.dirichletbc(p0, p_dofs, W.sub(1))
    bcs = [bc_u, bc_p]

    ksp_type = "minres"
    pc_type = "lu"
    rtol = 1e-10

    w_h = None
    iterations = 0
    try:
        problem = petsc.LinearProblem(
            a,
            L,
            bcs=bcs,
            petsc_options_prefix="stokes_",
            petsc_options={
                "ksp_type": ksp_type,
                "ksp_rtol": rtol,
                "pc_type": pc_type,
                "pc_factor_mat_solver_type": "mumps",
            },
        )
        w_h = problem.solve()
        try:
            iterations = int(problem.solver.getIterationNumber())
        except Exception:
            iterations = 0
    except Exception:
        ksp_type = "preonly"
        pc_type = "lu"
        problem = petsc.LinearProblem(
            a,
            L,
            bcs=bcs,
            petsc_options_prefix="stokes_fallback_",
            petsc_options={
                "ksp_type": ksp_type,
                "pc_type": pc_type,
                "pc_factor_mat_solver_type": "mumps",
            },
        )
        w_h = problem.solve()
        iterations = 1

    uh = w_h.sub(0).collapse()
    ph = w_h.sub(1).collapse()

    # Accuracy verification
    u_ex = fem.Function(V)
    u_ex.interpolate(fem.Expression(u_exact_ufl, V.element.interpolation_points))
    p_ex = fem.Function(Q)
    p_ex.interpolate(fem.Expression(p_exact_ufl, Q.element.interpolation_points))

    err_u_fun = fem.Function(V)
    err_u_fun.x.array[:] = uh.x.array - u_ex.x.array
    err_p_fun = fem.Function(Q)
    err_p_fun.x.array[:] = ph.x.array - p_ex.x.array

    l2_u_local = fem.assemble_scalar(fem.form(ufl.inner(err_u_fun, err_u_fun) * ufl.dx))
    l2_uex_local = fem.assemble_scalar(fem.form(ufl.inner(u_ex, u_ex) * ufl.dx))
    l2_p_local = fem.assemble_scalar(fem.form((err_p_fun * err_p_fun) * ufl.dx))
    l2_pex_local = fem.assemble_scalar(fem.form((p_ex * p_ex) * ufl.dx))

    l2_u = np.sqrt(comm.allreduce(l2_u_local, op=MPI.SUM))
    l2_uex = np.sqrt(comm.allreduce(l2_uex_local, op=MPI.SUM))
    l2_p = np.sqrt(comm.allreduce(l2_p_local, op=MPI.SUM))
    l2_pex = np.sqrt(comm.allreduce(l2_pex_local, op=MPI.SUM))

    rel_u = l2_u / max(l2_uex, 1e-16)
    rel_p = l2_p / max(l2_pex, 1e-16)

    u_grid = _sample_velocity_magnitude(uh, case_spec["output"]["grid"])

    wall_time = time.perf_counter() - t0
    solver_info = {
        "mesh_resolution": mesh_resolution,
        "element_degree": 3,
        "ksp_type": ksp_type,
        "pc_type": pc_type,
        "rtol": rtol,
        "iterations": int(iterations),
        "velocity_l2_error": float(l2_u),
        "velocity_relative_l2_error": float(rel_u),
        "pressure_l2_error": float(l2_p),
        "pressure_relative_l2_error": float(rel_p),
        "wall_time_sec": float(wall_time),
    }

    if rank == 0:
        return {"u": u_grid, "solver_info": solver_info}
    return {"u": None, "solver_info": solver_info}


if __name__ == "__main__":
    case_spec = {
        "pde": {"nu": 1.0, "time": None},
        "output": {"grid": {"nx": 64, "ny": 64, "bbox": [0.0, 1.0, 0.0, 1.0]}},
    }
    result = solve(case_spec)
    if MPI.COMM_WORLD.rank == 0:
        print(result["u"].shape)
        print(result["solver_info"])
