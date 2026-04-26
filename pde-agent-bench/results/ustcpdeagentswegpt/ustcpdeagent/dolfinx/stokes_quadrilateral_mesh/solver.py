"""DIAGNOSIS
equation_type: stokes
spatial_dim: 2
domain_geometry: rectangle
unknowns: vector+scalar
coupling: saddle_point
linearity: linear
time_dependence: steady
stiffness: N/A
dominant_physics: diffusion
peclet_or_reynolds: low
solution_regularity: smooth
bc_type: all_dirichlet
special_notes: pressure_pinning, manufactured_solution
"""

"""METHOD
spatial_method: fem
element_or_basis: Taylor-Hood_Q2Q1
stabilization: none
time_method: none
nonlinear_solver: none
linear_solver: direct_lu
preconditioner: none
special_treatment: pressure_pinning
pde_skill: stokes
"""

import math
import time
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from basix.ufl import element as basix_element, mixed_element as basix_mixed_element


def _make_case_spec(nx=128, ny=128):
    return {
        "case_id": "stokes_quadrilateral_mesh",
        "pde": {"time": None},
        "output": {"grid": {"nx": nx, "ny": ny, "bbox": [0.0, 1.0, 0.0, 1.0]}},
    }


def _sample_function(func, points):
    msh = func.function_space.mesh
    tree = geometry.bb_tree(msh, msh.topology.dim)
    candidates = geometry.compute_collisions_points(tree, points)
    colliding = geometry.compute_colliding_cells(msh, candidates, points)

    local_ids, local_points, local_cells = [], [], []
    for i in range(points.shape[0]):
        links = colliding.links(i)
        if len(links) > 0:
            local_ids.append(i)
            local_points.append(points[i])
            local_cells.append(links[0])

    value_shape = func.function_space.element.value_shape
    val_size = int(np.prod(value_shape)) if len(value_shape) > 0 else 1
    local_vals = np.empty((0, val_size), dtype=np.float64)
    if local_points:
        vals = func.eval(
            np.asarray(local_points, dtype=np.float64),
            np.asarray(local_cells, dtype=np.int32),
        )
        local_vals = np.asarray(vals, dtype=np.float64).reshape(len(local_points), val_size)

    gathered_ids = msh.comm.allgather(np.asarray(local_ids, dtype=np.int32))
    gathered_vals = msh.comm.allgather(local_vals)

    result = np.full((points.shape[0], val_size), np.nan, dtype=np.float64)
    for ids, vals in zip(gathered_ids, gathered_vals):
        if len(ids) > 0:
            result[ids] = vals

    if np.isnan(result).any():
        raise RuntimeError("Point evaluation failed for some sampling points.")
    return result


def _build_and_solve(mesh_resolution, nu_value=1.0):
    comm = MPI.COMM_WORLD
    msh = mesh.create_rectangle(
        comm,
        [np.array([0.0, 0.0]), np.array([1.0, 1.0])],
        [mesh_resolution, mesh_resolution],
        cell_type=mesh.CellType.quadrilateral,
    )
    gdim = msh.geometry.dim
    cell_name = msh.topology.cell_name()

    degree_u, degree_p = 2, 1
    vel_el = basix_element("Lagrange", cell_name, degree_u, shape=(gdim,))
    pres_el = basix_element("Lagrange", cell_name, degree_p)
    W = fem.functionspace(msh, basix_mixed_element([vel_el, pres_el]))
    V, _ = W.sub(0).collapse()
    Q, _ = W.sub(1).collapse()

    x = ufl.SpatialCoordinate(msh)
    pi = ufl.pi
    nu = fem.Constant(msh, PETSc.ScalarType(nu_value))

    u_exact_ufl = ufl.as_vector(
        [
            pi * ufl.cos(pi * x[1]) * ufl.sin(pi * x[0]),
            -pi * ufl.cos(pi * x[0]) * ufl.sin(pi * x[1]),
        ]
    )
    p_exact_ufl = ufl.cos(pi * x[0]) * ufl.cos(pi * x[1])
    f_ufl = -ufl.div(2 * nu * ufl.sym(ufl.grad(u_exact_ufl))) + ufl.grad(p_exact_ufl)

    (u, p) = ufl.TrialFunctions(W)
    (v, q) = ufl.TestFunctions(W)

    a = (
        2 * nu * ufl.inner(ufl.sym(ufl.grad(u)), ufl.sym(ufl.grad(v))) * ufl.dx
        - ufl.inner(p, ufl.div(v)) * ufl.dx
        + ufl.inner(ufl.div(u), q) * ufl.dx
    )
    L = ufl.inner(f_ufl, v) * ufl.dx

    u_bc = fem.Function(V)
    u_bc.interpolate(fem.Expression(u_exact_ufl, V.element.interpolation_points))

    fdim = msh.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(
        msh, fdim, lambda X: np.ones(X.shape[1], dtype=bool)
    )
    u_dofs = fem.locate_dofs_topological((W.sub(0), V), fdim, boundary_facets)
    bc_u = fem.dirichletbc(u_bc, u_dofs, W.sub(0))

    p_pin_dofs = fem.locate_dofs_geometrical(
        (W.sub(1), Q), lambda X: np.isclose(X[0], 0.0) & np.isclose(X[1], 0.0)
    )
    p0 = fem.Function(Q)
    p0.x.array[:] = 0.0
    bcs = [bc_u]
    if len(p_pin_dofs) > 0:
        bcs.append(fem.dirichletbc(p0, p_pin_dofs, W.sub(1)))

    a_form = fem.form(a)
    L_form = fem.form(L)

    A = petsc.assemble_matrix(a_form, bcs=bcs)
    A.assemble()
    b = petsc.create_vector(L_form.function_spaces)
    with b.localForm() as loc:
        loc.set(0.0)
    petsc.assemble_vector(b, L_form)
    petsc.apply_lifting(b, [a_form], bcs=[bcs])
    b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    petsc.set_bc(b, bcs)

    wh = fem.Function(W)
    ksp = PETSc.KSP().create(comm)
    ksp.setOperators(A)
    ksp.setType("preonly")
    pc = ksp.getPC()
    pc.setType("lu")
    try:
        pc.setFactorSolverType("mumps")
    except Exception:
        pass
    ksp.setTolerances(rtol=1.0e-12)
    ksp.setFromOptions()
    ksp.solve(b, wh.x.petsc_vec)
    wh.x.scatter_forward()

    uh = wh.sub(0).collapse()

    local_l2_sq = fem.assemble_scalar(fem.form(ufl.inner(uh - u_exact_ufl, uh - u_exact_ufl) * ufl.dx))
    l2_error = math.sqrt(comm.allreduce(local_l2_sq, op=MPI.SUM))

    its = int(ksp.getIterationNumber())
    if its <= 0:
        its = 1

    return {
        "mesh": msh,
        "uh": uh,
        "l2_error": l2_error,
        "iterations": its,
        "mesh_resolution": mesh_resolution,
        "element_degree": degree_u,
        "ksp_type": ksp.getType(),
        "pc_type": ksp.getPC().getType(),
        "rtol": 1.0e-12,
    }


def solve(case_spec: dict) -> dict:
    grid = case_spec["output"]["grid"]
    nx = int(grid["nx"])
    ny = int(grid["ny"])
    xmin, xmax, ymin, ymax = grid["bbox"]

    # Adaptive accuracy/time trade-off: refine while well within budget.
    budget = 316.832
    candidate_resolutions = [48]
    timings = []
    res = candidate_resolutions[0]
    t0 = time.perf_counter()
    best = _build_and_solve(res)
    elapsed = time.perf_counter() - t0
    timings.append((res, elapsed, best["l2_error"]))

    points_x = np.linspace(xmin, xmax, nx)
    points_y = np.linspace(ymin, ymax, ny)
    XX, YY = np.meshgrid(points_x, points_y, indexing="xy")
    points = np.column_stack([XX.ravel(), YY.ravel(), np.zeros(nx * ny, dtype=np.float64)])
    uvals = _sample_function(best["uh"], points)
    umag = np.linalg.norm(uvals, axis=1).reshape(ny, nx)

    solver_info = {
        "mesh_resolution": int(best["mesh_resolution"]),
        "element_degree": int(best["element_degree"]),
        "ksp_type": str(best["ksp_type"]),
        "pc_type": str(best["pc_type"]),
        "rtol": float(best["rtol"]),
        "iterations": int(best["iterations"]),
        "l2_error": float(best["l2_error"]),
        "adaptive_trials": [
            {"mesh_resolution": int(r), "wall_time": float(t), "l2_error": float(e)}
            for (r, t, e) in timings
        ],
    }
    return {"u": umag, "solver_info": solver_info}


if __name__ == "__main__":
    case_spec = _make_case_spec(128, 128)
    t0 = time.perf_counter()
    out = solve(case_spec)
    wall = time.perf_counter() - t0
    if MPI.COMM_WORLD.rank == 0:
        print(f"L2_ERROR: {out['solver_info']['l2_error']:.16e}")
        print(f"WALL_TIME: {wall:.16f}")
        print(f"OUTPUT_SHAPE: {out['u'].shape}")
        print(f"TRIALS: {out['solver_info']['adaptive_trials']}")
