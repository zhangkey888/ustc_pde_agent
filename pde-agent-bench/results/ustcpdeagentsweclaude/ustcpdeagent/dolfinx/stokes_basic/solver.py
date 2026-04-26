import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('NUMEXPR_NUM_THREADS', '1')
import math
import time
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
import ufl
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
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
# bc_type: all_dirichlet
# special_notes: manufactured_solution
# ```
#
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

COMM = MPI.COMM_WORLD
ScalarType = PETSc.ScalarType


def _u_exact_ufl(x):
    return ufl.as_vector(
        [
            ufl.pi * ufl.cos(ufl.pi * x[1]) * ufl.sin(ufl.pi * x[0]),
            -ufl.pi * ufl.cos(ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1]),
        ]
    )


def _p_exact_ufl(x):
    return ufl.cos(ufl.pi * x[0]) * ufl.cos(ufl.pi * x[1])


def _forcing_ufl(msh, nu):
    x = ufl.SpatialCoordinate(msh)
    u_ex = _u_exact_ufl(x)
    p_ex = _p_exact_ufl(x)
    lap_u = ufl.div(ufl.grad(u_ex))
    grad_p = ufl.grad(p_ex)
    return ufl.as_vector([
        -nu * lap_u[i] + grad_p[i] for i in range(msh.geometry.dim)
    ])


def _sample_function(func, points):
    msh = func.function_space.mesh
    tree = geometry.bb_tree(msh, msh.topology.dim)
    candidates = geometry.compute_collisions_points(tree, points)
    colliding = geometry.compute_colliding_cells(msh, candidates, points)

    point_ids = []
    points_local = []
    cells_local = []
    for i in range(points.shape[0]):
        links = colliding.links(i)
        if len(links) > 0:
            point_ids.append(i)
            points_local.append(points[i])
            cells_local.append(links[0])

    values = None
    if len(points_local) > 0:
        values = func.eval(
            np.asarray(points_local, dtype=np.float64),
            np.asarray(cells_local, dtype=np.int32),
        )
    return np.asarray(point_ids, dtype=np.int64), values


def _sample_velocity_magnitude(uh, grid_spec):
    nx = int(grid_spec["nx"])
    ny = int(grid_spec["ny"])
    xmin, xmax, ymin, ymax = map(float, grid_spec["bbox"])

    xs = np.linspace(xmin, xmax, nx, dtype=np.float64)
    ys = np.linspace(ymin, ymax, ny, dtype=np.float64)
    XX, YY = np.meshgrid(xs, ys, indexing="xy")
    points = np.column_stack(
        [XX.reshape(-1), YY.reshape(-1), np.zeros(nx * ny, dtype=np.float64)]
    )

    ids_local, vals_local = _sample_function(uh, points)
    gathered_ids = COMM.gather(ids_local, root=0)
    gathered_vals = COMM.gather(vals_local, root=0)

    if COMM.rank != 0:
        return None

    full = np.full((nx * ny, uh.function_space.mesh.geometry.dim), np.nan, dtype=np.float64)
    for ids, vals in zip(gathered_ids, gathered_vals):
        if ids.size > 0 and vals is not None:
            full[ids] = np.asarray(vals, dtype=np.float64)

    if np.isnan(full).any():
        raise RuntimeError("Failed to evaluate solution at one or more output grid points.")

    return np.linalg.norm(full, axis=1).reshape(ny, nx)


def _build_and_solve(mesh_resolution):
    msh = mesh.create_unit_square(COMM, mesh_resolution, mesh_resolution, cell_type=mesh.CellType.triangle)
    gdim = msh.geometry.dim
    cell_name = msh.topology.cell_name()

    vel_el = basix_element("Lagrange", cell_name, 2, shape=(gdim,))
    pre_el = basix_element("Lagrange", cell_name, 1)
    W = fem.functionspace(msh, basix_mixed_element([vel_el, pre_el]))
    V, _ = W.sub(0).collapse()
    Q, _ = W.sub(1).collapse()

    x = ufl.SpatialCoordinate(msh)
    nu = ScalarType(1.0)
    u_ex = _u_exact_ufl(x)
    p_ex = _p_exact_ufl(x)
    f = _forcing_ufl(msh, nu)

    u, p = ufl.TrialFunctions(W)
    v, q = ufl.TestFunctions(W)

    a = (
        nu * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
        - ufl.inner(p, ufl.div(v)) * ufl.dx
        + ufl.inner(ufl.div(u), q) * ufl.dx
    )
    L = ufl.inner(f, v) * ufl.dx

    fdim = msh.topology.dim - 1
    facets = mesh.locate_entities_boundary(msh, fdim, lambda X: np.ones(X.shape[1], dtype=bool))

    u_bc = fem.Function(V)
    u_bc.interpolate(fem.Expression(u_ex, V.element.interpolation_points))
    u_dofs = fem.locate_dofs_topological((W.sub(0), V), fdim, facets)
    bc_u = fem.dirichletbc(u_bc, u_dofs, W.sub(0))

    p_corner = fem.Function(Q)
    p_corner.interpolate(fem.Expression(p_ex, Q.element.interpolation_points))
    p_dofs = fem.locate_dofs_geometrical(
        (W.sub(1), Q),
        lambda X: np.isclose(X[0], 0.0) & np.isclose(X[1], 0.0),
    )
    bcs = [bc_u]
    if len(p_dofs) > 0:
        bcs.append(fem.dirichletbc(p_corner, p_dofs, W.sub(1)))

    petsc_options = {
        "ksp_type": "preonly",
        "pc_type": "lu",
        "pc_factor_mat_solver_type": "mumps",
    }

    try:
        problem = petsc.LinearProblem(
            a,
            L,
            bcs=bcs,
            petsc_options_prefix="stokes_basic_",
            petsc_options=petsc_options,
        )
        wh = problem.solve()
    except Exception:
        problem = petsc.LinearProblem(
            a,
            L,
            bcs=bcs,
            petsc_options_prefix="stokes_basic_fb_",
            petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
        )
        wh = problem.solve()

    wh.x.scatter_forward()
    uh = wh.sub(0).collapse()
    ph = wh.sub(1).collapse()
    uh.x.scatter_forward()
    ph.x.scatter_forward()

    u_exact = fem.Function(V)
    u_exact.interpolate(fem.Expression(u_ex, V.element.interpolation_points))
    u_exact.x.scatter_forward()

    eL2_form = fem.form(ufl.inner(uh - u_exact, uh - u_exact) * ufl.dx)
    eL2_local = fem.assemble_scalar(eL2_form)
    eL2 = math.sqrt(max(COMM.allreduce(eL2_local, op=MPI.SUM), 0.0))

    diff = fem.Function(V)
    diff.x.array[:] = uh.x.array - u_exact.x.array
    diff.x.scatter_forward()
    emax_local = np.max(np.abs(diff.x.array)) if diff.x.array.size > 0 else 0.0
    emax = COMM.allreduce(emax_local, op=MPI.MAX)

    return uh, ph, {
        "mesh_resolution": int(mesh_resolution),
        "element_degree": 2,
        "ksp_type": "preonly",
        "pc_type": "lu",
        "rtol": 1.0e-13,
        "iterations": 1,
        "verification": {
            "velocity_L2_error": float(eL2),
            "velocity_dof_max_error": float(emax),
        },
    }


def solve(case_spec: dict) -> dict:
    t0 = time.perf_counter()

    mesh_resolution = 96
    uh, ph, solver_info = _build_and_solve(mesh_resolution)
    u_grid = _sample_velocity_magnitude(uh, case_spec["output"]["grid"])

    elapsed = time.perf_counter() - t0
    solver_info["wall_time_sec"] = float(elapsed)

    if COMM.rank == 0:
        grid_spec = case_spec["output"]["grid"]
        nx = int(grid_spec["nx"])
        ny = int(grid_spec["ny"])
        xmin, xmax, ymin, ymax = map(float, grid_spec["bbox"])
        xs = np.linspace(xmin, xmax, nx, dtype=np.float64)
        ys = np.linspace(ymin, ymax, ny, dtype=np.float64)
        XX, YY = np.meshgrid(xs, ys, indexing="xy")
        ux = np.pi * np.cos(np.pi * YY) * np.sin(np.pi * XX)
        uy = -np.pi * np.cos(np.pi * XX) * np.sin(np.pi * YY)
        mag_ex = np.sqrt(ux * ux + uy * uy)
        solver_info["verification"]["output_grid_max_error"] = float(np.max(np.abs(u_grid - mag_ex)))
        solver_info["verification"]["output_grid_l2_mean_error"] = float(np.sqrt(np.mean((u_grid - mag_ex) ** 2)))
        return {"u": u_grid, "solver_info": solver_info}

    return {"u": None, "solver_info": solver_info}


if __name__ == "__main__":
    case_spec = {
        "pde": {"time": None},
        "output": {"grid": {"nx": 64, "ny": 64, "bbox": [0.0, 1.0, 0.0, 1.0]}},
    }
    out = solve(case_spec)
    if COMM.rank == 0:
        print(out["u"].shape)
        print(out["solver_info"])
