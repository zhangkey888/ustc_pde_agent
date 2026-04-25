import math
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc

from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl

# ```DIAGNOSIS
# equation_type: reaction_diffusion
# spatial_dim: 2
# domain_geometry: rectangle
# unknowns: scalar
# coupling: none
# linearity: nonlinear
# time_dependence: transient
# stiffness: stiff
# dominant_physics: mixed
# peclet_or_reynolds: low
# solution_regularity: smooth
# bc_type: all_dirichlet
# special_notes: oscillatory_rhs
# ```
#
# ```METHOD
# spatial_method: fem
# element_or_basis: Lagrange_P2
# stabilization: none
# time_method: backward_euler
# nonlinear_solver: newton
# linear_solver: gmres
# preconditioner: ilu
# special_treatment: none
# pde_skill: reaction_diffusion
# ```

ScalarType = PETSc.ScalarType


def _get_time_spec(case_spec: dict):
    pde = case_spec.get("pde", {})
    t = pde.get("time", {})
    t0 = float(t.get("t0", 0.0))
    t_end = float(t.get("t_end", 0.3))
    dt = float(t.get("dt", 0.005))
    scheme = str(t.get("scheme", "backward_euler"))
    return t0, t_end, dt, scheme


def _reaction(u):
    return u**3


def _initial_condition(x):
    return 0.2 * np.sin(3.0 * np.pi * x[0]) * np.sin(2.0 * np.pi * x[1])


def _rhs_expr(msh):
    x = ufl.SpatialCoordinate(msh)
    return ufl.sin(6.0 * ufl.pi * x[0]) * ufl.sin(5.0 * ufl.pi * x[1])


def _sample_on_grid(u_func, grid_spec):
    msh = u_func.function_space.mesh
    nx = int(grid_spec["nx"])
    ny = int(grid_spec["ny"])
    xmin, xmax, ymin, ymax = map(float, grid_spec["bbox"])

    xs = np.linspace(xmin, xmax, nx)
    ys = np.linspace(ymin, ymax, ny)
    XX, YY = np.meshgrid(xs, ys, indexing="xy")
    pts = np.column_stack(
        [XX.ravel(), YY.ravel(), np.zeros(nx * ny, dtype=np.float64)]
    )

    tree = geometry.bb_tree(msh, msh.topology.dim)
    candidates = geometry.compute_collisions_points(tree, pts)
    colliding = geometry.compute_colliding_cells(msh, candidates, pts)

    local_vals = np.full(nx * ny, np.nan, dtype=np.float64)
    points_on_proc = []
    cells_on_proc = []
    ids = []

    for i in range(pts.shape[0]):
        links = colliding.links(i)
        if len(links) > 0:
            points_on_proc.append(pts[i])
            cells_on_proc.append(links[0])
            ids.append(i)

    if len(points_on_proc) > 0:
        vals = u_func.eval(
            np.array(points_on_proc, dtype=np.float64),
            np.array(cells_on_proc, dtype=np.int32),
        )
        local_vals[np.array(ids, dtype=np.int32)] = np.asarray(vals).reshape(-1)

    owned = np.where(np.isnan(local_vals), 0.0, local_vals)
    counts = np.where(np.isnan(local_vals), 0, 1).astype(np.int32)
    global_vals = msh.comm.allreduce(owned, op=MPI.SUM)
    global_counts = msh.comm.allreduce(counts, op=MPI.SUM)
    global_vals[global_counts == 0] = 0.0
    return global_vals.reshape((ny, nx))


def _assemble_global_scalar(comm, expr):
    return float(comm.allreduce(fem.assemble_scalar(fem.form(expr)), op=MPI.SUM))


def _l2_norm(u):
    return math.sqrt(_assemble_global_scalar(u.function_space.mesh.comm, ufl.inner(u, u) * ufl.dx))


def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD
    t0, t_end, dt_suggested, scheme = _get_time_spec(case_spec)

    degree = 2
    mesh_resolution = 80
    epsilon = float(case_spec.get("pde", {}).get("epsilon", 0.02))

    if t_end > t0:
        dt = min(dt_suggested, 0.0025)
        n_steps = int(math.ceil((t_end - t0) / dt))
        dt = (t_end - t0) / n_steps
    else:
        dt = dt_suggested
        n_steps = 0

    msh = mesh.create_unit_square(
        comm, mesh_resolution, mesh_resolution, cell_type=mesh.CellType.triangle
    )
    V = fem.functionspace(msh, ("Lagrange", degree))

    fdim = msh.topology.dim - 1
    facets = mesh.locate_entities_boundary(
        msh, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
    )
    dofs = fem.locate_dofs_topological(V, fdim, facets)
    bc = fem.dirichletbc(ScalarType(0.0), dofs, V)

    u_n = fem.Function(V)
    u_n.interpolate(_initial_condition)
    u_n.x.scatter_forward()

    u = fem.Function(V)
    u.x.array[:] = u_n.x.array
    u.x.scatter_forward()

    grid_spec = case_spec["output"]["grid"]
    u_initial_grid = _sample_on_grid(u_n, grid_spec)

    v = ufl.TestFunction(V)
    dt_c = fem.Constant(msh, ScalarType(dt if dt > 0 else 1.0))
    eps_c = fem.Constant(msh, ScalarType(epsilon))
    f = _rhs_expr(msh)

    F = (
        ((u - u_n) / dt_c) * v * ufl.dx
        + eps_c * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
        + _reaction(u) * v * ufl.dx
        - f * v * ufl.dx
    )
    J = ufl.derivative(F, u)

    base_opts = {
        "snes_type": "newtonls",
        "snes_linesearch_type": "bt",
        "snes_rtol": 1e-8,
        "snes_atol": 1e-10,
        "snes_max_it": 30,
        "ksp_type": "gmres",
        "pc_type": "ilu",
        "ksp_rtol": 1e-9,
    }

    try:
        problem = petsc.NonlinearProblem(
            F, u, bcs=[bc], J=J, petsc_options_prefix="rd_", petsc_options=base_opts
        )
    except Exception:
        fallback_create_opts = dict(base_opts)
        fallback_create_opts["ksp_type"] = "preonly"
        fallback_create_opts["pc_type"] = "lu"
        problem = petsc.NonlinearProblem(
            F,
            u,
            bcs=[bc],
            J=J,
            petsc_options_prefix="rdlu_",
            petsc_options=fallback_create_opts,
        )

    snes = problem.solver
    nonlinear_iterations = []
    linear_iterations_total = 0

    for _ in range(n_steps):
        u.x.array[:] = u_n.x.array
        u.x.scatter_forward()

        try:
            problem.solve()
        except Exception:
            fallback_opts = {
                "snes_type": "newtonls",
                "snes_linesearch_type": "bt",
                "snes_rtol": 1e-9,
                "snes_atol": 1e-11,
                "snes_max_it": 40,
                "ksp_type": "preonly",
                "pc_type": "lu",
                "ksp_rtol": 1e-12,
            }
            problem = petsc.NonlinearProblem(
                F,
                u,
                bcs=[bc],
                J=J,
                petsc_options_prefix="rdfb_",
                petsc_options=fallback_opts,
            )
            snes = problem.solver
            problem.solve()

        try:
            nonlinear_iterations.append(int(snes.getIterationNumber()))
        except Exception:
            nonlinear_iterations.append(0)

        try:
            linear_iterations_total += int(snes.getLinearSolveIterations())
        except Exception:
            pass

        u.x.scatter_forward()
        u_n.x.array[:] = u.x.array
        u_n.x.scatter_forward()

    u_grid = _sample_on_grid(u, grid_spec)

    verification = {
        "solution_l2_norm": float(_l2_norm(u)),
        "mass": _assemble_global_scalar(comm, u * ufl.dx),
        "grad_sq_integral": _assemble_global_scalar(comm, ufl.inner(ufl.grad(u), ufl.grad(u)) * ufl.dx),
        "reaction_integral": _assemble_global_scalar(comm, _reaction(u) * ufl.dx),
    }

    solver_info = {
        "mesh_resolution": mesh_resolution,
        "element_degree": degree,
        "ksp_type": str(snes.getKSP().getType()) if snes is not None else "gmres",
        "pc_type": str(snes.getKSP().getPC().getType()) if snes is not None else "ilu",
        "rtol": 1e-9,
        "iterations": int(linear_iterations_total),
        "dt": float(dt),
        "n_steps": int(n_steps),
        "time_scheme": "backward_euler" if scheme.lower() == "backward_euler" else scheme,
        "nonlinear_iterations": nonlinear_iterations,
        "accuracy_verification": verification,
    }

    return {
        "u": u_grid,
        "u_initial": u_initial_grid,
        "solver_info": solver_info,
    }


if __name__ == "__main__":
    case_spec = {
        "pde": {
            "time": {
                "t0": 0.0,
                "t_end": 0.3,
                "dt": 0.005,
                "scheme": "backward_euler",
            }
        },
        "output": {
            "grid": {
                "nx": 64,
                "ny": 64,
                "bbox": [0.0, 1.0, 0.0, 1.0],
            }
        },
    }
    out = solve(case_spec)
    if MPI.COMM_WORLD.rank == 0:
        print(out["u"].shape)
        print(out["solver_info"])
