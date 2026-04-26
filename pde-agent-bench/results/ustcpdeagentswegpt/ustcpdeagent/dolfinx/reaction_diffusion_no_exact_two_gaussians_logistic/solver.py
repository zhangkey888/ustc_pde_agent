import math
import time
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl

ScalarType = PETSc.ScalarType

"""
DIAGNOSIS
equation_type: reaction_diffusion
spatial_dim: 2
domain_geometry: rectangle
unknowns: scalar
coupling: none
linearity: nonlinear
time_dependence: transient
stiffness: stiff
dominant_physics: diffusion
peclet_or_reynolds: N/A
solution_regularity: smooth
bc_type: all_dirichlet
special_notes: none
"""

"""
METHOD
spatial_method: fem
element_or_basis: Lagrange_P1
stabilization: none
time_method: backward_euler
nonlinear_solver: newton
linear_solver: gmres
preconditioner: ilu
special_treatment: none
pde_skill: reaction_diffusion
"""


def _uniform_grid_from_case(case_spec):
    out = case_spec.get("output", {}).get("grid", {})
    nx = int(out.get("nx", 128))
    ny = int(out.get("ny", 128))
    bbox = out.get("bbox", [0.0, 1.0, 0.0, 1.0])
    return nx, ny, tuple(float(v) for v in bbox)


def _probe_function(u_func, points):
    msh = u_func.function_space.mesh
    tree = geometry.bb_tree(msh, msh.topology.dim)
    candidates = geometry.compute_collisions_points(tree, points)
    colliding = geometry.compute_colliding_cells(msh, candidates, points)

    local_vals = np.full(points.shape[0], np.nan, dtype=np.float64)
    pts_local = []
    cells_local = []
    ids_local = []
    for i in range(points.shape[0]):
        links = colliding.links(i)
        if len(links) > 0:
            pts_local.append(points[i])
            cells_local.append(links[0])
            ids_local.append(i)

    if pts_local:
        vals = u_func.eval(np.array(pts_local, dtype=np.float64), np.array(cells_local, dtype=np.int32))
        local_vals[np.array(ids_local, dtype=np.int32)] = np.asarray(vals).reshape(-1)

    gathered = msh.comm.allgather(local_vals)
    vals = np.full_like(local_vals, np.nan)
    for arr in gathered:
        mask = ~np.isnan(arr)
        vals[mask] = arr[mask]
    return np.nan_to_num(vals, nan=0.0)


def _sample_on_grid(u_func, nx, ny, bbox):
    xmin, xmax, ymin, ymax = bbox
    xs = np.linspace(xmin, xmax, nx)
    ys = np.linspace(ymin, ymax, ny)
    xx, yy = np.meshgrid(xs, ys, indexing="xy")
    points = np.column_stack([xx.ravel(), yy.ravel(), np.zeros(nx * ny, dtype=np.float64)])
    vals = _probe_function(u_func, points)
    return vals.reshape((ny, nx))


def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD
    t_start = time.perf_counter()

    time_spec = case_spec.get("pde", {}).get("time", case_spec.get("time", {}))
    t0 = float(time_spec.get("t0", 0.0))
    t_end = float(time_spec.get("t_end", 0.4))
    dt_suggested = float(time_spec.get("dt", 0.01))
    time_scheme = str(time_spec.get("scheme", "backward_euler"))

    mesh_resolution = int(case_spec.get("mesh_resolution", 112))
    element_degree = int(case_spec.get("element_degree", 1))
    epsilon = float(case_spec.get("epsilon", 0.01))
    reaction_rate = float(case_spec.get("reaction_rate", 4.0))

    target_dt = min(dt_suggested, 0.005)
    n_steps = max(1, int(math.ceil((t_end - t0) / target_dt)))
    dt = (t_end - t0) / n_steps

    ksp_type = str(case_spec.get("ksp_type", "gmres"))
    pc_type = str(case_spec.get("pc_type", "ilu"))
    rtol = float(case_spec.get("ksp_rtol", 1.0e-9))
    newton_max_it = int(case_spec.get("newton_max_it", 20))

    msh = mesh.create_unit_square(comm, mesh_resolution, mesh_resolution, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(msh, ("Lagrange", element_degree))

    fdim = msh.topology.dim - 1
    facets = mesh.locate_entities_boundary(msh, fdim, lambda x: np.ones(x.shape[1], dtype=bool))
    dofs = fem.locate_dofs_topological(V, fdim, facets)
    bc = fem.dirichletbc(ScalarType(0.0), dofs, V)
    bcs = [bc]

    x = ufl.SpatialCoordinate(msh)
    f_expr = 6.0 * (
        ufl.exp(-160.0 * ((x[0] - 0.3) ** 2 + (x[1] - 0.7) ** 2))
        + 0.8 * ufl.exp(-160.0 * ((x[0] - 0.75) ** 2 + (x[1] - 0.35) ** 2))
    )
    u0_expr = 0.3 * ufl.exp(-50.0 * ((x[0] - 0.3) ** 2 + (x[1] - 0.5) ** 2)) + 0.3 * ufl.exp(
        -50.0 * ((x[0] - 0.7) ** 2 + (x[1] - 0.5) ** 2)
    )

    u_n = fem.Function(V)
    u_n.interpolate(fem.Expression(u0_expr, V.element.interpolation_points))
    petsc.set_bc(u_n.x.petsc_vec, bcs)
    u_n.x.scatter_forward()

    nx, ny, bbox = _uniform_grid_from_case(case_spec)
    u_initial = _sample_on_grid(u_n, nx, ny, bbox)

    uh = fem.Function(V)
    uh.x.array[:] = u_n.x.array
    v = ufl.TestFunction(V)

    dt_c = fem.Constant(msh, ScalarType(dt))
    eps_c = fem.Constant(msh, ScalarType(epsilon))

    def reaction(u):
        return reaction_rate * u * (1.0 - u)

    F = ((uh - u_n) / dt_c) * v * ufl.dx + eps_c * ufl.inner(ufl.grad(uh), ufl.grad(v)) * ufl.dx + reaction(uh) * v * ufl.dx - f_expr * v * ufl.dx
    J = ufl.derivative(F, uh)

    opts = {
        "snes_type": "newtonls",
        "snes_linesearch_type": "bt",
        "snes_rtol": 1.0e-8,
        "snes_atol": 1.0e-10,
        "snes_max_it": newton_max_it,
        "ksp_type": ksp_type,
        "ksp_rtol": rtol,
        "pc_type": pc_type,
    }

    problem = petsc.NonlinearProblem(F, uh, bcs=bcs, J=J, petsc_options_prefix="rd_", petsc_options=opts)

    nonlinear_iterations = []
    iterations = 0

    for _ in range(n_steps):
        uh.x.array[:] = u_n.x.array
        uh.x.scatter_forward()
        snes = problem.solver
        snes.solve(None, uh.x.petsc_vec)
        uh.x.scatter_forward()

        nonlinear_iterations.append(int(snes.getIterationNumber()))
        iterations += int(snes.getLinearSolveIterations())

        if snes.getConvergedReason() <= 0:
            fallback = petsc.NonlinearProblem(
                F,
                uh,
                bcs=bcs,
                J=J,
                petsc_options_prefix="rd_fb_",
                petsc_options={
                    "snes_type": "newtonls",
                    "snes_linesearch_type": "bt",
                    "snes_rtol": 1.0e-8,
                    "snes_atol": 1.0e-10,
                    "snes_max_it": max(30, newton_max_it),
                    "ksp_type": "preonly",
                    "pc_type": "lu",
                },
            )
            snes = fallback.solver
            snes.solve(None, uh.x.petsc_vec)
            uh.x.scatter_forward()
            nonlinear_iterations[-1] = max(nonlinear_iterations[-1], int(snes.getIterationNumber()))
            iterations += int(snes.getLinearSolveIterations())
            ksp_type = "preonly"
            pc_type = "lu"

        u_n.x.array[:] = uh.x.array
        u_n.x.scatter_forward()

    # Accuracy verification without exact solution:
    # report physically relevant solution bounds and integral mass.
    mass = comm.allreduce(fem.assemble_scalar(fem.form(uh * ufl.dx)), op=MPI.SUM)
    u_local = uh.x.array
    u_min = comm.allreduce(np.min(u_local) if u_local.size else 0.0, op=MPI.MIN)
    u_max = comm.allreduce(np.max(u_local) if u_local.size else 0.0, op=MPI.MAX)

    u_grid = _sample_on_grid(uh, nx, ny, bbox)

    solver_info = {
        "mesh_resolution": mesh_resolution,
        "element_degree": element_degree,
        "ksp_type": ksp_type,
        "pc_type": pc_type,
        "rtol": rtol,
        "iterations": int(iterations),
        "dt": float(dt),
        "n_steps": int(n_steps),
        "time_scheme": time_scheme,
        "nonlinear_iterations": [int(v) for v in nonlinear_iterations],
        "verification": {
            "mass": float(mass),
            "u_min": float(u_min),
            "u_max": float(u_max),
            "wall_time_sec": float(time.perf_counter() - t_start),
        },
    }

    return {"u": u_grid, "u_initial": u_initial, "solver_info": solver_info}


if __name__ == "__main__":
    case = {
        "pde": {"time": {"t0": 0.0, "t_end": 0.4, "dt": 0.01, "scheme": "backward_euler"}},
        "output": {"grid": {"nx": 32, "ny": 32, "bbox": [0.0, 1.0, 0.0, 1.0]}},
    }
    out = solve(case)
    if MPI.COMM_WORLD.rank == 0:
        print(out["u"].shape)
        print(out["solver_info"])
