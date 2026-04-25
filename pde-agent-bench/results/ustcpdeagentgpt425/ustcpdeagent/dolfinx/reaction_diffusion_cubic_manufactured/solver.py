import time
import math
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl

ScalarType = PETSc.ScalarType


def exact_solution_expr(x, t):
    return 0.2 * ufl.exp(-t) * ufl.sin(2.0 * ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])


def exact_solution_numpy(points, t):
    return 0.2 * np.exp(-t) * np.sin(2.0 * np.pi * points[0]) * np.sin(np.pi * points[1])


def manufactured_source_expr(domain, epsilon, t):
    x = ufl.SpatialCoordinate(domain)
    uex = exact_solution_expr(x, t)
    return -uex + epsilon * (5.0 * ufl.pi * ufl.pi) * uex + uex**3


def boundary_dofs(V, domain):
    fdim = domain.topology.dim - 1
    facets = mesh.locate_entities_boundary(domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool))
    return fem.locate_dofs_topological(V, fdim, facets)


def set_exact(func, t):
    func.interpolate(lambda x: exact_solution_numpy(x, t))


def sample_on_grid(u_func, nx, ny, bbox):
    comm = u_func.function_space.mesh.comm
    domain = u_func.function_space.mesh
    xs = np.linspace(bbox[0], bbox[1], nx)
    ys = np.linspace(bbox[2], bbox[3], ny)
    XX, YY = np.meshgrid(xs, ys)
    pts = np.column_stack([XX.ravel(), YY.ravel(), np.zeros(nx * ny)])

    tree = geometry.bb_tree(domain, domain.topology.dim)
    candidates = geometry.compute_collisions_points(tree, pts)
    colliding = geometry.compute_colliding_cells(domain, candidates, pts)

    local_ids = []
    local_pts = []
    local_cells = []
    for i in range(pts.shape[0]):
        links = colliding.links(i)
        if len(links) > 0:
            local_ids.append(i)
            local_pts.append(pts[i])
            local_cells.append(links[0])

    local_vals = np.full(pts.shape[0], np.nan, dtype=np.float64)
    if local_pts:
        vals = u_func.eval(np.array(local_pts, dtype=np.float64), np.array(local_cells, dtype=np.int32))
        local_vals[np.array(local_ids, dtype=np.int32)] = np.asarray(vals).reshape(-1)

    gathered = comm.gather(local_vals, root=0)
    if comm.rank == 0:
        merged = np.full(pts.shape[0], np.nan, dtype=np.float64)
        for arr in gathered:
            mask = ~np.isnan(arr)
            merged[mask] = arr[mask]
        if np.isnan(merged).any():
            raise RuntimeError("Sampling failed for some points")
        return merged.reshape(ny, nx)
    return None


def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD

    time_spec = case_spec.get("pde", {}).get("time", {})
    t0 = float(time_spec.get("t0", 0.0))
    t_end = float(time_spec.get("t_end", 0.2))
    dt_default = float(time_spec.get("dt", 0.005))
    time_scheme = str(time_spec.get("scheme", "backward_euler"))

    out_grid = case_spec["output"]["grid"]
    nx_out = int(out_grid["nx"])
    ny_out = int(out_grid["ny"])
    bbox = out_grid["bbox"]

    params = case_spec.get("agent_params", {})
    mesh_resolution = int(params.get("mesh_resolution", 24))
    element_degree = int(params.get("element_degree", 2))
    epsilon = float(params.get("epsilon", 0.05))
    dt = float(params.get("dt", dt_default))
    newton_rtol = float(params.get("newton_rtol", 1.0e-10))
    newton_max_it = int(params.get("newton_max_it", 20))
    ksp_type = str(params.get("ksp_type", "gmres"))
    pc_type = str(params.get("pc_type", "ilu"))
    rtol = float(params.get("rtol", 1.0e-9))

    domain = mesh.create_unit_square(comm, mesh_resolution, mesh_resolution, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", element_degree))

    u_n = fem.Function(V)
    set_exact(u_n, t0)

    u = fem.Function(V)
    u.x.array[:] = u_n.x.array
    u.x.scatter_forward()

    u_bc = fem.Function(V)
    set_exact(u_bc, t0)
    bc = fem.dirichletbc(u_bc, boundary_dofs(V, domain))

    v = ufl.TestFunction(V)
    eps_c = fem.Constant(domain, ScalarType(epsilon))
    dt_c = fem.Constant(domain, ScalarType(dt))

    n_steps = int(round((t_end - t0) / dt))
    nonlinear_iterations = []
    total_linear_iterations = 0

    for step in range(n_steps):
        t = t0 + (step + 1) * dt
        set_exact(u_bc, t)
        f_expr = manufactured_source_expr(domain, eps_c, ScalarType(t))

        F = ((u - u_n) / dt_c) * v * ufl.dx + eps_c * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx + u**3 * v * ufl.dx - f_expr * v * ufl.dx
        J = ufl.derivative(F, u)

        petsc_options = {
            "snes_type": "newtonls",
            "snes_linesearch_type": "bt",
            "snes_rtol": newton_rtol,
            "snes_atol": 1.0e-12,
            "snes_max_it": newton_max_it,
            "ksp_type": ksp_type,
            "pc_type": pc_type,
            "ksp_rtol": rtol,
        }

        try:
            problem = petsc.NonlinearProblem(F, u, bcs=[bc], J=J, petsc_options_prefix=f"rd_{step}_", petsc_options=petsc_options)
            u = problem.solve()
        except Exception:
            petsc_options_fallback = dict(petsc_options)
            petsc_options_fallback["ksp_type"] = "preonly"
            petsc_options_fallback["pc_type"] = "lu"
            problem = petsc.NonlinearProblem(F, u, bcs=[bc], J=J, petsc_options_prefix=f"rdlu_{step}_", petsc_options=petsc_options_fallback)
            u = problem.solve()
            ksp_type = "preonly"
            pc_type = "lu"

        u.x.scatter_forward()
        snes = problem.solver
        nonlinear_iterations.append(int(snes.getIterationNumber()))
        try:
            total_linear_iterations += int(snes.getLinearSolveIterations())
        except Exception:
            pass

        u_n.x.array[:] = u.x.array
        u_n.x.scatter_forward()

    u_exact = fem.Function(V)
    set_exact(u_exact, t_end)
    l2_local = fem.assemble_scalar(fem.form((u - u_exact) ** 2 * ufl.dx))
    l2_error = math.sqrt(comm.allreduce(l2_local, op=MPI.SUM))

    u_grid = sample_on_grid(u, nx_out, ny_out, bbox)

    if comm.rank == 0:
        xs = np.linspace(bbox[0], bbox[1], nx_out)
        ys = np.linspace(bbox[2], bbox[3], ny_out)
        XX, YY = np.meshgrid(xs, ys)
        u_initial = 0.2 * np.sin(2.0 * np.pi * XX) * np.sin(np.pi * YY)
        return {
            "u": u_grid,
            "solver_info": {
                "mesh_resolution": mesh_resolution,
                "element_degree": element_degree,
                "ksp_type": ksp_type,
                "pc_type": pc_type,
                "rtol": rtol,
                "iterations": total_linear_iterations,
                "dt": dt,
                "n_steps": n_steps,
                "time_scheme": time_scheme,
                "nonlinear_iterations": nonlinear_iterations,
                "l2_error": l2_error,
            },
            "u_initial": u_initial,
        }

    return {"u": None, "solver_info": {}, "u_initial": None}


def _demo_case():
    return {
        "pde": {
            "time": {
                "t0": 0.0,
                "t_end": 0.2,
                "dt": 0.005,
                "scheme": "backward_euler",
            }
        },
        "output": {
            "grid": {
                "nx": 81,
                "ny": 81,
                "bbox": [0.0, 1.0, 0.0, 1.0],
            }
        },
        "agent_params": {
            "mesh_resolution": 32,
            "element_degree": 2,
            "epsilon": 0.05,
            "dt": 0.005,
            "newton_rtol": 1.0e-10,
            "newton_max_it": 20,
            "ksp_type": "gmres",
            "pc_type": "ilu",
            "rtol": 1.0e-9,
        },
    }


if __name__ == "__main__":
    t0 = time.perf_counter()
    result = solve(_demo_case())
    wall = time.perf_counter() - t0
    if MPI.COMM_WORLD.rank == 0:
        print(f"L2_ERROR: {result['solver_info']['l2_error']:.12e}")
        print(f"WALL_TIME: {wall:.12e}")
        print(f"OUTPUT_SHAPE: {result['u'].shape}")
