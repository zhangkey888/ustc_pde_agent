import time
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
import ufl
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc

# ```DIAGNOSIS
# equation_type:        helmholtz
# spatial_dim:          2
# domain_geometry:      rectangle
# unknowns:             scalar
# coupling:             none
# linearity:            linear
# time_dependence:      steady
# stiffness:            N/A
# dominant_physics:     wave
# peclet_or_reynolds:   N/A
# solution_regularity:  smooth
# bc_type:              all_dirichlet
# special_notes:        none
# ```
#
# ```METHOD
# spatial_method:       fem
# element_or_basis:     Lagrange_P1
# stabilization:        none
# time_method:          none
# nonlinear_solver:     none
# linear_solver:        gmres
# preconditioner:       ilu
# special_treatment:    none
# pde_skill:            helmholtz
# ```

ScalarType = PETSc.ScalarType


def _boundary_marker(x):
    return (
        np.isclose(x[0], 0.0)
        | np.isclose(x[0], 1.0)
        | np.isclose(x[1], 0.0)
        | np.isclose(x[1], 1.0)
    )


def _sample_function_on_grid(u_func, domain, grid_spec):
    nx = int(grid_spec["nx"])
    ny = int(grid_spec["ny"])
    xmin, xmax, ymin, ymax = map(float, grid_spec["bbox"])
    xs = np.linspace(xmin, xmax, nx)
    ys = np.linspace(ymin, ymax, ny)
    XX, YY = np.meshgrid(xs, ys, indexing="xy")
    pts = np.column_stack([XX.ravel(), YY.ravel(), np.zeros(nx * ny)])

    tree = geometry.bb_tree(domain, domain.topology.dim)
    candidates = geometry.compute_collisions_points(tree, pts)
    colliding = geometry.compute_colliding_cells(domain, candidates, pts)

    local_vals = np.full(nx * ny, np.nan, dtype=np.float64)
    psel, csel, ids = [], [], []
    for i in range(nx * ny):
        links = colliding.links(i)
        if len(links) > 0:
            psel.append(pts[i])
            csel.append(links[0])
            ids.append(i)

    if ids:
        vals = u_func.eval(np.array(psel, dtype=np.float64), np.array(csel, dtype=np.int32))
        local_vals[np.array(ids, dtype=np.int32)] = np.real_if_close(np.asarray(vals).reshape(-1))

    gathered = domain.comm.gather(local_vals, root=0)
    if domain.comm.rank == 0:
        global_vals = np.full(nx * ny, np.nan, dtype=np.float64)
        for arr in gathered:
            mask = np.isfinite(arr)
            global_vals[mask] = arr[mask]
        return np.nan_to_num(global_vals, nan=0.0).reshape(ny, nx)
    return None


def _verification(domain, uh, f_fun, k_value):
    l2_local = fem.assemble_scalar(fem.form(uh * uh * ufl.dx))
    h1_local = fem.assemble_scalar(fem.form(ufl.inner(ufl.grad(uh), ufl.grad(uh)) * ufl.dx))
    fu_local = fem.assemble_scalar(fem.form(f_fun * uh * ufl.dx))
    l2 = domain.comm.allreduce(l2_local, op=MPI.SUM)
    h1 = domain.comm.allreduce(h1_local, op=MPI.SUM)
    fu = domain.comm.allreduce(fu_local, op=MPI.SUM)
    balance = h1 - (k_value ** 2) * l2 - fu
    denom = max(1.0, abs(h1) + abs((k_value ** 2) * l2) + abs(fu))
    return {
        "verification_type": "discrete_energy_balance",
        "l2_norm": float(np.sqrt(max(l2, 0.0))),
        "h1_seminorm": float(np.sqrt(max(h1, 0.0))),
        "energy_balance_rel": float(abs(balance) / denom),
    }


def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD
    t0 = time.time()

    k_value = float(case_spec.get("pde", {}).get("wavenumber", 22.0))
    grid_spec = case_spec["output"]["grid"]
    time_limit = float(case_spec.get("time_limit_sec", 370.381))

    degree = 1
    n = 56 if (comm.size == 1 and time_limit > 60.0) else 40
    rtol = 1.0e-8

    domain = mesh.create_unit_square(comm, n, n, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", degree))

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    x = ufl.SpatialCoordinate(domain)
    f_expr = ufl.sin(10.0 * ufl.pi * x[0]) * ufl.sin(8.0 * ufl.pi * x[1])

    f_fun = fem.Function(V)
    f_fun.interpolate(fem.Expression(f_expr, V.element.interpolation_points))

    a = (ufl.inner(ufl.grad(u), ufl.grad(v)) - ScalarType(k_value ** 2) * u * v) * ufl.dx
    L = ufl.inner(f_fun, v) * ufl.dx

    fdim = domain.topology.dim - 1
    facets = mesh.locate_entities_boundary(domain, fdim, _boundary_marker)
    dofs = fem.locate_dofs_topological(V, fdim, facets)
    bc = fem.dirichletbc(ScalarType(0.0), dofs, V)

    info = None
    uh = None
    for i, opts in enumerate((
        {"ksp_type": "gmres", "pc_type": "ilu", "ksp_rtol": rtol, "ksp_atol": 1e-12, "ksp_max_it": 1500},
        {"ksp_type": "preonly", "pc_type": "lu"},
    )):
        try:
            problem = petsc.LinearProblem(
                a, L, bcs=[bc], petsc_options=opts, petsc_options_prefix=f"helm_{i}_"
            )
            uh = problem.solve()
            uh.x.scatter_forward()
            ksp = problem.solver
            info = {
                "ksp_type": str(ksp.getType()),
                "pc_type": str(ksp.getPC().getType()),
                "iterations": int(ksp.getIterationNumber()),
            }
            break
        except Exception:
            uh = None

    if uh is None:
        raise RuntimeError("Helmholtz solve failed")

    verification = _verification(domain, uh, f_fun, k_value)
    u_grid = _sample_function_on_grid(uh, domain, grid_spec)
    wall = time.time() - t0

    if comm.rank == 0:
        return {
            "u": np.asarray(u_grid, dtype=np.float64),
            "solver_info": {
                "mesh_resolution": int(n),
                "element_degree": int(degree),
                "ksp_type": info["ksp_type"],
                "pc_type": info["pc_type"],
                "rtol": float(rtol),
                "iterations": int(info["iterations"]),
                "verification": verification,
                "wall_time_sec": float(wall),
            },
        }
    return {"u": None, "solver_info": None}
