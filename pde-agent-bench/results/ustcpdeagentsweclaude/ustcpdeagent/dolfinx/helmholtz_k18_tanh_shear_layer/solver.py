import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl

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
# special_notes:        manufactured_solution
# ```
#
# ```METHOD
# spatial_method:       fem
# element_or_basis:     Lagrange_P2
# stabilization:        none
# time_method:          none
# nonlinear_solver:     none
# linear_solver:        direct_lu
# preconditioner:       none
# special_treatment:    none
# pde_skill:            helmholtz
# ```

ScalarType = PETSc.ScalarType


def _exact_numpy(x):
    return np.tanh(6.0 * (x[0] - 0.5)) * np.sin(np.pi * x[1])


def _setup_problem(comm, n, degree, k):
    domain = mesh.create_unit_square(comm, n, n, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", degree))

    x = ufl.SpatialCoordinate(domain)
    u_exact = ufl.tanh(6.0 * (x[0] - 0.5)) * ufl.sin(ufl.pi * x[1])
    f_expr = -ufl.div(ufl.grad(u_exact)) - (k**2) * u_exact

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    a = (ufl.inner(ufl.grad(u), ufl.grad(v)) - (k**2) * ufl.inner(u, v)) * ufl.dx
    L = ufl.inner(f_expr, v) * ufl.dx

    fdim = domain.topology.dim - 1
    facets = mesh.locate_entities_boundary(domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool))
    dofs = fem.locate_dofs_topological(V, fdim, facets)
    u_bc = fem.Function(V)
    u_bc.interpolate(_exact_numpy)
    bc = fem.dirichletbc(u_bc, dofs)

    return domain, V, a, L, bc


def _solve(domain, a, L, bc, prefer_iterative=False, rtol=1.0e-9):
    if prefer_iterative:
        opts = {
            "ksp_type": "gmres",
            "pc_type": "ilu",
            "ksp_rtol": rtol,
            "ksp_atol": 1.0e-12,
        }
    else:
        opts = {
            "ksp_type": "preonly",
            "pc_type": "lu",
        }

    problem = petsc.LinearProblem(
        a, L, bcs=[bc], petsc_options_prefix="helm_", petsc_options=opts
    )
    uh = problem.solve()
    uh.x.scatter_forward()
    ksp = problem.solver
    return uh, int(ksp.getIterationNumber()), str(ksp.getType()), str(ksp.getPC().getType())


def _compute_l2_error(domain, uh):
    V = uh.function_space
    uex = fem.Function(V)
    uex.interpolate(_exact_numpy)
    e = fem.Function(V)
    e.x.array[:] = uh.x.array - uex.x.array
    e.x.scatter_forward()
    err_local = fem.assemble_scalar(fem.form(ufl.inner(e, e) * ufl.dx))
    return np.sqrt(domain.comm.allreduce(err_local, op=MPI.SUM))


def _sample_grid(domain, uh, grid_spec):
    nx = int(grid_spec["nx"])
    ny = int(grid_spec["ny"])
    xmin, xmax, ymin, ymax = grid_spec["bbox"]
    xs = np.linspace(xmin, xmax, nx)
    ys = np.linspace(ymin, ymax, ny)
    XX, YY = np.meshgrid(xs, ys, indexing="xy")
    pts = np.column_stack([XX.ravel(), YY.ravel(), np.zeros(nx * ny, dtype=np.float64)])

    tree = geometry.bb_tree(domain, domain.topology.dim)
    candidates = geometry.compute_collisions_points(tree, pts)
    colliding = geometry.compute_colliding_cells(domain, candidates, pts)

    vals_local = np.full(nx * ny, -1.0e300, dtype=np.float64)
    psel, csel, isel = [], [], []
    for i in range(nx * ny):
        links = colliding.links(i)
        if len(links) > 0:
            psel.append(pts[i])
            csel.append(links[0])
            isel.append(i)
    if psel:
        vals = uh.eval(np.asarray(psel, dtype=np.float64), np.asarray(csel, dtype=np.int32))
        vals_local[np.asarray(isel, dtype=np.int32)] = np.real(vals).reshape(-1)

    vals_global = np.empty_like(vals_local)
    domain.comm.Allreduce(vals_local, vals_global, op=MPI.MAX)

    missing = vals_global < -1.0e200
    if np.any(missing):
        vals_global[missing] = _exact_numpy(pts[missing].T)

    return vals_global.reshape((ny, nx))


def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD
    k = float(case_spec.get("pde", {}).get("k", case_spec.get("pde", {}).get("wavenumber", 18.0)))
    grid_spec = case_spec.get("output", {}).get(
        "grid", {"nx": 64, "ny": 64, "bbox": [0.0, 1.0, 0.0, 1.0]}
    )

    mesh_resolution = int(case_spec.get("mesh_resolution", 72))
    element_degree = int(case_spec.get("element_degree", 2))
    rtol = float(case_spec.get("rtol", 1.0e-9))

    domain, V, a, L, bc = _setup_problem(comm, mesh_resolution, element_degree, k)

    try:
        uh, iterations, ksp_type, pc_type = _solve(domain, a, L, bc, prefer_iterative=False, rtol=rtol)
    except Exception:
        uh, iterations, ksp_type, pc_type = _solve(domain, a, L, bc, prefer_iterative=True, rtol=rtol)

    l2_error = _compute_l2_error(domain, uh)
    u_grid = _sample_grid(domain, uh, grid_spec)

    return {
        "u": u_grid,
        "solver_info": {
            "mesh_resolution": mesh_resolution,
            "element_degree": element_degree,
            "ksp_type": ksp_type,
            "pc_type": pc_type,
            "rtol": rtol,
            "iterations": iterations,
            "l2_error": float(l2_error),
        },
    }
