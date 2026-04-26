import time
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
import ufl
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc

ScalarType = PETSc.ScalarType

"""
DIAGNOSIS
equation_type:        convection_diffusion
spatial_dim:          2
domain_geometry:      rectangle
unknowns:             scalar
coupling:             none
linearity:            linear
time_dependence:      steady
stiffness:            stiff
dominant_physics:     mixed
peclet_or_reynolds:   high
solution_regularity:  smooth
bc_type:              all_dirichlet
special_notes:        manufactured_solution
"""

"""
METHOD
spatial_method:       fem
element_or_basis:     Lagrange_P2
stabilization:        supg
time_method:          none
nonlinear_solver:     none
linear_solver:        gmres
preconditioner:       ilu
special_treatment:    none
pde_skill:            convection_diffusion
"""


def _exact_u(x):
    return ufl.sin(ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])


def _build_and_solve(n, degree, ksp_type="gmres", pc_type="ilu", rtol=1.0e-9):
    comm = MPI.COMM_WORLD
    msh = mesh.create_unit_square(comm, n, n, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(msh, ("Lagrange", degree))

    x = ufl.SpatialCoordinate(msh)
    u_ex = _exact_u(x)

    eps = 0.005
    beta_np = np.array([20.0, 10.0], dtype=np.float64)
    beta = fem.Constant(msh, ScalarType(beta_np))
    f = -eps * ufl.div(ufl.grad(u_ex)) + ufl.dot(beta, ufl.grad(u_ex))

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    h = ufl.CellDiameter(msh)
    bnorm = float(np.linalg.norm(beta_np))
    tau = h / (2.0 * bnorm)

    a = (eps * ufl.inner(ufl.grad(u), ufl.grad(v)) + ufl.dot(beta, ufl.grad(u)) * v) * ufl.dx
    L = f * v * ufl.dx

    r_trial = -eps * ufl.div(ufl.grad(u)) + ufl.dot(beta, ufl.grad(u))
    streamline_test = ufl.dot(beta, ufl.grad(v))
    a += tau * r_trial * streamline_test * ufl.dx
    L += tau * f * streamline_test * ufl.dx

    u_bc = fem.Function(V)
    u_bc.interpolate(fem.Expression(u_ex, V.element.interpolation_points))
    fdim = msh.topology.dim - 1
    facets = mesh.locate_entities_boundary(
        msh,
        fdim,
        lambda X: np.isclose(X[0], 0.0)
        | np.isclose(X[0], 1.0)
        | np.isclose(X[1], 0.0)
        | np.isclose(X[1], 1.0),
    )
    dofs = fem.locate_dofs_topological(V, fdim, facets)
    bc = fem.dirichletbc(u_bc, dofs)

    petsc_options = {
        "ksp_type": ksp_type,
        "pc_type": pc_type,
        "ksp_rtol": rtol,
        "ksp_atol": 1.0e-12,
        "ksp_max_it": 5000,
    }
    if ksp_type == "gmres":
        petsc_options["ksp_gmres_restart"] = 200

    problem = petsc.LinearProblem(
        a,
        L,
        bcs=[bc],
        petsc_options=petsc_options,
        petsc_options_prefix=f"cd_{n}_{degree}_",
    )
    uh = problem.solve()
    uh.x.scatter_forward()

    l2_local = fem.assemble_scalar(fem.form((uh - u_ex) ** 2 * ufl.dx))
    l2_error = np.sqrt(comm.allreduce(l2_local, op=MPI.SUM))

    return msh, uh, problem.solver, float(l2_error)


def _sample_grid(msh, uh, grid):
    nx = int(grid["nx"])
    ny = int(grid["ny"])
    xmin, xmax, ymin, ymax = map(float, grid["bbox"])

    xs = np.linspace(xmin, xmax, nx)
    ys = np.linspace(ymin, ymax, ny)
    XX, YY = np.meshgrid(xs, ys, indexing="xy")
    pts = np.column_stack((XX.ravel(), YY.ravel(), np.zeros(nx * ny)))

    tree = geometry.bb_tree(msh, msh.topology.dim)
    candidates = geometry.compute_collisions_points(tree, pts)
    colliding = geometry.compute_colliding_cells(msh, candidates, pts)

    local_vals = np.full(nx * ny, np.nan, dtype=np.float64)
    p_local = []
    c_local = []
    idx_local = []

    for i in range(pts.shape[0]):
        links = colliding.links(i)
        if len(links) > 0:
            p_local.append(pts[i])
            c_local.append(links[0])
            idx_local.append(i)

    if p_local:
        vals = uh.eval(np.array(p_local, dtype=np.float64), np.array(c_local, dtype=np.int32))
        vals = np.asarray(vals).reshape(len(p_local), -1)[:, 0]
        local_vals[np.array(idx_local, dtype=np.int32)] = vals

    comm = msh.comm
    gathered = comm.gather(local_vals, root=0)
    if comm.rank == 0:
        vals = gathered[0]
        for arr in gathered[1:]:
            mask = np.isnan(vals) & ~np.isnan(arr)
            vals[mask] = arr[mask]
        return vals.reshape(ny, nx)
    return None


def solve(case_spec: dict) -> dict:
    target_time = 20.0
    start_all = time.perf_counter()

    configs = [(96, 2), (128, 2), (160, 2)]
    best = None

    for n, degree in configs:
        t0 = time.perf_counter()
        try:
            msh, uh, ksp, l2_error = _build_and_solve(n, degree, "gmres", "ilu", 1.0e-9)
        except Exception:
            msh, uh, ksp, l2_error = _build_and_solve(n, degree, "preonly", "lu", 1.0e-12)

        best = {
            "mesh": msh,
            "uh": uh,
            "ksp": ksp,
            "l2_error": l2_error,
            "mesh_resolution": n,
            "element_degree": degree,
        }

        elapsed = time.perf_counter() - t0
        total = time.perf_counter() - start_all
        if target_time - total < 0.35 * max(elapsed, 1.0e-6):
            break

    u_grid = _sample_grid(best["mesh"], best["uh"], case_spec["output"]["grid"])
    if MPI.COMM_WORLD.rank == 0:
        return {
            "u": u_grid,
            "solver_info": {
                "mesh_resolution": int(best["mesh_resolution"]),
                "element_degree": int(best["element_degree"]),
                "ksp_type": best["ksp"].getType(),
                "pc_type": best["ksp"].getPC().getType(),
                "rtol": 1.0e-9 if best["ksp"].getType() != "preonly" else 1.0e-12,
                "iterations": int(best["ksp"].getIterationNumber()),
                "l2_error": float(best["l2_error"]),
                "stabilization": "supg",
            },
        }
    return {}
