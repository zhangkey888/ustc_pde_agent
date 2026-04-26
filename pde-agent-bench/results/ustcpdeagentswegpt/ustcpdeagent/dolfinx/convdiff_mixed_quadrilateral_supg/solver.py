import math
import time
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl

ScalarType = PETSc.ScalarType


def _boundary_all(x):
    return np.ones(x.shape[1], dtype=bool)


def _sample_on_grid(domain, uh, grid):
    nx = int(grid["nx"])
    ny = int(grid["ny"])
    xmin, xmax, ymin, ymax = grid["bbox"]
    xs = np.linspace(xmin, xmax, nx, dtype=np.float64)
    ys = np.linspace(ymin, ymax, ny, dtype=np.float64)
    XX, YY = np.meshgrid(xs, ys, indexing="xy")
    pts = np.column_stack((XX.ravel(), YY.ravel(), np.zeros(nx * ny, dtype=np.float64)))

    tree = geometry.bb_tree(domain, domain.topology.dim)
    candidates = geometry.compute_collisions_points(tree, pts)
    colliding = geometry.compute_colliding_cells(domain, candidates, pts)

    vals = np.full(nx * ny, np.nan, dtype=np.float64)
    p_local = []
    c_local = []
    ids = []
    for i in range(pts.shape[0]):
        links = colliding.links(i)
        if len(links) > 0:
            p_local.append(pts[i])
            c_local.append(links[0])
            ids.append(i)

    if ids:
        evaluated = uh.eval(np.array(p_local, dtype=np.float64), np.array(c_local, dtype=np.int32))
        vals[np.array(ids, dtype=np.int32)] = np.asarray(evaluated, dtype=np.float64).reshape(-1)

    gathered = domain.comm.gather(vals, root=0)
    if domain.comm.rank == 0:
        out = np.full(nx * ny, np.nan, dtype=np.float64)
        for arr in gathered:
            mask = ~np.isnan(arr)
            out[mask] = arr[mask]
        if np.isnan(out).any():
            out[np.isnan(out)] = 0.0
        return out.reshape((ny, nx))
    return None


def _build_problem(n, degree=1, tau_scale=1.0):
    comm = MPI.COMM_WORLD
    domain = mesh.create_rectangle(
        comm,
        [np.array([0.0, 0.0], dtype=np.float64), np.array([1.0, 1.0], dtype=np.float64)],
        [n, n],
        cell_type=mesh.CellType.quadrilateral,
    )
    V = fem.functionspace(domain, ("Lagrange", degree))
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    x = ufl.SpatialCoordinate(domain)

    eps_val = 0.01
    beta_np = np.array([14.0, 6.0], dtype=np.float64)
    eps = fem.Constant(domain, ScalarType(eps_val))
    beta = fem.Constant(domain, beta_np)
    beta_u = ufl.as_vector((beta[0], beta[1]))

    u_exact = ufl.sin(ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])
    grad_u_exact = ufl.grad(u_exact)
    f_expr = -eps * ufl.div(ufl.grad(u_exact)) + ufl.dot(beta_u, grad_u_exact)

    h = ufl.CellDiameter(domain)
    beta_norm = ufl.sqrt(ufl.dot(beta_u, beta_u))
    tau = tau_scale * h / (2.0 * beta_norm + 4.0 * eps / h)

    a = (
        eps * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
        + ufl.dot(beta_u, ufl.grad(u)) * v * ufl.dx
        + tau * (-eps * ufl.div(ufl.grad(u)) + ufl.dot(beta_u, ufl.grad(u))) * ufl.dot(beta_u, ufl.grad(v)) * ufl.dx
    )
    L = (
        f_expr * v * ufl.dx
        + tau * f_expr * ufl.dot(beta_u, ufl.grad(v)) * ufl.dx
    )

    u_bc = fem.Function(V)
    u_bc.interpolate(fem.Expression(u_exact, V.element.interpolation_points))
    fdim = domain.topology.dim - 1
    facets = mesh.locate_entities_boundary(domain, fdim, _boundary_all)
    dofs = fem.locate_dofs_topological(V, fdim, facets)
    bc = fem.dirichletbc(u_bc, dofs)

    return domain, V, a, L, bc, u_bc, u_exact


def _solve_once(n, degree=1, tau_scale=1.0):
    domain, V, a, L, bc, u_bc, u_exact = _build_problem(n, degree, tau_scale)
    uh = fem.Function(V)

    start = time.perf_counter()
    problem = petsc.LinearProblem(
        a,
        L,
        bcs=[bc],
        petsc_options_prefix=f"cd_{n}_{degree}_",
        petsc_options={
            "ksp_type": "preonly",
            "pc_type": "lu",
        },
    )
    uh = problem.solve()
    uh.x.scatter_forward()
    elapsed = time.perf_counter() - start

    e = fem.Function(V)
    e.x.array[:] = uh.x.array - u_bc.x.array
    err_l2 = math.sqrt(domain.comm.allreduce(fem.assemble_scalar(fem.form(ufl.inner(e, e) * ufl.dx)), op=MPI.SUM))

    h1_num = domain.comm.allreduce(
        fem.assemble_scalar(fem.form(ufl.inner(ufl.grad(e), ufl.grad(e)) * ufl.dx)), op=MPI.SUM
    )
    err_h1 = math.sqrt(max(h1_num, 0.0))

    return {
        "domain": domain,
        "uh": uh,
        "n": int(n),
        "degree": int(degree),
        "iterations": 1,
        "solve_time": float(elapsed),
        "err_l2": float(err_l2),
        "err_h1_semi": float(err_h1),
        "ksp_type": "preonly",
        "pc_type": "lu",
        "rtol": 1.0e-12,
    }


def solve(case_spec: dict) -> dict:
    # ```DIAGNOSIS
    # equation_type: convection_diffusion
    # spatial_dim: 2
    # domain_geometry: rectangle
    # unknowns: scalar
    # coupling: none
    # linearity: linear
    # time_dependence: steady
    # stiffness: stiff
    # dominant_physics: mixed
    # peclet_or_reynolds: high
    # solution_regularity: smooth
    # bc_type: all_dirichlet
    # special_notes: manufactured_solution
    # ```
    # ```METHOD
    # spatial_method: fem
    # element_or_basis: Lagrange_P1
    # stabilization: supg
    # time_method: none
    # nonlinear_solver: none
    # linear_solver: direct_lu
    # preconditioner: none
    # special_treatment: none
    # pde_skill: convection_diffusion / reaction_diffusion / biharmonic
    # ```

    grid = case_spec["output"]["grid"]

    best = _solve_once(n=52, degree=1, tau_scale=1.0)
    u_grid = _sample_on_grid(best["domain"], best["uh"], grid)

    if best["domain"].comm.rank == 0:
        return {
            "u": u_grid,
            "solver_info": {
                "mesh_resolution": int(best["n"]),
                "element_degree": int(best["degree"]),
                "ksp_type": str(best["ksp_type"]),
                "pc_type": str(best["pc_type"]),
                "rtol": float(best["rtol"]),
                "iterations": int(best["iterations"]),
                "l2_error": float(best["err_l2"]),
                "h1_semi_error": float(best["err_h1_semi"]),
            },
        }
    return {"u": None, "solver_info": {}}
