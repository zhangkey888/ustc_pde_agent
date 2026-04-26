import time
import math
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl

ScalarType = PETSc.ScalarType

DIAGNOSIS_CARD = "equation_type=convection_diffusion; spatial_dim=2; domain_geometry=rectangle; unknowns=scalar; coupling=none; linearity=linear; time_dependence=steady; stiffness=stiff; dominant_physics=mixed; peclet_or_reynolds=high; solution_regularity=smooth; bc_type=all_dirichlet; special_notes=manufactured_solution"
METHOD_CARD = "spatial_method=fem; element_or_basis=Lagrange_P2; stabilization=supg; time_method=none; nonlinear_solver=none; linear_solver=gmres; preconditioner=ilu; special_treatment=none; pde_skill=convection_diffusion"


def _coth(z):
    return ufl.cosh(z) / ufl.sinh(z)


def _u_exact_ufl(msh):
    x = ufl.SpatialCoordinate(msh)
    return ufl.sin(2.0 * ufl.pi * x[0]) * ufl.sin(2.0 * ufl.pi * x[1])


def _u_exact_numpy(x):
    return np.sin(2.0 * np.pi * x[0]) * np.sin(2.0 * np.pi * x[1])


def _rhs_ufl(msh, eps_c, beta):
    uex = _u_exact_ufl(msh)
    return -eps_c * ufl.div(ufl.grad(uex)) + ufl.dot(beta, ufl.grad(uex))


def _sample_on_grid(msh, uh, grid):
    nx = int(grid["nx"])
    ny = int(grid["ny"])
    xmin, xmax, ymin, ymax = grid["bbox"]
    xs = np.linspace(xmin, xmax, nx)
    ys = np.linspace(ymin, ymax, ny)
    XX, YY = np.meshgrid(xs, ys, indexing="xy")
    pts = np.column_stack([XX.ravel(), YY.ravel(), np.zeros(nx * ny)]).astype(np.float64)

    tree = geometry.bb_tree(msh, msh.topology.dim)
    candidates = geometry.compute_collisions_points(tree, pts)
    colliding = geometry.compute_colliding_cells(msh, candidates, pts)

    values = np.full(nx * ny, np.nan, dtype=np.float64)
    points_on_proc, cells_on_proc, ids = [], [], []
    for i in range(pts.shape[0]):
        links = colliding.links(i)
        if len(links) > 0:
            points_on_proc.append(pts[i])
            cells_on_proc.append(links[0])
            ids.append(i)

    if points_on_proc:
        vals = uh.eval(np.array(points_on_proc, dtype=np.float64),
                       np.array(cells_on_proc, dtype=np.int32))
        vals = np.asarray(vals).reshape(len(points_on_proc), -1)[:, 0]
        values[np.array(ids, dtype=np.int32)] = vals

    values = np.where(np.isnan(values), 0.0, values)
    values = msh.comm.allreduce(values, op=MPI.SUM)
    return values.reshape(ny, nx)


def _build_and_solve(nx, degree, epsilon, beta_vec, rtol, use_supg):
    msh = mesh.create_unit_square(MPI.COMM_WORLD, nx, nx, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(msh, ("Lagrange", degree))
    beta = fem.Constant(msh, np.array(beta_vec, dtype=np.float64))
    eps_c = fem.Constant(msh, ScalarType(epsilon))

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    u_exact = _u_exact_ufl(msh)
    f = _rhs_ufl(msh, eps_c, beta)

    uD = fem.Function(V)
    uD.interpolate(_u_exact_numpy)

    fdim = msh.topology.dim - 1
    facets = mesh.locate_entities_boundary(msh, fdim, lambda x: np.ones(x.shape[1], dtype=bool))
    bdofs = fem.locate_dofs_topological(V, fdim, facets)
    bc = fem.dirichletbc(uD, bdofs)

    a = eps_c * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx + ufl.dot(beta, ufl.grad(u)) * v * ufl.dx
    L = f * v * ufl.dx

    if use_supg:
        h = ufl.CellDiameter(msh)
        beta_norm = ufl.sqrt(ufl.dot(beta, beta) + ScalarType(1.0e-16))
        pe = beta_norm * h / (2.0 * eps_c + ScalarType(1.0e-16))
        tau = h / (2.0 * beta_norm + ScalarType(1.0e-16)) * (_coth(pe) - 1.0 / (pe + ScalarType(1.0e-16)))
        a += tau * ufl.dot(beta, ufl.grad(v)) * (-eps_c * ufl.div(ufl.grad(u)) + ufl.dot(beta, ufl.grad(u))) * ufl.dx
        L += tau * ufl.dot(beta, ufl.grad(v)) * f * ufl.dx

    try:
        problem = petsc.LinearProblem(
            a, L, bcs=[bc],
            petsc_options_prefix=f"cd_{nx}_{degree}_",
            petsc_options={
                "ksp_type": "gmres",
                "ksp_rtol": rtol,
                "ksp_atol": 1.0e-12,
                "pc_type": "ilu",
            },
        )
        uh = problem.solve()
        uh.x.scatter_forward()
        ksp = problem.solver
        info = {
            "iterations": int(ksp.getIterationNumber()),
            "ksp_type": str(ksp.getType()),
            "pc_type": str(ksp.getPC().getType()),
        }
    except Exception:
        problem = petsc.LinearProblem(
            a, L, bcs=[bc],
            petsc_options_prefix=f"cdlu_{nx}_{degree}_",
            petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
        )
        uh = problem.solve()
        uh.x.scatter_forward()
        ksp = problem.solver
        info = {
            "iterations": int(ksp.getIterationNumber()),
            "ksp_type": str(ksp.getType()),
            "pc_type": str(ksp.getPC().getType()),
        }

    err2_local = fem.assemble_scalar(fem.form((uh - u_exact) ** 2 * ufl.dx))
    err2 = msh.comm.allreduce(err2_local, op=MPI.SUM)
    l2 = math.sqrt(err2)
    return msh, uh, l2, info


def solve(case_spec: dict) -> dict:
    pde = case_spec.get("pde", {})
    grid = case_spec.get("output", {}).get("grid", {"nx": 64, "ny": 64, "bbox": [0.0, 1.0, 0.0, 1.0]})
    epsilon = float(pde.get("epsilon", 0.02))
    beta_in = pde.get("beta", [6.0, 2.0])
    beta_vec = (float(beta_in[0]), float(beta_in[1]))
    time_limit = float(case_spec.get("time_limit", 1.731))
    budget = 0.5 * time_limit

    configs = [
        (48, 2, 1.0e-8, True),
        (64, 2, 1.0e-9, True),
        (80, 2, 1.0e-9, True),
        (96, 2, 1.0e-9, True),
    ]

    best = None
    t_start = time.perf_counter()
    for nx, degree, rtol, use_supg in configs:
        t0 = time.perf_counter()
        msh, uh, l2, info = _build_and_solve(nx, degree, epsilon, beta_vec, rtol, use_supg)
        best = (msh, uh, nx, degree, rtol, info, l2)
        t1 = time.perf_counter()
        if (t1 - t_start) > budget or (t1 - t0) > budget:
            break

    msh, uh, nx, degree, rtol, info, l2 = best
    u_grid = _sample_on_grid(msh, uh, grid)
    solver_info = {
        "mesh_resolution": int(nx),
        "element_degree": int(degree),
        "ksp_type": str(info["ksp_type"]),
        "pc_type": str(info["pc_type"]),
        "rtol": float(rtol),
        "iterations": int(info["iterations"]),
    }
    return {
        "u": u_grid,
        "solver_info": solver_info,
        "diagnosis": DIAGNOSIS_CARD,
        "method": METHOD_CARD,
        "l2_error": float(l2),
    }


if __name__ == "__main__":
    case_spec = {
        "pde": {"epsilon": 0.02, "beta": [6.0, 2.0], "time": None},
        "output": {"grid": {"nx": 64, "ny": 64, "bbox": [0.0, 1.0, 0.0, 1.0]}},
        "time_limit": 1.731,
    }
    t0 = time.perf_counter()
    result = solve(case_spec)
    wall = time.perf_counter() - t0
    print("L2_ERROR:", result["l2_error"])
    print("WALL_TIME:", wall)
    print("U_SHAPE:", result["u"].shape)
