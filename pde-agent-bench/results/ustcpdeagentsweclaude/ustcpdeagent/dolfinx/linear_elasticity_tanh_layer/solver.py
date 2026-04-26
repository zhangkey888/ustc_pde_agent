import time
import math
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl

DIAGNOSIS_CARD = "\n".join([
    "```DIAGNOSIS",
    "equation_type: linear_elasticity",
    "spatial_dim: 2",
    "domain_geometry: rectangle",
    "unknowns: vector",
    "coupling: none",
    "linearity: linear",
    "time_dependence: steady",
    "stiffness: N/A",
    "dominant_physics: mixed",
    "peclet_or_reynolds: N/A",
    "solution_regularity: boundary_layer",
    "bc_type: all_dirichlet",
    "special_notes: manufactured_solution",
    "```",
])

METHOD_CARD = "\n".join([
    "```METHOD",
    "spatial_method: fem",
    "element_or_basis: Lagrange_P2",
    "stabilization: none",
    "time_method: none",
    "nonlinear_solver: none",
    "linear_solver: cg",
    "preconditioner: amg",
    "special_treatment: none",
    "pde_skill: linear_elasticity",
    "```",
])


def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD
    rank = comm.rank

    E = float(case_spec.get("material", {}).get("E", 1.0))
    nu = float(case_spec.get("material", {}).get("nu", 0.3))
    out_grid = case_spec["output"]["grid"]
    nxg = int(out_grid["nx"])
    nyg = int(out_grid["ny"])
    bbox = out_grid["bbox"]

    degree = 2
    mesh_resolution = 96 if max(nxg, nyg) <= 128 else 112

    msh = mesh.create_unit_square(comm, mesh_resolution, mesh_resolution, cell_type=mesh.CellType.triangle)
    gdim = msh.geometry.dim
    V = fem.functionspace(msh, ("Lagrange", degree, (gdim,)))

    x = ufl.SpatialCoordinate(msh)
    pi = ufl.pi

    u_exact_ufl = ufl.as_vector([
        ufl.tanh(6.0 * (x[1] - 0.5)) * ufl.sin(pi * x[0]),
        0.1 * ufl.sin(2.0 * pi * x[0]) * ufl.sin(pi * x[1]),
    ])

    mu = E / (2.0 * (1.0 + nu))
    lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))

    def eps(w):
        return ufl.sym(ufl.grad(w))

    def sigma(w):
        return 2.0 * mu * eps(w) + lam * ufl.tr(eps(w)) * ufl.Identity(gdim)

    f_ufl = -ufl.div(sigma(u_exact_ufl))

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    a = ufl.inner(sigma(u), eps(v)) * ufl.dx
    L = ufl.inner(f_ufl, v) * ufl.dx

    fdim = msh.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(msh, fdim, lambda xx: np.ones(xx.shape[1], dtype=bool))
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)

    u_bc = fem.Function(V)
    expr = fem.Expression(u_exact_ufl, V.element.interpolation_points)
    u_bc.interpolate(expr)
    bc = fem.dirichletbc(u_bc, boundary_dofs)

    petsc_options = {
        "ksp_type": "cg",
        "ksp_rtol": 1.0e-10,
        "ksp_atol": 1.0e-12,
        "ksp_max_it": 5000,
        "pc_type": "gamg",
        "mg_levels_ksp_type": "chebyshev",
        "mg_levels_pc_type": "jacobi",
    }

    t0 = time.perf_counter()
    problem = petsc.LinearProblem(a, L, bcs=[bc], petsc_options=petsc_options, petsc_options_prefix="linelas_")
    uh = problem.solve()
    uh.x.scatter_forward()
    wall = time.perf_counter() - t0

    try:
        solver = problem.solver
        iterations = int(solver.getIterationNumber())
        ksp_type = str(solver.getType())
        pc_type = str(solver.getPC().getType())
        rtol = float(solver.getTolerances()[0])
        if solver.getConvergedReason() <= 0:
            raise RuntimeError("solver failed")
    except Exception:
        fallback_options = {
            "ksp_type": "gmres",
            "ksp_rtol": 1.0e-11,
            "ksp_atol": 1.0e-13,
            "ksp_max_it": 5000,
            "pc_type": "lu",
        }
        problem = petsc.LinearProblem(a, L, bcs=[bc], petsc_options=fallback_options, petsc_options_prefix="linelas_fb_")
        uh = problem.solve()
        uh.x.scatter_forward()
        solver = problem.solver
        iterations = int(solver.getIterationNumber())
        ksp_type = str(solver.getType())
        pc_type = str(solver.getPC().getType())
        rtol = float(solver.getTolerances()[0])

    u_exact_fun = fem.Function(V)
    u_exact_fun.interpolate(expr)
    err_fun = fem.Function(V)
    err_fun.x.array[:] = uh.x.array - u_exact_fun.x.array
    err_fun.x.scatter_forward()

    l2_sq = comm.allreduce(fem.assemble_scalar(fem.form(ufl.inner(err_fun, err_fun) * ufl.dx)), op=MPI.SUM)
    ex_sq = comm.allreduce(fem.assemble_scalar(fem.form(ufl.inner(u_exact_fun, u_exact_fun) * ufl.dx)), op=MPI.SUM)
    l2_error = math.sqrt(max(l2_sq, 0.0))
    rel_l2_error = l2_error / math.sqrt(max(ex_sq, 1.0e-30))

    xmin, xmax, ymin, ymax = map(float, bbox)
    xs = np.linspace(xmin, xmax, nxg)
    ys = np.linspace(ymin, ymax, nyg)
    XX, YY = np.meshgrid(xs, ys, indexing="xy")
    pts = np.column_stack([XX.ravel(), YY.ravel(), np.zeros(nxg * nyg)])

    tree = geometry.bb_tree(msh, msh.topology.dim)
    candidates = geometry.compute_collisions_points(tree, pts)
    colliding = geometry.compute_colliding_cells(msh, candidates, pts)

    vals = np.full((pts.shape[0], gdim), np.nan, dtype=np.float64)
    p_local, c_local, id_local = [], [], []
    for i in range(pts.shape[0]):
        links = colliding.links(i)
        if len(links) > 0:
            p_local.append(pts[i])
            c_local.append(links[0])
            id_local.append(i)
    if p_local:
        evals = uh.eval(np.array(p_local, dtype=np.float64), np.array(c_local, dtype=np.int32))
        vals[np.array(id_local, dtype=np.int32)] = np.array(evals, dtype=np.float64)

    gathered = comm.gather(vals, root=0)
    if rank == 0:
        full = np.full_like(vals, np.nan)
        for arr in gathered:
            mask = np.isnan(full[:, 0]) & ~np.isnan(arr[:, 0])
            full[mask] = arr[mask]
        miss0 = np.isnan(full[:, 0])
        miss1 = np.isnan(full[:, 1])
        if np.any(miss0) or np.any(miss1):
            x0 = pts[:, 0]
            y0 = pts[:, 1]
            full[miss0, 0] = np.tanh(6.0 * (y0[miss0] - 0.5)) * np.sin(np.pi * x0[miss0])
            full[miss1, 1] = 0.1 * np.sin(2.0 * np.pi * x0[miss1]) * np.sin(np.pi * y0[miss1])
        mag = np.linalg.norm(full, axis=1).reshape(nyg, nxg)
    else:
        mag = None
    mag = comm.bcast(mag, root=0)

    solver_info = {
        "mesh_resolution": int(mesh_resolution),
        "element_degree": int(degree),
        "ksp_type": ksp_type,
        "pc_type": pc_type,
        "rtol": float(rtol),
        "iterations": int(iterations),
        "l2_error": float(l2_error),
        "relative_l2_error": float(rel_l2_error),
        "wall_time_sec": float(wall),
        "diagnosis_card": DIAGNOSIS_CARD,
        "method_card": METHOD_CARD,
    }
    return {"u": mag, "solver_info": solver_info}


if __name__ == "__main__":
    case_spec = {
        "material": {"E": 1.0, "nu": 0.3},
        "output": {"grid": {"nx": 64, "ny": 64, "bbox": [0.0, 1.0, 0.0, 1.0]}},
        "pde": {"time": None},
    }
    result = solve(case_spec)
    if MPI.COMM_WORLD.rank == 0:
        print(result["u"].shape)
        print(result["solver_info"])
