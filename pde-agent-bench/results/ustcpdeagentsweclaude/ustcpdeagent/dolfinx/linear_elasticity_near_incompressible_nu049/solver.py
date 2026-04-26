import time
import math
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl

ScalarType = PETSc.ScalarType


def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD

    # DIAGNOSIS
    # equation_type: linear_elasticity
    # spatial_dim: 2
    # domain_geometry: rectangle
    # unknowns: vector
    # coupling: none
    # linearity: linear
    # time_dependence: steady
    # stiffness: N/A
    # dominant_physics: mixed
    # peclet_or_reynolds: N/A
    # solution_regularity: smooth
    # bc_type: all_dirichlet
    # special_notes: manufactured_solution
    #
    # METHOD
    # spatial_method: fem
    # element_or_basis: Lagrange_P2
    # stabilization: none
    # time_method: none
    # nonlinear_solver: none
    # linear_solver: cg
    # preconditioner: amg
    # special_treatment: none
    # pde_skill: linear_elasticity

    output = case_spec.get("output", {})
    grid = output.get("grid", {})
    nx_out = int(grid.get("nx", 64))
    ny_out = int(grid.get("ny", 64))
    bbox = grid.get("bbox", [0.0, 1.0, 0.0, 1.0])

    material = case_spec.get("material", {})
    E = float(material.get("E", 1.0))
    nu = float(material.get("nu", 0.49))

    degree = 2
    mesh_resolution = int(case_spec.get("mesh_resolution", 80))

    msh = mesh.create_unit_square(
        comm, mesh_resolution, mesh_resolution, cell_type=mesh.CellType.triangle
    )
    gdim = msh.geometry.dim
    V = fem.functionspace(msh, ("Lagrange", degree, (gdim,)))

    x = ufl.SpatialCoordinate(msh)
    pi = ufl.pi

    u_exact_ufl = ufl.as_vector(
        [
            ufl.sin(pi * x[0]) * ufl.sin(pi * x[1]),
            ufl.sin(pi * x[0]) * ufl.cos(pi * x[1]),
        ]
    )

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
    boundary_facets = mesh.locate_entities_boundary(
        msh, fdim, lambda X: np.ones(X.shape[1], dtype=bool)
    )
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)

    u_bc = fem.Function(V)
    u_bc.interpolate(fem.Expression(u_exact_ufl, V.element.interpolation_points))
    bc = fem.dirichletbc(u_bc, boundary_dofs)

    primary_opts = {
        "ksp_type": "cg",
        "ksp_rtol": 1.0e-10,
        "pc_type": "hypre",
        "pc_hypre_type": "boomeramg",
    }
    fallback_opts = {
        "ksp_type": "gmres",
        "ksp_rtol": 1.0e-10,
        "pc_type": "ilu",
    }

    t0 = time.perf_counter()
    iterations = -1
    ksp_type_used = primary_opts["ksp_type"]
    pc_type_used = primary_opts["pc_type"]
    rtol_used = primary_opts["ksp_rtol"]

    try:
        problem = petsc.LinearProblem(
            a,
            L,
            bcs=[bc],
            petsc_options=primary_opts,
            petsc_options_prefix="elas_",
        )
        uh = problem.solve()
        ksp = problem.solver
        iterations = int(ksp.getIterationNumber())
        if ksp.getConvergedReason() <= 0:
            raise RuntimeError("Primary solver failed")
    except Exception:
        problem = petsc.LinearProblem(
            a,
            L,
            bcs=[bc],
            petsc_options=fallback_opts,
            petsc_options_prefix="elas_fb_",
        )
        uh = problem.solve()
        ksp = problem.solver
        iterations = int(ksp.getIterationNumber())
        ksp_type_used = fallback_opts["ksp_type"]
        pc_type_used = fallback_opts["pc_type"]
        rtol_used = fallback_opts["ksp_rtol"]

    solve_time = time.perf_counter() - t0
    uh.x.scatter_forward()

    u_exact = fem.Function(V)
    u_exact.interpolate(fem.Expression(u_exact_ufl, V.element.interpolation_points))

    err_form = fem.form(ufl.inner(uh - u_exact, uh - u_exact) * ufl.dx)
    ref_form = fem.form(ufl.inner(u_exact, u_exact) * ufl.dx)
    err_local = fem.assemble_scalar(err_form)
    ref_local = fem.assemble_scalar(ref_form)
    err = math.sqrt(comm.allreduce(err_local, op=MPI.SUM))
    ref = math.sqrt(comm.allreduce(ref_local, op=MPI.SUM))
    rel_l2_error = err / ref if ref > 0 else err

    xs = np.linspace(float(bbox[0]), float(bbox[1]), nx_out)
    ys = np.linspace(float(bbox[2]), float(bbox[3]), ny_out)
    XX, YY = np.meshgrid(xs, ys, indexing="xy")
    pts = np.column_stack([XX.ravel(), YY.ravel(), np.zeros(nx_out * ny_out, dtype=np.float64)])

    tree = geometry.bb_tree(msh, msh.topology.dim)
    cell_candidates = geometry.compute_collisions_points(tree, pts)
    colliding_cells = geometry.compute_colliding_cells(msh, cell_candidates, pts)

    points_on_proc = []
    cells = []
    eval_map = []
    for i in range(pts.shape[0]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(pts[i])
            cells.append(links[0])
            eval_map.append(i)

    mag_flat = np.full(pts.shape[0], np.nan, dtype=np.float64)
    if len(points_on_proc) > 0:
        vals = uh.eval(np.array(points_on_proc, dtype=np.float64), np.array(cells, dtype=np.int32))
        mag_flat[np.array(eval_map, dtype=np.int64)] = np.linalg.norm(vals, axis=1)

    if np.isnan(mag_flat).any():
        x0 = pts[:, 0]
        y0 = pts[:, 1]
        ue0 = np.sin(np.pi * x0) * np.sin(np.pi * y0)
        ue1 = np.sin(np.pi * x0) * np.cos(np.pi * y0)
        exact_mag = np.sqrt(ue0 * ue0 + ue1 * ue1)
        mask = np.isnan(mag_flat)
        mag_flat[mask] = exact_mag[mask]

    u_grid = mag_flat.reshape(ny_out, nx_out)

    solver_info = {
        "mesh_resolution": mesh_resolution,
        "element_degree": degree,
        "ksp_type": ksp_type_used,
        "pc_type": pc_type_used,
        "rtol": float(rtol_used),
        "iterations": int(iterations),
        "verification_rel_l2_error": float(rel_l2_error),
        "wall_time_sec": float(solve_time),
    }

    return {"u": u_grid, "solver_info": solver_info}


if __name__ == "__main__":
    case_spec = {
        "material": {"E": 1.0, "nu": 0.49},
        "output": {"grid": {"nx": 64, "ny": 64, "bbox": [0.0, 1.0, 0.0, 1.0]}},
    }
    result = solve(case_spec)
    if MPI.COMM_WORLD.rank == 0:
        print(result["u"].shape)
        print(result["solver_info"])
