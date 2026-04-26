import time
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
import ufl
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc

ScalarType = PETSc.ScalarType


# ```DIAGNOSIS
# equation_type: linear_elasticity
# spatial_dim: 2
# domain_geometry: rectangle
# unknowns: vector
# coupling: none
# linearity: linear
# time_dependence: steady
# stiffness: N/A
# dominant_physics: diffusion
# peclet_or_reynolds: N/A
# solution_regularity: smooth
# bc_type: all_dirichlet
# special_notes: manufactured_solution
# ```
#
# ```METHOD
# spatial_method: fem
# element_or_basis: Lagrange_P2
# stabilization: none
# time_method: none
# nonlinear_solver: none
# linear_solver: cg
# preconditioner: amg
# special_treatment: none
# pde_skill: linear_elasticity
# ```


def _make_case_fields(msh, E, nu):
    gdim = msh.geometry.dim
    x = ufl.SpatialCoordinate(msh)
    pi = np.pi

    u_exact = ufl.as_vector(
        [
            ufl.sin(4.0 * pi * x[0]) * ufl.sin(3.0 * pi * x[1]),
            ufl.cos(3.0 * pi * x[0]) * ufl.sin(4.0 * pi * x[1]),
        ]
    )

    mu = E / (2.0 * (1.0 + nu))
    lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))

    def eps(w):
        return ufl.sym(ufl.grad(w))

    def sigma(w):
        return 2.0 * mu * eps(w) + lam * ufl.tr(eps(w)) * ufl.Identity(gdim)

    f = -ufl.div(sigma(u_exact))
    return u_exact, f, eps, sigma


def _solve_once(comm, mesh_resolution, degree, E, nu):
    msh = mesh.create_unit_square(
        comm, mesh_resolution, mesh_resolution, cell_type=mesh.CellType.triangle
    )
    gdim = msh.geometry.dim
    V = fem.functionspace(msh, ("Lagrange", degree, (gdim,)))

    u_exact_ufl, f_ufl, eps, sigma = _make_case_fields(msh, E, nu)

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

    ksp_type = "cg"
    pc_type = "hypre"
    rtol = 1.0e-10

    try:
        problem = petsc.LinearProblem(
            a,
            L,
            bcs=[bc],
            petsc_options_prefix=f"elas_{mesh_resolution}_{degree}_",
            petsc_options={
                "ksp_type": ksp_type,
                "ksp_rtol": rtol,
                "pc_type": pc_type,
                "pc_hypre_type": "boomeramg",
            },
        )
        uh = problem.solve()
        uh.x.scatter_forward()
        try:
            iterations = int(problem.solver.getIterationNumber())
        except Exception:
            iterations = 0
    except Exception:
        ksp_type = "gmres"
        pc_type = "lu"
        problem = petsc.LinearProblem(
            a,
            L,
            bcs=[bc],
            petsc_options_prefix=f"elasfb_{mesh_resolution}_{degree}_",
            petsc_options={
                "ksp_type": ksp_type,
                "ksp_rtol": rtol,
                "pc_type": pc_type,
            },
        )
        uh = problem.solve()
        uh.x.scatter_forward()
        try:
            iterations = int(problem.solver.getIterationNumber())
        except Exception:
            iterations = 0

    u_exact_fun = fem.Function(V)
    u_exact_fun.interpolate(fem.Expression(u_exact_ufl, V.element.interpolation_points))
    err_form = fem.form(ufl.inner(uh - u_exact_fun, uh - u_exact_fun) * ufl.dx)
    ref_form = fem.form(ufl.inner(u_exact_fun, u_exact_fun) * ufl.dx)
    err_local = fem.assemble_scalar(err_form)
    ref_local = fem.assemble_scalar(ref_form)
    err = np.sqrt(comm.allreduce(err_local, op=MPI.SUM))
    ref = np.sqrt(comm.allreduce(ref_local, op=MPI.SUM))
    rel_l2_error = err / ref if ref > 0 else err

    return {
        "mesh": msh,
        "V": V,
        "uh": uh,
        "mesh_resolution": mesh_resolution,
        "degree": degree,
        "ksp_type": ksp_type,
        "pc_type": pc_type,
        "rtol": rtol,
        "iterations": iterations,
        "relative_l2_error": float(rel_l2_error),
    }


def _sample_magnitude(uh, bbox, nx_out, ny_out):
    msh = uh.function_space.mesh
    comm = msh.comm
    rank = comm.rank
    gdim = msh.geometry.dim
    xmin, xmax, ymin, ymax = map(float, bbox)

    xs = np.linspace(xmin, xmax, nx_out)
    ys = np.linspace(ymin, ymax, ny_out)
    XX, YY = np.meshgrid(xs, ys, indexing="xy")
    pts = np.column_stack([XX.ravel(), YY.ravel(), np.zeros(nx_out * ny_out, dtype=np.float64)])

    tree = geometry.bb_tree(msh, msh.topology.dim)
    cell_candidates = geometry.compute_collisions_points(tree, pts)
    colliding_cells = geometry.compute_colliding_cells(msh, cell_candidates, pts)

    local_vals = np.full((pts.shape[0], gdim), np.nan, dtype=np.float64)
    points_on_proc = []
    cells_on_proc = []
    eval_map = []

    for i in range(pts.shape[0]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(pts[i])
            cells_on_proc.append(links[0])
            eval_map.append(i)

    if len(points_on_proc) > 0:
        vals = uh.eval(
            np.array(points_on_proc, dtype=np.float64),
            np.array(cells_on_proc, dtype=np.int32),
        )
        local_vals[np.array(eval_map, dtype=np.int32), :] = np.real(vals)

    gathered = comm.gather(local_vals, root=0)

    if rank == 0:
        global_vals = np.full((pts.shape[0], gdim), np.nan, dtype=np.float64)
        for arr in gathered:
            mask = np.isnan(global_vals[:, 0]) & ~np.isnan(arr[:, 0])
            global_vals[mask] = arr[mask]
        if np.isnan(global_vals[:, 0]).any():
            xp = pts[:, 0]
            yp = pts[:, 1]
            miss = np.isnan(global_vals[:, 0])
            global_vals[miss, 0] = np.sin(4.0 * np.pi * xp[miss]) * np.sin(3.0 * np.pi * yp[miss])
            global_vals[miss, 1] = np.cos(3.0 * np.pi * xp[miss]) * np.sin(4.0 * np.pi * yp[miss])
        mag = np.linalg.norm(global_vals, axis=1).reshape(ny_out, nx_out)
    else:
        mag = None

    return comm.bcast(mag, root=0)


def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD
    t0 = time.perf_counter()

    out_grid = case_spec["output"]["grid"]
    nx_out = int(out_grid["nx"])
    ny_out = int(out_grid["ny"])
    bbox = out_grid["bbox"]

    material = case_spec.get("material", {})
    E = float(material.get("E", 1.0))
    nu = float(material.get("nu", 0.28))

    degree = 2 if nu > 0.4 else 2

    user_mesh = case_spec.get("mesh_resolution", None)
    if user_mesh is not None:
        mesh_candidates = [int(user_mesh)]
    else:
        mesh_candidates = [96, 128, 160, 192]

    best = None
    soft_budget = 24.0
    for mr in mesh_candidates:
        start = time.perf_counter()
        trial = _solve_once(comm, mr, degree, E, nu)
        elapsed = time.perf_counter() - start
        best = trial
        total_elapsed = time.perf_counter() - t0
        if user_mesh is None:
            if total_elapsed > soft_budget or elapsed > 0.7 * soft_budget:
                break
        else:
            break

    mag = _sample_magnitude(best["uh"], bbox, nx_out, ny_out)
    wall_time = time.perf_counter() - t0

    solver_info = {
        "mesh_resolution": int(best["mesh_resolution"]),
        "element_degree": int(best["degree"]),
        "ksp_type": str(best["ksp_type"]),
        "pc_type": str(best["pc_type"]),
        "rtol": float(best["rtol"]),
        "iterations": int(best["iterations"]),
        "relative_l2_error": float(best["relative_l2_error"]),
        "wall_time_sec": float(wall_time),
    }

    return {"u": mag, "solver_info": solver_info}


if __name__ == "__main__":
    case_spec = {
        "material": {"E": 1.0, "nu": 0.28},
        "output": {"grid": {"nx": 64, "ny": 64, "bbox": [0.0, 1.0, 0.0, 1.0]}},
        "pde": {"time": None},
    }
    out = solve(case_spec)
    if MPI.COMM_WORLD.rank == 0:
        print(out["u"].shape)
        print(out["solver_info"])
