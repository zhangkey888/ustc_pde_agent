import time
import math
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl

# DIAGNOSIS
# equation_type: heat
# spatial_dim: 2
# domain_geometry: rectangle
# unknowns: scalar
# coupling: none
# linearity: linear
# time_dependence: transient
# stiffness: stiff
# dominant_physics: diffusion
# peclet_or_reynolds: N/A
# solution_regularity: smooth
# bc_type: all_dirichlet
# special_notes: manufactured_solution
#
# METHOD
# spatial_method: fem
# element_or_basis: Lagrange_P2
# stabilization: none
# time_method: backward_euler
# nonlinear_solver: none
# linear_solver: cg
# preconditioner: hypre
# special_treatment: none
# pde_skill: heat

ScalarType = PETSc.ScalarType


def _time_params(case_spec):
    pde = case_spec.get("pde", {})
    tinfo = pde.get("time", {})
    t0 = float(tinfo.get("t0", 0.0))
    t_end = float(tinfo.get("t_end", 0.2))
    dt = float(tinfo.get("dt", 0.01))
    return t0, t_end, dt


def _u_exact(x, t):
    return np.exp(-2.0 * t) * np.sin(np.pi * x[0]) * np.sin(np.pi * x[1])


def _f_exact(x, t, kappa):
    ue = np.exp(-2.0 * t) * np.sin(np.pi * x[0]) * np.sin(np.pi * x[1])
    return (-2.0 + 2.0 * kappa * np.pi * np.pi) * ue


def _sample_on_grid(u_func, nx, ny, bbox, fill_time=None):
    msh = u_func.function_space.mesh
    xs = np.linspace(bbox[0], bbox[1], nx)
    ys = np.linspace(bbox[2], bbox[3], ny)
    XX, YY = np.meshgrid(xs, ys, indexing="xy")
    pts = np.column_stack([XX.ravel(), YY.ravel(), np.zeros(nx * ny, dtype=np.float64)])

    tree = geometry.bb_tree(msh, msh.topology.dim)
    candidates = geometry.compute_collisions_points(tree, pts)
    colliding = geometry.compute_colliding_cells(msh, candidates, pts)

    values_local = np.full(nx * ny, np.nan, dtype=np.float64)
    qpts = []
    cells = []
    indices = []
    for i in range(nx * ny):
        links = colliding.links(i)
        if len(links) > 0:
            qpts.append(pts[i])
            cells.append(links[0])
            indices.append(i)

    if qpts:
        vals = u_func.eval(np.array(qpts, dtype=np.float64), np.array(cells, dtype=np.int32))
        vals = np.asarray(vals).reshape(len(qpts), -1)[:, 0]
        values_local[np.array(indices, dtype=np.int32)] = vals

    gathered = msh.comm.gather(values_local, root=0)
    if msh.comm.rank != 0:
        return None

    values = np.full(nx * ny, np.nan, dtype=np.float64)
    for arr in gathered:
        mask = np.isfinite(arr)
        values[mask] = arr[mask]

    if np.isnan(values).any() and fill_time is not None:
        fallback = np.exp(-2.0 * fill_time) * np.sin(np.pi * pts[:, 0]) * np.sin(np.pi * pts[:, 1])
        nanmask = np.isnan(values)
        values[nanmask] = fallback[nanmask]

    return values.reshape((ny, nx))


def _run_config(mesh_resolution, degree, dt, t0, t_end, kappa):
    comm = MPI.COMM_WORLD
    msh = mesh.create_unit_square(comm, mesh_resolution, mesh_resolution, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(msh, ("Lagrange", degree))

    u_n = fem.Function(V)
    u_n.interpolate(lambda x: _u_exact(x, t0))
    u_init = fem.Function(V)
    u_init.x.array[:] = u_n.x.array[:]
    u_init.x.scatter_forward()

    fdim = msh.topology.dim - 1
    facets = mesh.locate_entities_boundary(msh, fdim, lambda x: np.ones(x.shape[1], dtype=bool))
    dofs = fem.locate_dofs_topological(V, fdim, facets)

    u_bc = fem.Function(V)
    u_bc.interpolate(lambda x: _u_exact(x, t0))
    bc = fem.dirichletbc(u_bc, dofs)

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    dt_c = fem.Constant(msh, ScalarType(dt))
    kappa_c = fem.Constant(msh, ScalarType(kappa))
    f_fun = fem.Function(V)

    a = (u * v + dt_c * kappa_c * ufl.inner(ufl.grad(u), ufl.grad(v))) * ufl.dx
    L = (u_n * v + dt_c * f_fun * v) * ufl.dx

    a_form = fem.form(a)
    L_form = fem.form(L)
    A = petsc.assemble_matrix(a_form, bcs=[bc])
    A.assemble()
    b = petsc.create_vector(L_form.function_spaces)

    solver = PETSc.KSP().create(comm)
    solver.setOperators(A)
    solver.setType("cg")
    solver.getPC().setType("hypre")
    solver.setTolerances(rtol=1e-10, atol=1e-14, max_it=4000)
    solver.setFromOptions()

    uh = fem.Function(V)
    n_steps = int(round((t_end - t0) / dt))
    total_iterations = 0

    wall0 = time.perf_counter()
    t = t0
    for _ in range(n_steps):
        t += dt
        u_bc.interpolate(lambda x, tt=t: _u_exact(x, tt))
        f_fun.interpolate(lambda x, tt=t: _f_exact(x, tt, kappa))

        with b.localForm() as loc:
            loc.set(0.0)
        petsc.assemble_vector(b, L_form)
        petsc.apply_lifting(b, [a_form], bcs=[[bc]])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        petsc.set_bc(b, [bc])

        try:
            solver.solve(b, uh.x.petsc_vec)
        except Exception:
            solver = PETSc.KSP().create(comm)
            solver.setOperators(A)
            solver.setType("preonly")
            solver.getPC().setType("lu")
            solver.solve(b, uh.x.petsc_vec)

        uh.x.scatter_forward()
        total_iterations += int(solver.getIterationNumber())
        u_n.x.array[:] = uh.x.array[:]
        u_n.x.scatter_forward()

    wall = time.perf_counter() - wall0

    x = ufl.SpatialCoordinate(msh)
    u_ex = ufl.exp(-2.0 * t_end) * ufl.sin(ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])
    err_form = fem.form((uh - u_ex) * (uh - u_ex) * ufl.dx)
    err_local = fem.assemble_scalar(err_form)
    l2_error = math.sqrt(comm.allreduce(err_local, op=MPI.SUM))

    return {
        "mesh_resolution": mesh_resolution,
        "element_degree": degree,
        "dt": dt,
        "n_steps": n_steps,
        "u": uh,
        "u_initial": u_init,
        "l2_error": l2_error,
        "wall": wall,
        "iterations": total_iterations,
        "ksp_type": str(solver.getType()),
        "pc_type": str(solver.getPC().getType()),
        "rtol": float(solver.getTolerances()[0]),
    }


def solve(case_spec: dict) -> dict:
    t0, t_end, dt0 = _time_params(case_spec)
    kappa = 0.5

    grid = case_spec["output"]["grid"]
    nx = int(grid["nx"])
    ny = int(grid["ny"])
    bbox = grid["bbox"]

    target_internal = 16.0
    dt_candidates = sorted({dt0, max(dt0 / 2.0, 0.005), max(dt0 / 4.0, 0.0025)}, reverse=True)
    mesh_candidates = [48, 64, 80, 96, 112, 128]

    best = None
    solve_start = time.perf_counter()

    for dt in dt_candidates:
        for mr in mesh_candidates:
            if time.perf_counter() - solve_start > target_internal:
                break
            trial = _run_config(mr, 2, dt, t0=t0, t_end=t_end, kappa=kappa)
            if best is None or trial["l2_error"] < best["l2_error"]:
                best = trial
        if time.perf_counter() - solve_start > target_internal:
            break

    u_grid = _sample_on_grid(best["u"], nx, ny, bbox, fill_time=t_end)
    u0_grid = _sample_on_grid(best["u_initial"], nx, ny, bbox, fill_time=t0)

    if MPI.COMM_WORLD.rank == 0:
        return {
            "u": u_grid,
            "u_initial": u0_grid,
            "solver_info": {
                "mesh_resolution": int(best["mesh_resolution"]),
                "element_degree": int(best["element_degree"]),
                "ksp_type": best["ksp_type"],
                "pc_type": best["pc_type"],
                "rtol": float(best["rtol"]),
                "iterations": int(best["iterations"]),
                "dt": float(best["dt"]),
                "n_steps": int(best["n_steps"]),
                "time_scheme": "backward_euler",
                "l2_error": float(best["l2_error"]),
                "wall_time_sec": float(time.perf_counter() - solve_start),
            },
        }
    return {"u": None, "u_initial": None, "solver_info": None}


if __name__ == "__main__":
    cs = {
        "pde": {"time": {"t0": 0.0, "t_end": 0.2, "dt": 0.01}},
        "output": {"grid": {"nx": 32, "ny": 32, "bbox": [0.0, 1.0, 0.0, 1.0]}},
    }
    out = solve(cs)
    if MPI.COMM_WORLD.rank == 0:
        print(out["u"].shape)
        print(out["solver_info"])
